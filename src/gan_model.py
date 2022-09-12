# -*- coding: utf-8 -*-
"""
Created on Thu June 26 11:00:10 2020

@author: Gerardo Cervantes

Purpose: This will handle having both the generator and discriminator, as well as training the models given the data
"""

import torch
import torch.optim as optim
from src.losses.loss_functions import supported_loss_functions
from src.losses.loss_functions import supported_losses
from torch_ema import ExponentialMovingAverage
import os


class GanModel:

    def __init__(self, generator, discriminator, num_classes, device, model_arch_config, train_config):
        self.netG = generator.to(device)
        self.netD = discriminator.to(device)
        self.num_classes = num_classes
        self.device = device
        self.latent_vector_size = int(model_arch_config['latent_vector_size'])
        beta1 = float(train_config['beta1'])
        beta2 = float(train_config['beta2'])
        generator_lr = float(train_config['generator_lr'])
        discriminator_lr = float(train_config['discriminator_lr'])
        self.orthogonal_value = float(model_arch_config['orthogonal_value'])
        self.accumulation_iterations = int(train_config['accumulation_iterations'])
        self.mixed_precision = train_config.getboolean('mixed_precision')
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.batch_iterations = 0

        self.criterion, self.fake_label, self.real_label = supported_loss_functions(train_config['loss_function'],
                                                                                    device=device)
        if self.criterion is None:
            raise ValueError("Loss values options: " + str(supported_losses()))

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(beta1, beta2))
        self.optimizerG = optim.Adam(generator.parameters(), lr=generator_lr, betas=(beta1, beta2))

        self.netD.zero_grad()
        self.netG.zero_grad()

        # Set EMA
        ema_enabled = model_arch_config.getboolean('generator_ema')
        ema_decay = model_arch_config['ema_decay']
        self.ema = ExponentialMovingAverage(generator.parameters(), decay=float(ema_decay)) if ema_enabled else None

    def update_minimax(self, real_data, labels, train_generator=True):
        b_size = real_data.size(0)
        device_type, dtype = ('cpu', torch.bfloat16) if self.device.type == 'cpu' else ('cuda', torch.float16)

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=self.mixed_precision):
            real_label = torch.full((b_size,), self.real_label, dtype=real_data.dtype, device=self.device)
            discrim_output = self.netD(real_data, labels)
            discrim_on_real_error = self.criterion(discrim_output, real_label)
            if not self.orthogonal_value == 0:
                discrim_on_real_error += self.apply_orthogonal_regularization(self.netD)
            discrim_on_real_error = discrim_on_real_error / self.accumulation_iterations

        self.grad_scaler.scale(discrim_on_real_error).backward()

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=self.mixed_precision):
            # Train with all-fake batch
            noise = torch.randn(b_size, self.latent_vector_size, device=self.device)
            random_class_labels = torch.randint(low=0, high=self.num_classes, size=[b_size], device=self.device, dtype=torch.int64)
            # Generate fake image batch with G
            fake = self.netG(noise, random_class_labels)
            fake_label = torch.full((b_size,), self.fake_label, dtype=real_data.dtype, device=self.device)
            # Classify all fake batch with D
            fake_output = self.netD(fake.detach(), random_class_labels)
            # Calculate D's loss on the all-fake batch
            discrim_on_fake_error = self.criterion(fake_output, fake_label.reshape_as(fake_output))
            discrim_on_fake_error = discrim_on_fake_error / self.accumulation_iterations

        self.grad_scaler.scale(discrim_on_fake_error).backward()
        total_discrim_error = discrim_on_real_error + discrim_on_fake_error

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=self.mixed_precision):
            self.netD.requires_grad_(requires_grad=False)
            output_update = self.netD(fake, random_class_labels)
            self.netD.requires_grad_(requires_grad=True)
            generator_error = self.criterion(output_update, real_label)
            if not self.orthogonal_value == 0:
                generator_error += self.apply_orthogonal_regularization(self.netG)
            generator_error = generator_error / self.accumulation_iterations
        self.grad_scaler.scale(generator_error).backward()

        self.batch_iterations += 1
        if self.batch_iterations % self.accumulation_iterations == 0:
            self.grad_scaler.step(self.optimizerD)
            self.grad_scaler.step(self.optimizerG)
            self.grad_scaler.update()
            if self.ema:
                self.ema.update()
            self.netD.zero_grad()
            self.netG.zero_grad()

        return total_discrim_error * self.accumulation_iterations, generator_error * self.accumulation_iterations

    def apply_orthogonal_regularization(self, model):
        model_orthogonal_loss = 0.0
        for name, param in model.named_parameters():
            if len(param.size()) >= 2 and param.requires_grad and 'conv' in name:
                param = torch.reshape(param, [param.size(dim=0), -1])
                mult_out = torch.mm(torch.t(param), param)
                orthogonal_matrix = torch.mul(mult_out, 1 - torch.eye(mult_out.size(dim=0), device=self.device))
                # Frobenius norm
                model_orthogonal_loss += self.orthogonal_value * torch.norm(orthogonal_matrix, p='fro')
        return model_orthogonal_loss

    def generate_images(self, noise, labels):
        with torch.no_grad():
            if self.ema:
                with self.ema.average_parameters():
                    fake = self.netG(noise, labels).detach().cpu()
            else:
                fake = self.netG(noise, labels).detach().cpu()
        return fake

    def save(self, model_dir, step_or_epoch_num):
        generator_path = os.path.join(model_dir, 'gen_epoch_' + str(step_or_epoch_num) + '.pt')
        discriminator_path = os.path.join(model_dir, 'discrim_epoch_' + str(step_or_epoch_num) + '.pt')
        torch.save(self.netG.state_dict(), generator_path)
        torch.save(self.netD.state_dict(), discriminator_path)
