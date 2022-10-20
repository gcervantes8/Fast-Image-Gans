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
from torchinfo import summary

from src import os_helper, data_load
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
        cpu_enabled = self.device.type == 'cpu'
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision and not cpu_enabled)
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
        self.ema = ExponentialMovingAverage(generator.parameters(),
                                            decay=float(ema_decay)) if ema_enabled else None

    def update_minimax(self, real_data, labels):
        b_size = real_data.size(0)
        # device_type and dtype are only used in
        device_type, dtype = self.device.type, None
        if self.mixed_precision:
            if self.device.type == 'cpu':
                device_type, dtype = ('cpu', torch.bfloat16)
            else:
                device_type, dtype = ('cuda', torch.float16)

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
            # Generate fake image batch with G
            fake = self.netG(noise, labels)
            fake_label = torch.full((b_size,), self.fake_label, dtype=real_data.dtype, device=self.device)
            # Classify all fake batch with D
            fake_output = self.netD(fake.detach(), labels)
            # Calculate D's loss on the all-fake batch
            discrim_on_fake_error = self.criterion(fake_output, fake_label.reshape_as(fake_output))
            discrim_on_fake_error = discrim_on_fake_error / self.accumulation_iterations

        self.grad_scaler.scale(discrim_on_fake_error).backward()
        total_discrim_error = discrim_on_real_error + discrim_on_fake_error

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=self.mixed_precision):
            self.netD.requires_grad_(requires_grad=False)
            output_update = self.netD(fake, labels)
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
            self.netD.zero_grad(set_to_none=True)
            self.netG.zero_grad(set_to_none=True)

        return (total_discrim_error * self.accumulation_iterations).item(), \
               (generator_error * self.accumulation_iterations).item()

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

    # If class_embeddings is provided, then it will ignore labels to generate the images
    def generate_images(self, noise, labels, class_embeddings=None):
        device_type, dtype = ('cpu', torch.bfloat16) if self.device.type == 'cpu' else ('cuda', torch.float16)
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=self.mixed_precision):
            with torch.no_grad():
                if self.ema:
                    with self.ema.average_parameters():
                        if class_embeddings:
                            forward_with_class_embeddings = getattr(self.netG, "forward_with_class_embeddings", None)
                            if callable(forward_with_class_embeddings):
                                fake = forward_with_class_embeddings(noise, class_embeddings)
                            else:
                                raise ValueError('Given generator model does not support providing class embeddings '
                                                 'directly.')
                        else:
                            fake = self.netG(noise, labels)
                else:
                    fake = self.netG(noise, labels)
            return fake.detach().cpu()

    def save(self, model_dir, step_num):
        generator_path = os.path.join(model_dir, os_helper.ModelType.GENERATOR.value + '_step_' + str(step_num) + '.pt')
        discrim_path = os.path.join(model_dir, os_helper.ModelType.DISCRIMINATOR.value + '_step_' +
                                    str(step_num) + '.pt')
        torch.save(self.netG.state_dict(), generator_path)
        torch.save(self.netD.state_dict(), discrim_path)
        if self.ema:
            ema_path = os.path.join(model_dir, os_helper.ModelType.EMA.value + '_step_' + str(step_num) + '.pt')
            torch.save(self.ema.state_dict(), ema_path)

    def load(self, generator_path, discrim_path, ema_path=None):
        self.netG.load_state_dict(torch.load(generator_path))
        self.netD.load_state_dict(torch.load(discrim_path))
        if self.ema and ema_path:
            self.ema.load_state_dict(torch.load(ema_path))

    # Writes text file with information of the generator and discriminator instances
    def save_architecture(self, save_dir, data_config):

        image_height, image_width = data_load.get_image_height_and_width(data_config)
        discriminator_stats = summary(self.netD, input_data=[torch.zeros((1, 3, image_height, image_width)),
                                                             torch.zeros(1, dtype=torch.int64)], verbose=0,
                                      device=self.device)
        generator_stats = summary(self.netG, input_data=[torch.zeros(1, self.latent_vector_size),
                                                         torch.zeros(1, dtype=torch.int64)], verbose=0,
                                  device=self.device)

        with open(os.path.join(save_dir, 'architecture.txt'), 'w', encoding='utf-8') as text_file:
            text_file.write('Generator\n\n')
            text_file.write(str(generator_stats))
            text_file.write(str(self.netG))
            text_file.write('\n\nDiscriminator\n\n')
            text_file.write(str(discriminator_stats))
            text_file.write(str(self.netD))
