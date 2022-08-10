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
import os


class GanModel:

    def __init__(self, generator, discriminator, device, model_arch_config, train_config):
        self.netG = generator
        self.netD = discriminator
        self.device = device
        self.latent_vector_size = int(model_arch_config['latent_vector_size'])
        beta1 = float(train_config['beta1'])
        beta2 = float(train_config['beta2'])
        generator_lr = float(train_config['generator_lr'])
        discriminator_lr = float(train_config['discriminator_lr'])
        self.accumulation_iterations = int(train_config['accumulation_iterations'])
        self.mixed_precision = bool(train_config['mixed_precision'])
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.batch_iterations = 0

        self.criterion, self.fake_label, self.real_label = supported_loss_functions(train_config['loss_function'])
        if self.criterion is None:
            raise ValueError("Loss values options: " + str(supported_losses()))

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(beta1, beta2))
        self.optimizerG = optim.Adam(generator.parameters(), lr=generator_lr, betas=(beta1, beta2))

    def update_minimax(self, real_data):
        batch_size = real_data.size(0)

        # Train discriminator with all-real batch
        with torch.autocast(device_type=str(self.device), dtype=torch.float16, enabled=self.mixed_precision):
            real_label_tensor = torch.full((batch_size,), self.real_label, device=self.device)
            discrim_output = self.netD(real_data)
            discrim_on_real_error = self.criterion(discrim_output, real_label_tensor)
            discrim_on_real_error = discrim_on_real_error / self.accumulation_iterations

        # Calculate gradients for D in backward pass
        self.grad_scaler.scale(discrim_on_real_error).backward()
        D_x = discrim_output.mean().item()

        # Train discriminator with a generated batch
        with torch.autocast(device_type=str(self.device), dtype=torch.float16, enabled=self.mixed_precision):
            # Generate fake image batch with G, training with a fake batch
            fake = self.netG(torch.randn(batch_size, self.latent_vector_size, 1, 1, device=self.device))
            fake_output = self.netD(fake.detach())

            # Calculate D's loss on the all-fake batch
            discrim_on_fake_error = self.criterion(fake_output, torch.full((batch_size,), self.fake_label, device=self.device).reshape_as(fake_output))
            discrim_on_fake_error = discrim_on_fake_error / self.accumulation_iterations

        # Calculate the gradients for this batch
        self.grad_scaler.scale(discrim_on_fake_error).backward()
        D_G_z1 = fake_output.mean().item()

        # ------ Generator ------
        # Since we just updated D, perform another forward pass of all-fake batch through D
        with torch.autocast(device_type=str(self.device), dtype=torch.float16, enabled=self.mixed_precision):
            output_update = self.netD(fake)
            generator_error = self.criterion(output_update, real_label_tensor)
            generator_error = generator_error / self.accumulation_iterations

        # Calculate gradients for G
        self.grad_scaler.scale(generator_error).backward()
        D_G_z2 = output_update.mean().item()

        self.batch_iterations += 1
        if self.batch_iterations % self.accumulation_iterations == 0:
            # Update generator and discriminator
            self.grad_scaler.step(self.optimizerG)
            self.grad_scaler.step(self.optimizerD)
            self.grad_scaler.update()
            # Reset the grad back to 0 after a step
            self.netG.zero_grad()
            self.netD.zero_grad()
            self.batch_iterations = 0

        # Add the gradients from the all-real and all-fake batches
        total_discrim_error = discrim_on_real_error + discrim_on_fake_error

        return total_discrim_error, generator_error, D_x, D_G_z1, D_G_z2

    def generate_images(self, noise):
        with torch.no_grad():
            fake = self.netG(noise).detach().cpu()
        return fake

    def save(self, model_dir, step_or_epoch_num):
        generator_path = os.path.join(model_dir, 'gen_epoch_' + str(step_or_epoch_num) + '.pt')
        discriminator_path = os.path.join(model_dir, 'discrim_epoch_' + str(step_or_epoch_num) + '.pt')
        torch.save(self.netG.state_dict(), generator_path)
        torch.save(self.netD.state_dict(), discriminator_path)
