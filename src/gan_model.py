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

        self.criterion, self.fake_label, self.real_label = supported_loss_functions(train_config['loss_function'])
        if self.criterion is None:
            raise ValueError("Loss values options: " + str(supported_losses()))

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(beta1, beta2))
        self.optimizerG = optim.Adam(generator.parameters(), lr=generator_lr, betas=(beta1, beta2))

    def update_minimax(self, real_data):

        # Train with all-real batch
        self.netD.zero_grad()

        b_size = real_data.size(0)
        real_label_tensor = torch.full((b_size,), self.real_label, dtype=real_data.dtype, device=self.device)

        discrim_output = self.netD(real_data)
        discrim_on_real_error = self.criterion(discrim_output, real_label_tensor)

        # Calculate gradients for D in backward pass
        discrim_on_real_error.backward()
        D_x = discrim_output.mean().item()

        # Train with all-fake batch
        noise = torch.randn(b_size, self.latent_vector_size, 1, 1, device=self.device)
        # Generate fake image batch with G
        fake = self.netG(noise)
        fake_label = torch.full((b_size,), self.fake_label, dtype=real_data.dtype, device=self.device)
        # Classify all fake batch with D
        fake_output = self.netD(fake.detach())
        # Calculate D's loss on the all-fake batch
        discrim_on_fake_error = self.criterion(fake_output, fake_label.reshape_as(fake_output))
        # Calculate the gradients for this batch
        discrim_on_fake_error.backward()
        D_G_z1 = fake_output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        total_discrim_error = discrim_on_real_error + discrim_on_fake_error
        # Update D
        self.optimizerD.step()

        # ------ Generator ------
        self.netG.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output_update = self.netD(fake)
        generator_error = self.criterion(output_update, real_label_tensor)
        # Calculate gradients for G
        generator_error.backward()
        D_G_z2 = output_update.mean().item()
        # Update G
        self.optimizerG.step()
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
