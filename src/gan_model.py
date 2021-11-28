# -*- coding: utf-8 -*-
"""
Created on Thu June 26 11:00:10 2020

@author: Gerardo Cervantes

Purpose: This will handle having both the generator and discriminator, as well as training the models given the data
"""

import torch
import torch.nn as nn
import torch.optim as optim


class GanModel:

    criterion = nn.BCELoss()
    fake_label, real_label = 0, 1

    def __init__(self, generator, discriminator, device, config):
        self.netG = generator
        self.netD = discriminator
        self.device = device
        self.latent_vector_size = int(config['CONFIGS']['latent_vector_size'])
        beta1 = float(config['CONFIGS']['beta1'])
        lr = float(config['CONFIGS']['lr'])

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    def update_minimax(self, real_data):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        self.netD.zero_grad()

        b_size = real_data.size(0)
        real_label = torch.full((b_size,), GanModel.real_label, dtype=real_data.dtype, device=self.device)

        # Note discriminator output must be 1 integer, or else criterion will throw an error
        discrim_output = self.netD(real_data)
        errD_real = GanModel.criterion(discrim_output, real_label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = discrim_output.mean().item()

        # Train with all-fake batch
        noise = torch.randn(b_size, self.latent_vector_size, 1, 1, device=self.device)
        # Generate fake image batch with G
        fake = self.netG(noise)
        fake_label = torch.full((b_size,), GanModel.fake_label, dtype=real_data.dtype, device=self.device)
        # Classify all fake batch with D
        fake_output = self.netD(fake.detach())
        # Calculate D's loss on the all-fake batch
        errD_fake = GanModel.criterion(fake_output, fake_label.reshape_as(fake_output))
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = fake_output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.netG.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output_update = self.netD(fake)
        errG = GanModel.criterion(output_update, real_label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output_update.mean().item()
        # Update G
        self.optimizerG.step()
        return errD, errG, D_x, D_G_z1, D_G_z2

    def generate_images(self, noise):
        with torch.no_grad():
            fake = self.netG(noise).detach().cpu()
        return fake
