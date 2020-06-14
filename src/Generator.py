# -*- coding: utf-8 -*-
"""
Created on Thu May 11 00:23:38 2020

@author: Gerardo Cervantes

Purpose: The Generator class part of the GAN.  Customizable in the creation.
The class takes in a latent vector to generate new images (in the forward pass)
"""

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_gpu, latent_vector_size, ngf, num_channels):
        super(Generator, self).__init__()
        self.n_gpu = num_gpu
        self.main = nn.Sequential(
            # (in-1) * s - 2 * p + (1 * (k - 1) + 1
            # input is latent vector of given size (n_channels, 1, 1)
            nn.ConvTranspose2d(latent_vector_size, ngf * 8, kernel_size=(4, 5), stride=1, padding=0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 4 x 5
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=(4, 5), stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 8 x 11
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 16 x 22
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=(5, 4), stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 33 x 44
            nn.ConvTranspose2d(ngf, num_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # state size: (nc) x 66 x 88
        )

    def forward(self, generator_input):
        return self.main(generator_input)

