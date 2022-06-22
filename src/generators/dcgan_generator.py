# -*- coding: utf-8 -*-
"""
Created on Thu May 11 00:23:38 2020

@author: Gerardo Cervantes

Purpose: The Generator class part of the GAN.  Customizable in the creation.
The class takes in a latent vector to generate new images (in the forward pass)
"""

import torch.nn as nn
from torch.nn.utils import spectral_norm
from src.generators.base_generator import BaseGenerator


class DcganGenerator(BaseGenerator):
    def __init__(self, num_gpu, latent_vector_size, ngf, num_channels):
        super(DcganGenerator, self).__init__(num_gpu, latent_vector_size, ngf, num_channels)
        self.n_gpu = num_gpu
        self.main = nn.Sequential(
            # (in-1) * s - 2 * p + (1 * (k - 1) + 1
            # input is latent vector of given size (n_channels, 1, 1)
            spectral_norm(nn.ConvTranspose2d(latent_vector_size, ngf * 16, kernel_size=3, stride=1, padding=0)),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size: (ngf*8) x 3 x 3
            spectral_norm(nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=(3, 4), stride=2, padding=1)),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*4) x 5 x 6
            spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*2) x 9 x 11
            spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(3, 4), stride=2, padding=1)),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 17 x 22
            spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=(3, 4), stride=2, padding=1)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 33 x 44
            spectral_norm(nn.ConvTranspose2d(ngf, num_channels, kernel_size=3, stride=2, padding=1)),
            nn.Tanh()
            # state size: (nc) x 65 x 87
        )

    def forward(self, generator_input):
        if generator_input.is_cuda and self.n_gpu > 1:
            output = nn.parallel.data_parallel(self.main, generator_input, range(self.n_gpu))
        else:
            output = self.main(generator_input)
        return output
