# -*- coding: utf-8 -*-
"""

@author: Gerardo Cervantes

Purpose: The Generator part of BigGan.  Customizable in the creation.
The class takes in images to classify whether the images are real or fake (generated)
"""

import torch
import torch.nn as nn
from src.generators.base_generator import BaseGenerator
from src.generators.res_up import ResUp
from src.layers.nonlocal_block import NonLocalBlock
from torch.nn.utils.parametrizations import spectral_norm


class BigganGenerator(BaseGenerator):
    def __init__(self, num_gpu, latent_vector_size, ngf, num_channels):
        super(BigganGenerator, self).__init__(num_gpu, latent_vector_size, ngf, num_channels)
        self.n_gpu = num_gpu
        self.ngf = ngf

        self.generator_layers = nn.ModuleList()
        self.initial_linear = spectral_norm(nn.Linear(in_features=latent_vector_size, out_features=4 * 4 * 16 * ngf))

        # Input is nfg*16 x 4 x 4 matrix
        # Output of ResUp: ngf*16 x 8 x 8
        self.generator_layers.append(ResUp(ngf * 16, ngf * 16))

        # ngf*8 x 16 x 16
        self.generator_layers.append(ResUp(ngf * 16, ngf * 8))

        # ngf*4 x 32 x 32
        self.generator_layers.append(ResUp(ngf * 8, ngf * 4))

        # self.generator_layers.append(NonLocalBlock(ngf * 4))

        # ngf*2 x 64 x 64
        self.generator_layers.append(ResUp(ngf * 4, ngf * 2))

        # ngf x 128 x 128
        self.generator_layers.append(ResUp(ngf * 2, ngf))

        self.generator_layers.append(nn.BatchNorm2d(num_features=ngf))

        self.generator_layers.append(nn.ReLU())

        self.generator_layers.append(nn.Conv2d(ngf, 3, kernel_size=3, padding='same'))

        self.generator_layers.append(nn.Tanh())

    def forward(self, discriminator_input):
        # [B, Z, 1, 1] - Z is size of latent vector
        batch_size = discriminator_input.size(dim=0)

        # [B, Z]
        discriminator_input = torch.squeeze(discriminator_input)

        # [B, 4*4*16*ngf]
        out = self.initial_linear(discriminator_input)

        # [B, 16 * ngf, 4, 4]
        out = torch.reshape(out, [batch_size, 16 * self.ngf, 4, 4])
        for generator_layer in self.generator_layers:
            out = generator_layer(out)

        return out
