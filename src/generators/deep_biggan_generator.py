# -*- coding: utf-8 -*-
"""

@author: Gerardo Cervantes

Purpose: The Generator part of BigGan.  Customizable in the creation.
The class takes in images to classify whether the images are real or fake (generated)
"""

import torch
import torch.nn as nn
from src.generators.base_generator import BaseGenerator
from src.generators.deep_res_up import DeepResUp
from src.layers.nonlocal_block import NonLocalBlock
from torch.nn.utils.parametrizations import spectral_norm


class DeepBigganGenerator(BaseGenerator):
    def __init__(self, num_gpu, latent_vector_size, ngf, num_channels):
        super(DeepBigganGenerator, self).__init__(num_gpu, latent_vector_size, ngf, num_channels)
        self.n_gpu = num_gpu
        self.ngf = ngf

        self.generator_layers = nn.ModuleList()
        self.initial_linear = spectral_norm(nn.Linear(in_features=latent_vector_size, out_features=4 * 4 * 16 * ngf))

        # Input should be: [B, ngf * 16, 4, 4]
        self.generator_layers.append(DeepResUp(ngf * 16, ngf * 16, upsample=False))

        # [B, ngf * 16, 8, 8]
        self.generator_layers.append(DeepResUp(ngf * 16, ngf * 16))
        self.generator_layers.append(DeepResUp(ngf * 16, ngf * 16, upsample=False))

        # [B, ngf * 8, 16, 16]
        self.generator_layers.append(DeepResUp(ngf * 16, ngf * 8))
        self.generator_layers.append(DeepResUp(ngf * 8, ngf * 8, upsample=False))

        # [B, ngf * 4, 32, 32]
        self.generator_layers.append(DeepResUp(ngf * 8, ngf * 4))

        self.generator_layers.append(DeepResUp(ngf * 4, ngf * 4, upsample=False))

        self.generator_layers.append(NonLocalBlock(ngf * 4))
        # [B, ngf * 2, 64, 64]
        self.generator_layers.append(DeepResUp(ngf * 4, ngf * 2))

        self.generator_layers.append(DeepResUp(ngf * 2, ngf * 2, upsample=False))
        # [B, ngf, 128, 128]
        self.generator_layers.append(DeepResUp(ngf * 2, ngf))

        self.generator_layers.append(nn.BatchNorm2d(num_features=ngf, eps=1e-04))
        self.generator_layers.append(nn.ReLU())
        self.generator_layers.append(spectral_norm(nn.Conv2d(ngf, 3, kernel_size=3, padding='same')))
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
