# -*- coding: utf-8 -*-
"""

@author: Gerardo Cervantes

Purpose: The Discriminator part of the BigGan.  Customizable in the creation.
The class takes in images to classify whether the images are real or fake (generated)
"""

import torch
import torch.nn as nn
from src.discriminators.base_discriminator import BaseDiscriminator
from src.discriminators.deep_res_down import DeepResDown
from src.layers.nonlocal_block import NonLocalBlock
from torch.nn.utils.parametrizations import spectral_norm


class DeepBigganDiscriminator(BaseDiscriminator):
    def __init__(self, num_gpu, ndf, num_channels):
        super(DeepBigganDiscriminator, self).__init__(num_gpu, ndf, num_channels)
        self.n_gpu = num_gpu
        # Input is Batch_size x 3 x 128 x 128 matrix
        self.discrim_layers = nn.ModuleList()

        # [B, ndf, 128, 128]
        self.discrim_layers.append(nn.Conv2d(3, ndf, kernel_size=3, padding='same'))
        # [B, ndf * 2, 64, 64]
        self.discrim_layers.append(DeepResDown(ndf, ndf * 2))

        self.discrim_layers.append(DeepResDown(ndf * 2, ndf * 2, pooling=False))

        # [B, ndf * 4, 32, 32]
        self.discrim_layers.append(DeepResDown(ndf * 2, ndf * 4))
        self.discrim_layers.append(DeepResDown(ndf * 4, ndf * 4, pooling=False))
        self.discrim_layers.append(NonLocalBlock(ndf * 4))
        # [B, ndf * 8, 16, 16]
        self.discrim_layers.append(DeepResDown(ndf * 4, ndf * 8))
        self.discrim_layers.append(DeepResDown(ndf * 8, ndf * 8, pooling=False))
        # [B, ndf * 16, 8, 8]
        self.discrim_layers.append(DeepResDown(ndf * 8, ndf * 16))
        self.discrim_layers.append(DeepResDown(ndf * 16, ndf * 16, pooling=False))
        # [B, ndf * 16, 4, 4]
        self.discrim_layers.append(DeepResDown(ndf * 16, ndf * 16))
        self.discrim_layers.append(DeepResDown(ndf * 16, ndf * 16, pooling=False))

        # [B, ndf * 16, 4, 4]
        self.discrim_layers.append(nn.ReLU())
        # [B, ndf * 16, 2, 2]
        self.discrim_layers.append(nn.AvgPool2d(kernel_size=2))

        # [B, ndf * 16 * 2 * 2]
        self.discrim_layers.append(nn.Flatten())

        # [B, 1]
        self.discrim_layers.append(spectral_norm(nn.Linear(in_features=ndf*16*2*2, out_features=1)))

    def forward(self, discriminator_input):
        out = discriminator_input
        for discrim_layer in self.discrim_layers:
            out = discrim_layer(out)
        out = torch.squeeze(out, dim=1)
        return out
