# -*- coding: utf-8 -*-
"""
Created on Thu May 11 00:23:38 2020

@author: Gerardo Cervantes

Purpose: The Discriminator class part of the GAN.  Customizable in the creation.
The class takes in images to classify whether the images are real or fake (generated)
"""

import torch.nn as nn
from torch.nn.utils import spectral_norm
from src.discriminators.base_discriminator import BaseDiscriminator


class DcganDiscriminator(BaseDiscriminator):
    def __init__(self, num_gpu, ndf, num_channels):
        super(DcganDiscriminator, self).__init__(num_gpu, ndf, num_channels)
        self.n_gpu = num_gpu
        self.main = nn.Sequential(

            # input is (num_channels) x 65 x 87 (height goes first, when specifying tuples)
            spectral_norm(nn.Conv2d(num_channels, ndf, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # When dilation and padding is 1: ((in + 2p - (k - 1) - 1) / s) + 1

            # state: (ndf*2) x 33 x 44
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=(3, 4), stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # state: (ndf*4) x 17 x 22
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(3, 4), stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # state:  (ndf*4) x 9 x 11
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # state:  (ndf*8) x 5 x 6
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, kernel_size=(3, 4), stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # state:  (ndf*8) x 3 x 3
            spectral_norm(nn.Conv2d(ndf * 16, 1, kernel_size=3, stride=1)),
            # Output is 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, discriminator_input):
        # Expands input by 2 dimensions
        discriminator_input.expand(-1, -1, 1, 1)
        if discriminator_input.is_cuda and self.n_gpu > 1:
            output = nn.parallel.data_parallel(self.main, discriminator_input, range(self.n_gpu))
        else:
            output = self.main(discriminator_input)

        return output.view(-1, 1).squeeze(1)
