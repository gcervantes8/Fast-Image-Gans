# -*- coding: utf-8 -*-
"""
Created on Thu May 11 00:23:38 2020

@author: Gerardo Cervantes

Purpose: The Discriminator class part of the GAN.  Customizable in the creation.
The class takes in images to classify whether the images are real or fake (generated)
"""

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_gpu, ndf, num_channels):
        super(Discriminator, self).__init__()
        self.ngpu = num_gpu
        self.main = nn.Sequential(

            # input is (num_channels) x 66 x 88 (height goes first, when specifying tuples)
            nn.Conv2d(num_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # When dilation and padding is 1: ((in + 2p - (k - 1) - 1) / s) + 1

            # state: (ndf*2) x 33 x 44
            nn.Conv2d(ndf, ndf * 2, kernel_size=(3, 4), stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*4) x 17 x 22
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(3, 4), stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state:  (ndf*4) x 9 x 11
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state:  (ndf*8) x 4 x 5
            nn.Conv2d(ndf * 8, 1, kernel_size=(4, 5), stride=1),
            # Output is 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, discriminator_input):
        return self.main(discriminator_input)
