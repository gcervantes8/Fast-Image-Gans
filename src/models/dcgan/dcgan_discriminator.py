# -*- coding: utf-8 -*-
"""
Created on Thu May 11 00:23:38 2020

@author: Gerardo Cervantes

Purpose: The Discriminator class part of the GAN.  Customizable in the creation.
The class takes in images to classify whether the images are real or fake (generated)
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from src.models.base_discriminator import BaseDiscriminator
from src.models import create_model


class DcganDiscriminator(BaseDiscriminator):
    def __init__(self, base_width, base_height, upsample_layers, ndf, num_channels, num_classes, output_size: int,
                 project_labels: bool):
        super(DcganDiscriminator, self).__init__(base_width, base_height, upsample_layers, ndf, num_channels,
                                                 num_classes)

        if output_size > 1:
            raise NotImplementedError('Having an output greater than 1 is not currently supported for dcgan')
        # Input is Batch_size x 3 x image_width x image_height matrix
        self.discrim_layers = nn.ModuleList()

        embedding_size = 32
        self.embeddings = nn.Embedding(num_classes, embedding_size)
        nn.init.orthogonal_(self.embeddings.weight)

        self.image_height, self.image_width = base_height * (2 ** upsample_layers), base_width * (2 ** upsample_layers)
        self.fc_layer = spectral_norm(nn.Linear(in_features=embedding_size,
                                                out_features=self.image_height * self.image_width))
        nn.init.orthogonal_(self.fc_layer.weight)

        if upsample_layers == 5:
            conv_channels = [num_channels + 1, ndf, ndf * 2, ndf * 4, ndf * 8, ndf * 16, 1]
        else:
            raise NotImplementedError(str(upsample_layers) + ' layers for dcgan discriminator is not supported.  You'
                                                             ' can either use a different amount of layers, or make a'
                                                             ' list with the channels you want with those layers')

        self.discrim_layers.append(nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1))
        self.discrim_layers.append(nn.LeakyReLU(0.1, inplace=False))
        previous_out_channel = conv_channels[1]
        for i, layer_channel in enumerate(conv_channels[2:]):
            is_last_layer = i == len(conv_channels[2:]) - 1

            if is_last_layer:
                self.discrim_layers.append(spectral_norm(nn.Conv2d(previous_out_channel, layer_channel,
                                                                   kernel_size=(base_height, base_width), stride=1)))
                self.discrim_layers.append(nn.Sigmoid())
            else:
                self.discrim_layers.append(spectral_norm(nn.Conv2d(previous_out_channel, layer_channel,
                                                                   kernel_size=4, stride=2, padding=1)))
                self.discrim_layers.append(nn.LeakyReLU(0.1, inplace=False))
            previous_out_channel = layer_channel

        self.apply(create_model.weights_init)

    def forward(self, discriminator_input, labels):
        batch_size = discriminator_input.size(dim=0)
        # discriminator_input is of size (B, channels, width, height)

        embed_vector = self.embeddings(labels)
        # out is (B, image_height * image_width)
        out = self.fc_layer(embed_vector)

        out = torch.reshape(out, [batch_size, 1, self.image_height, self.image_width])

        # out is of size (B, channels+1, width, height)
        out = torch.concat((discriminator_input, out), axis=1)

        for discrim_layer in self.discrim_layers:
            out = discrim_layer(out)

        return torch.squeeze(out)
