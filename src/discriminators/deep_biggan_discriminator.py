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
    def __init__(self, num_gpu, base_width, base_height, upsample_layers, ndf, num_channels, num_classes):
        super(DeepBigganDiscriminator, self).__init__(num_gpu, base_width, base_height, upsample_layers, ndf,
                                                      num_channels, num_classes)
        self.n_gpu = num_gpu
        self.base_width, self.base_height = base_width, base_height
        # [B, ndf, image_width, image_height]
        initial_conv = spectral_norm(nn.Conv2d(3, ndf, kernel_size=3, padding='same'), eps=1e-04)
        nn.init.orthogonal_(initial_conv.weight)

        n_upsample_layers = 5
        if n_upsample_layers == 5:
            residual_channels = [ndf, ndf * 2, ndf * 2, ndf * 4, ndf * 4, ndf * 8, ndf * 8, ndf * 16, ndf * 16,
                                 ndf * 16, ndf * 16]
            downsample_layers = [True, False, True, False, True, False, True, False, True, False]
            nonlocal_block_index = 1
        else:
            raise NotImplementedError(str(n_upsample_layers) + ' layers for biggan discriminator is not supported.  You'
                                                               ' can either use a different amount of layers, or make a'
                                                               ' list with the channels you want with those layers')

        # Input is Batch_size x 3 x image_width x image_height matrix
        self.discrim_layers = nn.ModuleList()

        self.discrim_layers.append(initial_conv)

        self.discrim_layers.append(DeepResDown(residual_channels[0], residual_channels[1], pooling=downsample_layers[0]))
        previous_out_channel = residual_channels[1]
        for i, layer_channel in enumerate(residual_channels[2:]):
            if nonlocal_block_index == i:
                self.discrim_layers.append(NonLocalBlock(previous_out_channel))
            self.discrim_layers.append(DeepResDown(previous_out_channel, layer_channel, pooling=downsample_layers[i+1]))
            previous_out_channel = layer_channel

        # [B, ndf * 16, base_width, base_height]
        self.discrim_layers.append(nn.ReLU())
        self.embeddings = torch.nn.Embedding(num_classes, ndf * 16)
        nn.init.orthogonal_(self.embeddings.weight)
        # Fully connected layer
        self.fc_layer = spectral_norm(nn.Linear(in_features=ndf * 16, out_features=1), eps=1e-04)
        nn.init.orthogonal_(self.fc_layer.weight)

    def forward(self, discriminator_input, labels):
        out = discriminator_input
        for discrim_layer in self.discrim_layers:
            out = discrim_layer(out)
        # ndf * 16 - Global Sum Pooling
        out = torch.sum(out, dim=[2, 3])
        # 1 - Fully connected layer
        fc_out = self.fc_layer(out)
        fc_out = torch.squeeze(fc_out, dim=1)
        # embed_vector is of size [B, ndf*16]
        embed_vector = self.embeddings(labels)
        # TODO not sure if sum is needed
        out = fc_out + torch.sum(torch.mul(embed_vector, out), 1)
        return out
