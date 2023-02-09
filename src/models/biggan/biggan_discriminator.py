# -*- coding: utf-8 -*-
"""

@author: Gerardo Cervantes

Purpose: The Discriminator part of the BigGan.  Customizable in the creation.
The class takes in images to classify whether the images are real or fake (generated)
"""

import torch
import torch.nn as nn
from src.models.base_discriminator import BaseDiscriminator
from src.models.biggan.res_down import ResDown
from src.layers.nonlocal_block import NonLocalBlock
from torch.nn.utils.parametrizations import spectral_norm


class BigganDiscriminator(BaseDiscriminator):
    def __init__(self, base_width: int, base_height: int, upsample_layers: int, ndf: int, num_channels: int, 
                 num_classes: int, output_size: int, project_labels: bool):
        super(BigganDiscriminator, self).__init__(base_width, base_height, upsample_layers, ndf, num_channels,
                                                  num_classes)
        if upsample_layers == 5:
            layer_channels = [3, ndf, ndf * 2, ndf * 4, ndf * 8, ndf * 16, ndf * 16]
            downsample_layers = [True, True, True, True, True, False]
            nonlocal_block_index = 1
        else:
            raise NotImplementedError(str(upsample_layers) + ' layers for biggan discriminator is not supported.  You'
                                                             ' can either use a different amount of layers, or make a'
                                                             ' list with the channels you want with those layers')

        if project_labels and output_size > 1:
            raise NotImplementedError('Projecting labels and having an output greater than 1 currently not supported')                                                         
        # Input is Batch_size x 3 x image_width x image_height matrix
        self.discrim_layers = nn.ModuleList()

        self.discrim_layers.append(ResDown(layer_channels[0], layer_channels[1], pooling=downsample_layers[0]))
        previous_out_channel = layer_channels[1]
        for i, layer_channel in enumerate(layer_channels[2:]):
            if nonlocal_block_index == i:
                self.discrim_layers.append(NonLocalBlock(previous_out_channel))
            self.discrim_layers.append(ResDown(previous_out_channel, layer_channel, pooling=downsample_layers[i+1]))
            previous_out_channel = layer_channel
        self.discrim_layers.append(nn.ReLU())

        self.project_labels = project_labels
        
        if self.project_labels:
            self.embeddings = torch.nn.Embedding(num_classes, ndf * 16)
            nn.init.orthogonal_(self.embeddings.weight)
        # Fully connected layer
        self.fc_layer = spectral_norm(nn.Linear(in_features=ndf*16, out_features=output_size), eps=1e-04)
        nn.init.orthogonal_(self.fc_layer.weight)

    # Output is of size [Batch size, output_size]
    def forward(self, discriminator_input, labels):
        out = discriminator_input
        for discrim_layer in self.discrim_layers:
            out = discrim_layer(out)
        # ndf * 16 - Global Sum Pooling
        out = torch.sum(out, dim=[2, 3])
        # Size, [B, output_size] - Fully connected layer
        fc_out = self.fc_layer(out)

        if self.project_labels:
            fc_out = torch.squeeze(fc_out, dim=1)
            # embed_vector is of size [B, ndf*16]
            embed_vector = self.embeddings(labels)
            # out is of size [B, ndf*16]
            # fc_out is of size
            # TODO not sure if sum is needed
            out = fc_out + torch.sum(torch.mul(embed_vector, out), 1)
            return out
        return fc_out
