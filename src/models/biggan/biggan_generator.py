# -*- coding: utf-8 -*-
"""

@author: Gerardo Cervantes

Purpose: The Generator part of BigGan.  Customizable in the creation.
The class takes in images to classify whether the images are real or fake (generated)
"""

import torch
import torch.nn as nn
from src.models.base_generator import BaseGenerator
from src.models.biggan.res_up import ResUp
from src.layers.nonlocal_block import NonLocalBlock
from torch.nn.utils.parametrizations import spectral_norm


class BigganGenerator(BaseGenerator):
    def __init__(self, num_gpu, base_width, base_height, upsample_layers, latent_vector_size, ngf, num_channels,
                 num_classes):
        super(BigganGenerator, self).__init__(num_gpu, base_width, base_height, upsample_layers, latent_vector_size,
                                              ngf, num_channels, num_classes)
        self.n_gpu = num_gpu
        self.ngf = ngf
        self.base_width, self.base_height = base_width, base_height
        # Embedding size of 128 is used for the biggan and deep-biggan paper
        embedding_size = 128
        self.embeddings = torch.nn.Embedding(num_classes, embedding_size)
        nn.init.orthogonal_(self.embeddings.weight)

        self.residual_layers = nn.ModuleList()

        if upsample_layers == 5:
            layer_channels = [ngf * 16, ngf * 16, ngf * 8, ngf * 4, ngf * 2, ngf]
            self.nonlocal_block_index = 3
            n_layers = 6
        else:
            raise NotImplementedError(str(upsample_layers) + ' layers for biggan discriminator is not supported.  You'
                                                             ' can either use a different amount of layers, or make a'
                                                             ' list with the channels you want with those layers')

        self.z_split_sizes = self.z_vector_split(latent_vector_size, n_layers)
        self.initial_linear = spectral_norm(nn.Linear(in_features=self.z_split_sizes[0],
                                                      out_features=base_width * base_height * 16 * ngf), eps=1e-04)
        nn.init.orthogonal_(self.initial_linear.weight)

        self.residual_layers.append(ResUp(layer_channels[0], layer_channels[1], self.z_split_sizes[1] + embedding_size))
        previous_out_channel = layer_channels[1]
        for i, layer_channel in enumerate(layer_channels[2:]):
            if self.nonlocal_block_index == i:
                self.nonlocal_block = NonLocalBlock(previous_out_channel)
            self.residual_layers.append(ResUp(previous_out_channel, layer_channel,
                                              self.z_split_sizes[i + 2] + embedding_size))
            previous_out_channel = layer_channel

        self.batch_norm = nn.BatchNorm2d(num_features=ngf)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(ngf, 3, kernel_size=3, padding='same')
        nn.init.orthogonal_(self.conv.weight)
        self.tanh = nn.Tanh()

    # Should always return list of size n_layers unless invalid input
    def z_vector_split(self, latent_vector_size, n_layers):
        temp_size_left = latent_vector_size
        z_split_sizes = []
        for split_size in range(n_layers, 0, -1):
            z_split_sizes.append(int(temp_size_left/split_size))
            temp_size_left = temp_size_left - int(temp_size_left/split_size)
        return z_split_sizes

    def forward(self, latent_vector, labels):
        # [B, Z] - Z is size of latent vector
        batch_size = latent_vector.size(dim=0)

        # [B, embedding_size]
        embed_vector = self.embeddings(labels)
        z_splits = torch.split(latent_vector, self.z_split_sizes, dim=1)

        # [B, 4*4*16*ngf]
        out = self.initial_linear(z_splits[0])

        # [B, 16 * ngf, 4, 4]
        out = torch.reshape(out, [batch_size, 16 * self.ngf, self.base_height, self.base_width])
        for i, generator_layer in enumerate(self.residual_layers):
            conditioning_vector = torch.concat((z_splits[i+1], embed_vector), axis=1)
            out = generator_layer(out, conditioning_vector)
            if i == self.nonlocal_block_index:
                out = self.nonlocal_block(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.tanh(out)

        return out
