# -*- coding: utf-8 -*-
"""
Created on Thu May 11 00:23:38 2020

@author: Gerardo Cervantes

Purpose: The Generator class part of the GAN.  Customizable in the creation.
The class takes in a latent vector to generate new images (in the forward pass)
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from src.models.base_generator import BaseGenerator
from src import create_model


class DcganGenerator(BaseGenerator):
    def __init__(self, num_gpu, base_width, base_height, upsample_layers, latent_vector_size, ngf, num_channels,
                 num_classes):
        super(DcganGenerator, self).__init__(num_gpu, base_width, base_height, upsample_layers, latent_vector_size,
                                             ngf, num_channels, num_classes)
        self.n_gpu = num_gpu
        self.generator_layers = nn.ModuleList()

        embedding_size = 32
        self.embeddings = torch.nn.Embedding(num_classes, embedding_size)
        nn.init.orthogonal_(self.embeddings.weight)

        latent_embed_vector_size = latent_vector_size + embedding_size
        if upsample_layers == 5:
            conv_channels = [latent_embed_vector_size, ngf * 16, ngf * 8, ngf * 4, ngf * 2, ngf, num_channels]
        else:
            raise NotImplementedError(str(upsample_layers) + ' layers for dcgan generator is not supported.  You'
                                                             ' can either use a different amount of layers, or make a'
                                                             ' list with the channels you want with those layers')

        # input of conv is size (latent_embed_vector_size, 1, 1)
        self.generator_layers.append(spectral_norm(nn.ConvTranspose2d(conv_channels[0], conv_channels[1],
                                                                      kernel_size=(base_height, base_width),
                                                                      stride=1, padding=0, bias=False)))
        self.generator_layers.append(nn.BatchNorm2d(conv_channels[1]))
        self.generator_layers.append(nn.ReLU(False))
        # Output of this should be base_channels
        previous_out_channel = conv_channels[1]
        for i, layer_channel in enumerate(conv_channels[2:]):
            is_last_layer = i == len(conv_channels[2:]) - 1
            self.generator_layers.append(spectral_norm(nn.ConvTranspose2d(previous_out_channel, layer_channel,
                                                                          kernel_size=4, stride=2, padding=1,
                                                                          bias=is_last_layer)))
            if is_last_layer:
                self.generator_layers.append(nn.Tanh())
            else:
                self.generator_layers.append(nn.BatchNorm2d(layer_channel))
                self.generator_layers.append(nn.ReLU(False))

            previous_out_channel = layer_channel

        self.apply(create_model.weights_init)

    def forward(self, latent_vector, labels):

        embed_vector = self.embeddings(labels)
        latent_embed_vector = torch.concat((latent_vector, embed_vector), axis=1)

        # latent_embed_vector is of size (B, latent_embed_vector_size)
        latent_embed_vector = torch.unsqueeze(latent_embed_vector, -1)
        out = torch.unsqueeze(latent_embed_vector, -1)
        # out is of size (B, latent_embed_vector_size, 1, 1)

        for i, generator_layer in enumerate(self.generator_layers):
            out = generator_layer(out)
        return out
