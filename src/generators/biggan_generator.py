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
    def __init__(self, num_gpu, latent_vector_size, ngf, num_channels, num_classes):
        super(BigganGenerator, self).__init__(num_gpu, latent_vector_size, ngf, num_channels, num_classes)
        self.n_gpu = num_gpu
        self.ngf = ngf
        embedding_size = 128
        self.embeddings = torch.nn.Embedding(num_classes, embedding_size)
        nn.init.orthogonal_(self.embeddings.weight)
        self.generator_layers = nn.ModuleList()

        latent_embed_vector_size = latent_vector_size + embedding_size
        self.initial_linear = spectral_norm(nn.Linear(in_features=latent_embed_vector_size,
                                                      out_features=4 * 4 * 16 * ngf))

        # Input is nfg*16 x 4 x 4 matrix
        # Output of ResUp: ngf*16 x 8 x 8
        self.generator_layers.append(ResUp(ngf * 16, ngf * 16, latent_embed_vector_size))

        # ngf*8 x 16 x 16
        self.generator_layers.append(ResUp(ngf * 16, ngf * 8, latent_embed_vector_size))

        # ngf*4 x 32 x 32
        self.generator_layers.append(ResUp(ngf * 8, ngf * 4, latent_embed_vector_size))

        self.generator_layers.append(NonLocalBlock(ngf * 4))

        # ngf*2 x 64 x 64
        self.generator_layers.append(ResUp(ngf * 4, ngf * 2, latent_embed_vector_size))

        # ngf x 128 x 128
        self.generator_layers.append(ResUp(ngf * 2, ngf, latent_embed_vector_size))

        self.batch_norm = nn.BatchNorm2d(num_features=ngf)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(ngf, 3, kernel_size=3, padding='same')
        self.tanh = nn.Tanh()

    def forward(self, latent_vector, labels):
        # [B, Z] - Z is size of latent vector
        batch_size = latent_vector.size(dim=0)

        # [B, embedding_size]
        embed_vector = self.embeddings(labels)
        latent_embed_vector = torch.concat((latent_vector, embed_vector), axis=1)

        # [B, 4*4*16*ngf]
        out = self.initial_linear(latent_embed_vector)

        # [B, 16 * ngf, 4, 4]
        out = torch.reshape(out, [batch_size, 16 * self.ngf, 4, 4])
        for generator_layer in self.generator_layers:
            out = generator_layer(out, latent_embed_vector)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.tanh(out)

        return out
