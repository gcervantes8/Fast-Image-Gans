# -*- coding: utf-8 -*-
"""
Created on Thu May 17

@author: Gerardo Cervantes

Purpose: Functions that are used to create/init the GAN model
"""

import torch
import torch.nn as nn

from src import Generator, Discriminator


# Creates the generator and discriminator using the configuration file
def create_gan_instances(model_arch_config, num_channels, n_gpus=0):

    latent_vector_size = int(model_arch_config['latent_vector_size'])
    ngf = int(model_arch_config['ngf'])
    ndf = int(model_arch_config['ndf'])

    device = torch.device('cuda:0' if (torch.cuda.is_available() and n_gpus > 0) else 'cpu')

    # Create the generator and discriminator
    generator = Generator.Generator(n_gpus, latent_vector_size, ngf, num_channels).to(device)
    discriminator = Discriminator.Discriminator(n_gpus, ndf, num_channels).to(device)

    generator = _handle_multiple_gpus(generator, n_gpus, device)
    discriminator = _handle_multiple_gpus(discriminator, n_gpus, device)
    return generator, discriminator, device


# Handle multi-gpu if desired, returns the new instance that is multi-gpu capable
def _handle_multiple_gpus(torch_obj, num_gpu, device):
    if (device.type == 'cuda') and (num_gpu > 1):
        return nn.DataParallel(torch_obj, list(range(num_gpu)))
    else:
        return torch_obj


# custom weights initialization, used by the generator and discriminator
def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
