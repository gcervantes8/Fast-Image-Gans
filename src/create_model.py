# -*- coding: utf-8 -*-
"""
Created on Thu May 17

@author: Gerardo Cervantes

Purpose: Functions that are used to create/init the GAN model
"""

import torch
import torch.nn as nn
from src import saver_and_loader, os_helper
from src.gan_model import GanModel

from src.models.dcgan.dcgan_generator import DcganGenerator
from src.models.biggan.biggan_generator import BigganGenerator
from src.models.deep_biggan.deep_biggan_generator import DeepBigganGenerator

from src.models.dcgan.dcgan_discriminator import DcganDiscriminator
from src.models.biggan.biggan_discriminator import BigganDiscriminator
from src.models.deep_biggan.deep_biggan_discriminator import DeepBigganDiscriminator


def create_gen_and_discrim(model_name: str):
    model_name = model_name.lower()
    models_supported = {
        'dcgan': (DcganGenerator, DcganDiscriminator),
        'biggan': (BigganGenerator, BigganDiscriminator),
        'deep_biggan': (DeepBigganGenerator, DeepBigganDiscriminator),
    }
    if model_name not in models_supported:
        raise ValueError("Given model name in config file is not supported\n" +
                         'Supported models: ' + str(list(models_supported.keys())))
    return models_supported[model_name]


def get_device(n_gpus):
    device = torch.device('cuda' if (torch.cuda.is_available() and n_gpus > 0) else 'cpu')
    return device


# Creates the generator and discriminator using the configuration file
def create_gan_instances(model_arch_config, data_config, device, num_classes=1, n_gpus=0):

    model_type = model_arch_config['model_type']
    num_channels = int(data_config['num_channels'])
    latent_vector_size = int(model_arch_config['latent_vector_size'])
    ngf = int(model_arch_config['ngf'])
    ndf = int(model_arch_config['ndf'])
    base_width = int(data_config['base_width'])
    base_height = int(data_config['base_height'])
    upsample_layers = int(data_config['upsample_layers'])
    generator, discriminator = create_gen_and_discrim(model_type)
    # Create the generator and discriminator
    generator = generator(n_gpus, base_width, base_height, upsample_layers, latent_vector_size, ngf, num_channels,
                          num_classes).to(device)
    discriminator = discriminator(n_gpus, base_width, base_height, upsample_layers, ndf, num_channels,
                                  num_classes).to(device)

    generator = _handle_multiple_gpus(generator, n_gpus, device)
    discriminator = _handle_multiple_gpus(discriminator, n_gpus, device)
    return generator, discriminator


def create_gan_model(run_dir, model_arch_config, data_config, train_config, num_classes, device, n_gpus):

    generator, discriminator = create_gan_instances(model_arch_config, data_config, device,
                                                    num_classes=num_classes, n_gpus=n_gpus)
    saver_and_loader.save_architecture(generator, discriminator, run_dir, data_config, model_arch_config)
    return _to_gan_model(generator, discriminator, num_classes, device, model_arch_config, train_config)


def restore_model(model_dir, model_arch_config, train_config, data_config, num_classes, device):
    generator_path, discrim_path, step_num = os_helper.find_latest_generator_model(model_dir)
    generator, discriminator = saver_and_loader.load_discrim_and_generator(model_arch_config, data_config, num_classes,
                                                                           generator_path, discrim_path, device)
    return _to_gan_model(generator, discriminator, num_classes, device, model_arch_config, train_config), step_num


def _to_gan_model(generator, discriminator, num_classes, device, model_arch_config, train_config):
    return GanModel(generator, discriminator, num_classes, device, model_arch_config, train_config=train_config)


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
