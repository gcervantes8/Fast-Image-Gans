# -*- coding: utf-8 -*-
"""
Created on Thu May 17

@author: Gerardo Cervantes

Purpose: Functions that can save output into file or loads models.
"""

import os
import shutil
import torch
import torchvision.utils as torch_utils
from src.data_load import get_data_batch, color_transform, normalize
from torchsummary import summary

from src import Generator, Discriminator, create_model


# Writes text file with information of the generator and discriminator instances
def save_architecture(generator, discriminator, save_dir, data_config, model_arch_config):

    image_height = int(data_config['image_height'])
    image_width = int(data_config['image_width'])
    latent_vector_size = int(model_arch_config['latent_vector_size'])

    gen_model_stats = summary(generator, input_data=(latent_vector_size, 1, 1), verbose=0)
    gen_summary_str = str(gen_model_stats)
    discrim_model_stats = summary(discriminator, input_data=(3, image_height, image_width), verbose=0)
    discrim_summary_str = str(discrim_model_stats)

    with open(os.path.join(save_dir, 'architecture.txt'), 'w', encoding='utf-8') as text_file:
        text_file.write('Generator\n\n')
        text_file.write(str(generator))
        text_file.write(str(gen_summary_str))
        text_file.write('\n\nDiscriminator\n\n')
        text_file.write(str(discriminator))
        text_file.write(str(discrim_summary_str))


# Saves the python files Generator.py, and Discriminator.py to given directory
def save_gan_files(run_dir):
    shutil.copy(Generator.__name__.replace('.', '/') + '.py', os.path.abspath(run_dir))
    shutil.copy(Discriminator.__name__.replace('.', '/') + '.py', os.path.abspath(run_dir))


# Saves the trained generator and discriminator models in the given directories
def save_model(generator, discriminator, generator_path, discriminator_path):
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)


# Takes in pytorch's data-loader, and saves the training images in given directory
def save_train_batch(data_loader, device, save_path: str):
    # Save training images
    batch_images = get_data_batch(data_loader, device)
    batch_images = normalize(color_transform(batch_images))
    save_images(batch_images, save_path)


# Tensor should be of shape (batch_size, n_channels, height, width) as outputted by pytorch Data-Loader
def save_images(tensor, save_path, normalized=True):
    torch_utils.save_image(tensor, save_path, normalize=normalized)


def load_discrim_and_generator(config, generator_path, discrim_path):
    generator, discriminator, device = create_model.create_gan_instances(config)
    generator.load_state_dict(torch.load(generator_path))
    discriminator.load_state_dict(torch.load(discrim_path))
    return generator, discriminator, device


# Creates the generator and loads the given values from the model file
# Returns 2-tuple (loaded generator instance, device loaded with), device can be GPU or CPU
def load_generator(config, generator_path):
    generator, discriminator, device = create_model.create_gan_instances(config)
    generator.load_state_dict(torch.load(generator_path))
    return generator, device
