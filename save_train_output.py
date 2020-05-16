# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:12:31 2020

@author: Gerardo Cervantes
"""

import string
import os
import random
import torch

import torch.nn as nn
import Generator
import Discriminator
import shutil
import torchvision.utils as torch_utils

# This function throws exception if the directory doesn't exist, and gives the given error message
def is_valid_dir(dir_path, error_msg):
    if not os.path.isdir(dir_path):
        raise OSError(error_msg)


# Creates random combination of ascii and numbers
# taken from https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# Creates a directory with given name run_id in the output_dir
def create_run_dir(output_dir, run_id):
    is_valid_dir(output_dir, output_dir + '\nIs not a valid directory')
    try:
        run_dir_path = os.path.join(output_dir, run_id)
        
        os.mkdir(run_dir_path)
        print('Directory ', run_dir_path,  ' Created ')
        return run_dir_path
    except FileExistsError:
        raise OSError('Could create new directory with identifier: ' + run_id +
                      '\nIt\'s possible one exists already in the ' + output_dir)


# Writes architecture details onto file in run directory
def save_architecture(generator, discriminator, save_dir):
    with open(os.path.join(save_dir, 'architecture.txt'), "w") as text_file:
        text_file.write(str(generator))
        text_file.write(str(discriminator))


def create_gan_instances(config):

    n_gpu = int(config['CONFIGS']['ngpu'])
    latent_vector_size = int(config['CONFIGS']['latent_vector_size'])
    ngf = int(config['CONFIGS']['ngf'])
    ndf = int(config['CONFIGS']['ndf'])
    num_channels = int(config['CONFIGS']['num_channels'])

    device = torch.device('cuda:0' if (torch.cuda.is_available() and n_gpu > 0) else 'cpu')

    # Create the generator and discriminator
    generator = Generator.Generator(n_gpu, latent_vector_size, ngf, num_channels).to(device)
    discriminator = Discriminator.Discriminator(n_gpu, ndf, num_channels).to(device)

    generator = handle_multiple_gpus(generator, n_gpu, device)
    discriminator = handle_multiple_gpus(discriminator, n_gpu, device)
    return generator, discriminator, device


# Handle multi-gpu if desired, returns the new instance that is multi-gpu capable
def handle_multiple_gpus(torch_obj, num_gpu, device):
    if (device.type == 'cuda') and (num_gpu > 1):
        return nn.DataParallel(torch_obj, list(range(num_gpu)))
    else:
        return torch_obj


def save_gan_files(run_dir):
    shutil.copy(Generator.__name__ + '.py', os.path.abspath(run_dir))
    shutil.copy(Discriminator.__name__ + '.py', os.path.abspath(run_dir))


def save_model(generator, discriminator, generator_path, discriminator_path):
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)


def save_training_images(loader, save_dir):
    # Save training images
    real_batch = next(iter(loader))

    # batch_images is of size (batch_size, num_channels, height, width)
    batch_images = real_batch[0]
    save_path = os.path.join(save_dir, 'training batch.png')
    save_images(batch_images, save_path)


def save_images(tensor, save_path):
    # tensor should be of shape (batch_size, num_channels, height, width) as outputted by DataLoader
    torch_utils.save_image(tensor, save_path, normalize=True)


def load_model(config, generator_path, discriminator_path):
    generator, discriminator, device = create_gan_instances(config)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()
    discriminator.load_state_dict(torch.load(discriminator_path))
    discriminator.eval()
    return generator, discriminator, device
