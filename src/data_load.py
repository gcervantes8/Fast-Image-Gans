# -*- coding: utf-8 -*-
"""
Created on Thu May 17

@author: Gerardo Cervantes

Purpose: Functions that are used to generate and transform image data
"""

import torch
import torchvision.datasets as torch_data_set
import torchvision.transforms as transforms
from src import os_helper


def normalize(images, norm_mean=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
              norm_std=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)):
    normalize_transform = torch.nn.Sequential(
        transforms.Normalize(norm_mean, norm_std),
    )
    return normalize_transform(images)


def unnormalize(images, norm_mean=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
                norm_std=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)):
    unnormalize_transform = torch.nn.Sequential(
        transforms.Normalize((-norm_mean / norm_std).tolist(),
                             (1.0 / norm_std).tolist()))
    return unnormalize_transform(images)


def color_transform(images, brightness=0.1, contrast=0.05, saturation=0.1, hue=0.05):
    train_transform_augment = torch.nn.Sequential(
        transforms.ColorJitter(brightness=brightness, contrast=contrast,
                               saturation=saturation, hue=hue),
    )
    return train_transform_augment(images)


def data_loader_from_config(data_config, using_gpu=False):
    data_dir = data_config['train_dir']
    os_helper.is_valid_dir(data_dir, 'Invalid training data directory\nPath is an invalid directory: ' + data_dir)
    image_height = int(data_config['image_height'])
    image_width = int(data_config['image_width'])
    batch_size = int(data_config['batch_size'])
    n_workers = int(data_config['workers'])
    return create_data_loader(data_dir, image_height, image_width, using_gpu=using_gpu, batch_size=batch_size,
                              n_workers=n_workers)


def create_data_loader(data_dir: str, image_height: int, image_width: int, using_gpu=False, batch_size=1, n_workers=1):

    data_transform = transforms.Compose([transforms.Resize((image_height, image_width)),
                                         transforms.ToTensor()])
    try:
        data_set = torch_data_set.ImageFolder(root=data_dir, transform=data_transform)
    except FileNotFoundError:
        raise FileNotFoundError('Data directory provided should contain directories that have images in them, '
                                'directory provided: ' + data_dir)

    # Create the data-loader
    torch_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                               shuffle=True, num_workers=n_workers, pin_memory=using_gpu)
    return torch_loader


# Returns images of size: (batch_size, num_channels, height, width)
def get_data_batch(data_loader, device):
    return next(iter(data_loader))[0].to(device)
