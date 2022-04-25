# -*- coding: utf-8 -*-
"""
Created on Thu May 17

@author: Gerardo Cervantes

Purpose: Functions that are used to generate and transform image data
"""

import torch
import torchvision.datasets as torch_data_set
import torchvision.transforms as transforms


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


def create_data_loader(config, data_dir: str):

    image_height = int(config['CONFIGS']['image_height'])
    image_width = int(config['CONFIGS']['image_width'])
    try:
        data_set = torch_data_set.ImageFolder(root=data_dir,
                                              transform=transforms.Compose([
                                                  transforms.Resize((image_height, image_width)),
                                                  transforms.ToTensor(),
                                              ]))
    except FileNotFoundError:
        raise FileNotFoundError('Data directory provided should contain directories that have images in them, '
                                'directory provided: ' + data_dir)

    batch_size = int(config['CONFIGS']['batch_size'])
    n_workers = int(config['CONFIGS']['workers'])
    # Create the data-loader
    torch_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                               shuffle=True, num_workers=n_workers)
    return torch_loader


# Returns images of size: (batch_size, num_channels, height, width)
def get_data_batch(data_loader, device):
    return next(iter(data_loader))[0].to(device)
