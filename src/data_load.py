# -*- coding: utf-8 -*-
"""
Created on Thu May 17

@author: Gerardo Cervantes

Purpose: Functions that are used to generate and transform image data
"""

import torch
import PIL
import torchvision.datasets as torch_data_set
import torchvision.transforms.v2 as transforms
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
    image_height, image_width = get_image_height_and_width(data_config)
    batch_size = int(data_config['batch_size'])
    n_workers = int(data_config['workers'])
    return create_data_loader(data_dir, image_height, image_width, using_gpu=using_gpu,
                              batch_size=batch_size, n_workers=n_workers)


def get_image_height_and_width(data_config):
    image_height = int(int(data_config['base_height']) * (2 ** int(data_config['upsample_layers'])))
    image_width = int(int(data_config['base_width']) * (2 ** int(data_config['upsample_layers'])))
    return image_height, image_width


def create_latent_vector(data_config, model_arch_config, device):
    latent_vector_size = int(model_arch_config['latent_vector_size'])
    fixed_noise = torch.randn(int(data_config['batch_size']), latent_vector_size, device=device,
                              requires_grad=False)
    truncation_value = float(model_arch_config['truncation_value'])
    if truncation_value != 0.0:
        # https://github.com/pytorch/pytorch/blob/a40812de534b42fcf0eb57a5cecbfdc7a70100cf/torch/nn/init.py#L153
        fixed_noise = torch.nn.init.trunc_normal_(fixed_noise, a=(truncation_value * -1), b=truncation_value)
    return fixed_noise


def get_num_classes(data_config):
    data_loader = data_loader_from_config(data_config)
    return len(data_loader.dataset.classes)

def to_int16(label):
    return torch.tensor(label, dtype=torch.int16)

def create_data_loader(data_dir: str, image_height: int, image_width: int, dtype=torch.float16, using_gpu=False,
                       batch_size=1, n_workers=1):

    data_transform = transforms.Compose([transforms.Resize((image_height, image_width)),
                                         transforms.ToImageTensor(),
                                         transforms.ConvertImageDtype(torch.float32), # Converting to float16 is increasing VRAM? Strange.
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                         ])
    label_transform = to_int16
    try:
        data_set = torch_data_set.ImageFolder(root=data_dir, transform=data_transform)
        # data_set = torch_data_set.ImageFolder(root=data_dir, transform=data_transform, target_transform=label_transform)
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


# Resize images so width and height are both greater than min_size. Keep images the same if they already are bigger
# Keeps aspect ratio
def upscale_images(images, min_size: int):
    if len(images.size()) != 4:
        raise ValueError("Could not upscale images.  Images should be tensor of size (batch size, n_channels, w, h)")
    height = images.size(dim=2)
    width = images.size(dim=3)

    if width > min_size and height > min_size:
        return images

    ratio_to_upscale = float(min_size / min(width, height))

    if width < height:
        new_width = min_size
        new_height = int(ratio_to_upscale * height)
        # Safety check
        new_height = new_height if new_height >= min_size else min_size
    else:
        new_width = int(ratio_to_upscale * width)
        new_height = min_size
        # Safety check
        new_width = new_width if new_width >= min_size else min_size

    return _antialias_resize(images, new_width, new_height)


# As seen in https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/#evaluation-metrics
def _antialias_resize(batch, width, height):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((width, height), PIL.Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)
