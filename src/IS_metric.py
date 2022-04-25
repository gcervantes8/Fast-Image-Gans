# -*- coding: utf-8 -*-
"""
Created on Thu June 26 00:01:34 2020

@author: Gerardo Cervantes

Purpose: This metric takes images and returns the inception score (IS)
        Based on https://pytorch.org/hub/pytorch_vision_inception_v3/
"""
import torch
from ignite.engine import Engine
import torchvision.transforms as transforms
from ignite.metrics import InceptionScore

from src.data_load import unnormalize


def _inception_score_metric(images):
    metric = InceptionScore()

    # create default evaluator for doctests
    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)
    metric.attach(default_evaluator, "is")
    state = default_evaluator.run([images])
    # is_metric = InceptionScore(output_transform=lambda x: x[0])
    return state.metrics["is"]


# Resize images so width and height are both greater than min_size. Keep images the same if they already are bigger
# Keeps aspect ratio
def upscale_images(images, min_size: int):
    if len(images.size()) != 4:
        raise ValueError("Could not upscale images.  Images should be tensor of size (batch size, n_channels, w, h)")
    width = images.size(dim=2)
    height = images.size(dim=3)
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
    resize_transform = torch.nn.Sequential(transforms.Resize((new_height, new_width)))
    return resize_transform(images)


# Images should be a tensor of size (batch size, n_channels, width, height) - n_channels is 3 for RGB
# Images will be upscaled if the width or height are not above 300.  This is required by the inception model
def inception_score(images, normalized=False):
    if normalized:
        images = unnormalize(images)

    images = upscale_images(images, 300)

    images_inception_format = unnormalize(images, norm_mean=torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
                                          norm_std=torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32))
    images_inception_format = images_inception_format.cpu()
    return _inception_score_metric(images_inception_format)
