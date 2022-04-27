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
from ignite.metrics import InceptionScore, FID
import PIL


def _compute_is(default_evaluator, images, real_images):
    state = default_evaluator.run([[images, real_images]])
    is_score = state.metrics["is"]
    return is_score


def _create_default_engine():
    # create default evaluator
    def eval_step(engine, batch):
        pred, true = batch
        return pred, true
    default_evaluator = Engine(eval_step)
    return default_evaluator


def _compute_fid(default_evaluator, images_pred, images_true):

    state = default_evaluator.run([[images_pred, images_true]])
    fid_score = state.metrics["fid"]
    return fid_score


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


def preprocess_images_for_metric(images, device):
    images = upscale_images(images, 299)
    images = images * 255
    return images.to(device)


# Images should be a tensor of size (batch size, n_channels, width, height) - n_channels is 3 for RGB
# Images will be upscaled if the width or height are not above 300.  This is required by the inception model
def score_metrics(images, compute_is, compute_fid, real_images=None, device=None):
    if device is None:
        device = torch.device("cpu")

    images = preprocess_images_for_metric(images, device)
    if real_images is not None:
        real_images = preprocess_images_for_metric(real_images, device)
    else:
        compute_fid = False

    is_score, fid_score = None, None
    default_evaluator = _create_default_engine()
    if compute_is:
        is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])
        is_metric.attach(default_evaluator, "is")
    if compute_fid:
        fid_metric = FID(device=device)
        fid_metric.attach(default_evaluator, "fid")
    if compute_is:
        is_score = _compute_is(default_evaluator, images, images)
    if compute_fid:
        fid_score = _compute_fid(default_evaluator, images, real_images)
    return is_score, fid_score
