# -*- coding: utf-8 -*-
"""
Created on Thu June 26 00:01:34 2020

@author: Gerardo Cervantes

Purpose: This metric takes images and returns the inception score (IS)
        Based on https://pytorch.org/hub/pytorch_vision_inception_v3/
"""
import torch
from ignite.engine import Engine
from ignite.metrics import InceptionScore, FID

from src.data_load import upscale_images


def _create_default_engine():
    # create default evaluator
    def eval_step(engine, batch):
        pred, true = batch
        return pred, true

    default_evaluator = Engine(eval_step)
    return default_evaluator


class Metrics:

    def __init__(self, compute_is, compute_fid, device=None):
        self.default_evaluator = _create_default_engine()
        self.device = device
        if device is None:
            self.device = torch.device("cpu")

        if compute_is:
            is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])
            is_metric.attach(self.default_evaluator, "is")
        if compute_fid:
            fid_metric = FID(device=device)
            fid_metric.attach(self.default_evaluator, "fid")

    def _compute_is(self, images, real_images):
        state = self.default_evaluator.run([[images, real_images]])
        is_score = state.metrics["is"]
        return is_score

    def _compute_fid(self, generated_images, real_images):
        state = self.default_evaluator.run([[generated_images, real_images]])
        fid_score = state.metrics["fid"]
        return fid_score

    # Images should be a tensor of size (batch size, n_channels, width, height) - n_channels is 3 for RGB
    # Images will be upscaled if the width or height are not above 300.  This is required by the inception model
    # Images should be scaled from 0 to 1, and they will be normalized to be from -1 to 1 for the inception model
    def score_metrics(self, images, compute_is, compute_fid, real_images=None):

        images = upscale_images(images, 299).to(self.device)
        real_images = upscale_images(real_images, 299).to(self.device)

        # Converts from 0 to 1 values to -1 to 1 values
        images = (images - 0.5) * 2
        real_images = (real_images - 0.5) * 2
        is_score, fid_score = None, None

        if compute_is:
            is_score = self._compute_is(images, real_images)
        if compute_fid:
            fid_score = self._compute_fid(images, real_images)
        return is_score, fid_score
