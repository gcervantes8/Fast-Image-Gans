# -*- coding: utf-8 -*-
"""
Created on Thu June 26 00:01:34 2020

@author: Gerardo Cervantes

Purpose: This metric takes images and returns the inception score (IS)
"""
import torch
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from src.data.data_load import get_data_batch
from tqdm import tqdm
import logging


class Metrics:

    def __init__(self, compute_is, compute_fid, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cpu")
        self.is_metric = None
        self.fid_metric = None
        if compute_is:
            self.is_metric = InceptionScore(normalize=True)
            self.is_metric = self.is_metric.to(device)
        if compute_fid:
            self.fid_metric = FrechetInceptionDistance(normalize=True, reset_real_features=False)
            self.fid_metric = self.fid_metric.to(device)

    def _compute_is(self):
        is_score = self.is_metric.compute()
        return is_score

    def _compute_fid(self):
        fid_score = self.fid_metric.compute()
        return fid_score

    # Aggregate images for FID or Inception Score with a data loader
    def aggregate_data_loader_images(self, data_loader, eval_bs, device, real=True):
        def get_data_load_images():
            return get_data_batch(data_loader, device, unnormalize_batch=True)
        self.aggregate_images_from_fn(get_data_load_images, eval_bs, real=real)

    def aggregate_images_from_fn(self, fn_for_images, eval_bs, real=True):
        with tqdm(total=eval_bs) as pbar:
            accum_batch_size = 0
            while accum_batch_size < eval_bs:
                images = fn_for_images()
                self.aggregate_images(images, real=real)
                num_images_aggregated = images.size(axis=0)
                accum_batch_size += num_images_aggregated
                pbar.update(num_images_aggregated)

    def aggregate_images(self, images, real=False):
        if self.fid_metric:
            self.fid_metric.update(images, real=real)
        if self.is_metric and not real:
            self.is_metric.update(images)

    def reset_metrics(self):
        self.is_metric.reset()
        self.fid_metric.reset()

    # Images should be a tensor of size (batch size, n_channels, height, width) - n_channels is 3 for RGB
    # Images will be upscaled if the width or height are not above 300.  This is required by the inception model
    # Images should be scaled from 0 to 1, and they will be normalized to be from -1 to 1 for the inception model
    def score_metrics(self, compute_is, compute_fid):

        is_score, fid_score = None, None

        if compute_is:
            is_score = self._compute_is()
        if compute_fid:
            fid_score = self._compute_fid()
        return is_score, fid_score
    
    def log_scores(self, is_score, fid_score):
        if is_score:
            is_score_avg, is_score_std = is_score
            logging.info('Inception Score: μ = %.2f σ = %.2f' % (round(float(is_score_avg), 2), round(float(is_score_std), 2)))
        if fid_score:
            logging.info('FID Score: %.2f' % round(float(fid_score), 2))
