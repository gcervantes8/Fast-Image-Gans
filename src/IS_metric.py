# -*- coding: utf-8 -*-
"""
Created on Thu June 26 00:01:34 2020

@author: Gerardo Cervantes

Purpose: This metric takes images and returns the inception score (IS)
"""

from torchvision.models.inception import inception_v3
import numpy as np


def inception_score_metric(images, n_split=10, eps=1E-16):
	inception_model = inception_v3(pretrained=True)
	inception_model.eval()
	y_hat = inception_model(images)

	# enumerate splits of images/predictions
	scores = list()
	n_part = int(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve p(y|x)
		ix_start, ix_end = i * n_part, i * n_part + n_part
		p_yx = y_hat[ix_start:ix_end]
		# calculate p(y)
		p_y = np.expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = np.mean(sum_kl_d)
		# undo the log
		is_score = np.exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg, is_std = np.mean(scores), np.std(scores)
	return is_avg, is_std
