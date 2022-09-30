# -*- coding: utf-8 -*-
"""
Created on Thu May 08 2020

@author: Gerardo Cervantes

Purpose: Loads trained models and generates new images
Specify the run folder, reads the config file to recreate the model, and loads the latest model in the directory
"""

import torch
from src import saver_and_loader, os_helper
from src.configs import ini_parser
from src.metrics import score_metrics
from src.data_load import unnormalize

import os
import logging
import argparse


def generate_images(model_dir: str, num_images: int):
    # Loads latest model and config file
    config_file_path = os_helper.find_config_file(model_dir)
    ini_config = ini_parser.read(config_file_path)
    generator_path, _, _ = os_helper.find_latest_generator_model(os.path.join(model_dir, 'models'))
    generator, device = saver_and_loader.load_generator(ini_config, generator_path)
    generator.eval()

    # Generate images
    latent_vector_size = int(ini_config['MODEL ARCHITECTURE']['latent_vector_size'])
    latent_vector = torch.randn(num_images, latent_vector_size, 1, 1, device=device)
    fake_images = generator.forward(latent_vector)
    return fake_images


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Generate images from a trained model')
    parser.add_argument('model_dir', type=str,
                        help='the directory containing the model that will generate the images')
    parser.add_argument('num_images', type=int,
                        help='the amount of images it will generate')
    parser.add_argument('--is', type=bool,
                        help='If true will calculate the inception score')
    args = parser.parse_args()
    fake_imgs = generate_images(args.model_dir, args.num_images)
    images_output_path = os.path.join(args.model_dir, 'fake_images.png')
    saver_and_loader.save_images(fake_imgs, images_output_path)
    fake_images_unnormalized = unnormalize(fake_imgs)  # Values from 0 to 1

    is_score, _ = score_metrics(fake_images_unnormalized, True, False)
    logging.info("Inception Score is: " + str(is_score))



