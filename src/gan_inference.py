# -*- coding: utf-8 -*-
"""
Created on Thu May 08 2020

@author: Gerardo Cervantes

Purpose: Loads trained models and generates new images
Specify the run folder, reads the config file to recreate the model, and loads the latest model in the directory
"""

import torch
from src import ini_parser, saver_and_loader, os_helper
import os

run_dir = 'output/7CXI'
n_images = 64


# Loads latest model and config file
config_file_path = os_helper.find_config_file(run_dir)
ini_config = ini_parser.read(config_file_path)
print('Loaded config file!' + config_file_path)
generator_path = os_helper.find_latest_generator_model(run_dir)
print('Loading model: ' + generator_path)
generator, device, _ = saver_and_loader.load_generator(ini_config, generator_path)
generator.eval()
print('Loaded model!')

# Generate images
latent_vector_size = int(ini_config['CONFIGS']['latent_vector_size'])
latent_vector = torch.randn(n_images, latent_vector_size, 1, 1, device=device)
print('Created latent vector')
fake_images = generator.forward(latent_vector)
print('Finished generating images')

# Saves generated images
images_output_path = os.path.join(run_dir, 'fake_images.png')
saver_and_loader.save_images(fake_images, images_output_path)
print('Saved generated images, ' + images_output_path)

