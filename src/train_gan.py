# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 00:23:38 2020

@author: Gerardo Cervantes

Purpose: Train the GAN (Generative Adversarial Network) model
"""

from __future__ import print_function
import torch

from src import ini_parser, saver_and_loader, os_helper, create_model
from src.gan_model import GanModel

import shutil
import logging
import os
import time

if __name__ == '__main__':

    # Config file
    config_file_path = 'model_config.ini'
    if not os.path.exists(config_file_path):
        raise OSError('Invalid configuration file path: ' +  config_file_path + ' \ndoesn\'t exist')

    config = ini_parser.read(config_file_path)
    # Creates the run directory in the output folder specified in the configuration file
    output_dir = config['CONFIGS']['output_dir']
    os_helper.is_valid_dir(output_dir, 'Output image directory is invalid\nPath is not a directory: ' + output_dir)
    run_dir, run_id = os_helper.create_run_dir(output_dir)
    img_dir = os_helper.create_dir(run_dir, 'images')
    model_dir = os_helper.create_dir(run_dir, 'models')

    # Logs training information, everything logged will also be outputted to stdout (printed)
    log_path = os.path.join(run_dir, 'train.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Directory ' + run_dir + ' created, training output will be saved here')

    # Copies config and python model files
    shutil.copy(config_file_path, os.path.abspath(run_dir))
    logging.info('Copied config file!')
    saver_and_loader.save_gan_files(run_dir)
    logging.info('Copied the Generator and Discriminator files')

    # Creates data-loader
    data_dir = config['CONFIGS']['dataroot']
    os_helper.is_valid_dir(data_dir, 'Invalid training data directory\nPath is an invalid directory: ' + data_dir)
    data_loader = create_model.create_data_loader(config, data_dir)
    logging.info('Data size is ' + str(len(data_loader.dataset)) + ' images')

    # Save training images
    saver_and_loader.save_training_images(data_loader, img_dir, 'train_batch.png')

    # Create model
    netG, netD, device = create_model.create_gan_instances(config)
    saver_and_loader.save_architecture(netG, netD, run_dir, config)
    logging.info('Is GPU available? ' + str(torch.cuda.is_available()))
    netD.apply(create_model.weights_init)
    netG.apply(create_model.weights_init)

    gan_model = GanModel(netG, netD, device, config)
    latent_vector_size = int(config['CONFIGS']['latent_vector_size'])
    fixed_noise = torch.randn(64, latent_vector_size, 1, 1, device=device)

    n_epochs = int(config['CONFIGS']['num_epochs'])
    logging.info("Starting Training Loop...")

    for epoch in range(n_epochs):
        train_seq_start_time = time.time()
        # For each batch in the data-loader
        data_get_time = 0
        model_update_time = 0
        data_start_time = time.time()
        for i, data in enumerate(data_loader, 0):

            real_data = data[0].to(device)
            data_get_time += time.time() - data_start_time

            model_start_time = time.time()
            errD, errG, D_x, D_G_z1, D_G_z2 = gan_model.update_minimax(real_data)
            model_update_time += time.time() - model_start_time

            # Output training stats
            if i % 150 == 0:
                logging.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tTime: %.2fs'
                             % (epoch, n_epochs, i, len(data_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,
                                time.time() - train_seq_start_time))
                logging.info('Data retrieve time: %.2fs Model updating time: %.2fs' % (data_get_time, model_update_time))
                data_get_time = 0
                model_update_time = 0
                train_seq_start_time = time.time()

            data_start_time = time.time()

        save_imgs_start_time = time.time()
        fake_img_output_path = os.path.join(img_dir, 'fake_epoch_' + str(epoch + 1) + '.png')
        logging.info('Saving fake images: ' + fake_img_output_path)
        fake_images = gan_model.generate_images(fixed_noise)
        saver_and_loader.save_images(fake_images, fake_img_output_path)
        print('Time to save images: %.2fs ' % (time.time() - save_imgs_start_time))

        # Saves models
        save_models_start_time = time.time()
        generator_path = os.path.join(model_dir, 'gen_epoch_' + str(epoch) + '.pt')
        discriminator_path = os.path.join(model_dir, 'discrim_epoch_' + str(epoch) + '.pt')
        saver_and_loader.save_model(netG, netD, generator_path, discriminator_path)
        print('Time to save models: %.2fs ' % (time.time() - save_models_start_time))
    logging.info('Training complete! Models and output saved in the output directory:')
    logging.info(run_dir)
