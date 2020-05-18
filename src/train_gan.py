# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 00:23:38 2020

@author: Gerardo Cervantes
"""

from __future__ import print_function
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from src import ini_parser, saver_and_loader, os_helper, create_model

import shutil
import os
import time

if __name__ == '__main__':

    config_file_path = 'model_config.ini'
    ini_config = ini_parser.read(config_file_path)

    output_dir = ini_config['CONFIGS']['output_dir']
    os_helper.is_valid_dir(output_dir, 'Output image directory is invalid' +
                                        '\nPath is not a directory: ' + output_dir)
    # Each run will be saved with model details, real images, and generated (fake) images, and models
    run_dir, run_id = os_helper.create_run_dir(output_dir)
    print('Output will be saved to ' + run_dir)

    shutil.copy(config_file_path, os.path.abspath(run_dir))
    print('Copied config file!')
    saver_and_loader.save_gan_files(run_dir)
    print('Copies the Generator and Discriminator files')

    data_dir = ini_config['CONFIGS']['dataroot']
    os_helper.is_valid_dir(data_dir, 'Invalid training data directory' +
                                  '\nPath is an invalid directory: ' + data_dir)
    data_loader = create_model.create_data_loader(ini_config, data_dir)
    saver_and_loader.save_training_images(data_loader, run_dir)

    latent_vector_size = int(ini_config['CONFIGS']['latent_vector_size'])

    netG, netD, device = create_model.create_gan_instances(ini_config)
    print('Is GPU available? ' + str(torch.cuda.is_available()))

    # Apply weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(create_model.weights_init)
    netG.apply(create_model.weights_init)

    saver_and_loader.save_architecture(netG, netD, run_dir)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors used to visualize the generator
    fixed_noise = torch.randn(64, latent_vector_size, 1, 1, device=device)

    fake_label, real_label = 0, 1

    beta1 = float(ini_config['CONFIGS']['beta1'])
    lr = float(ini_config['CONFIGS']['lr'])

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    n_epochs = int(ini_config['CONFIGS']['num_epochs'])

    print("Starting Training Loop...")

    for epoch in range(n_epochs):
        train_seq_start_time = time.time()
        # For each batch in the data-loader
        for i, data in enumerate(data_loader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, latent_vector_size, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tTime: %.2fs'
                      % (epoch, n_epochs, i, len(data_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, time.time() - train_seq_start_time))
                train_seq_start_time = time.time()
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

        # Check how the generator is doing by saving G's output on fixed_noise
        print('Saving fake images')
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        fake_img_output_path = run_dir + '/' + 'fake_images_epoch_' + str(epoch) + '.png'
        print(fake_img_output_path)
        saver_and_loader.save_images(fake, fake_img_output_path)

        generator_path = run_dir + '/' + 'generator_epoch_' + str(epoch) + '.pt'
        discriminator_path = run_dir + '/' + 'discriminator_epoch_' + str(epoch) + '.pt'
        saver_and_loader.save_model(netG, netD, generator_path, discriminator_path)
print('Training complete! Models and output saved in the output directory:')
print(run_dir)
