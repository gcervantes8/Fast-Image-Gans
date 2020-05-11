# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 00:23:38 2020

@author: Gerardo Cervantes
"""

from __future__ import print_function
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as torch_data_set
import torchvision.transforms as transforms
import torchvision.utils as torch_utils

import save_train_output
import ini_parser

import os
import time

if __name__ == '__main__':

    config_file_path = 'model_config.ini'
    ini_config = ini_parser.read(config_file_path)

    def create_run_dir(directory_path):
        save_train_output.is_valid_dir(directory_path, 'Output image directory is invalid' +
                                      '\nPath is not a directory: ' + directory_path)

        # Each run will be saved with model details, real images, and generated (fake) images, and models
        run_id = save_train_output.id_generator()
        run_path_dir = save_train_output.create_run_dir(output_dir, run_id)
        return run_path_dir


    output_dir = ini_config['CONFIGS']['output_dir']
    run_dir = create_run_dir(output_dir)
    print('Output will be saved to ' + run_dir)

    # Create the data-set using an image folder
    def create_data_loader(config):
        data_dir = ini_config['CONFIGS']['dataroot']
        save_train_output.is_valid_dir(data_dir, 'Training data directory is invalid' +
                                      '\nPath is not a directory: ' + data_dir)

        image_size = int(ini_config['CONFIGS']['image_size'])
        data_set = torch_data_set.ImageFolder(root=data_dir,
                                              transform=transforms.Compose([
                                                  transforms.Resize(image_size),
                                                  transforms.CenterCrop(image_size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                              ]))

        batch_size = int(config['CONFIGS']['batch_size'])
        n_workers = int(config['CONFIGS']['workers'])
        # Create the data-loader
        torch_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                                   shuffle=True, num_workers=n_workers)
        return torch_loader


    data_loader = create_data_loader(ini_config)

    n_gpu = int(ini_config['CONFIGS']['ngpu'])
    device = torch.device('cuda:0' if (torch.cuda.is_available() and n_gpu > 0) else 'cpu')
    print('Is GPU available? ' + str(torch.cuda.is_available()))

    def save_images(tensor, save_path):
        # tensor should be of shape (batch_size, num_channels, height, width) as outputted by DataLoader
        torch_utils.save_image(tensor, save_path, normalize=True)

    def save_training_images(loader, save_dir):
        # Save training images
        real_batch = next(iter(loader))

        # batch_images is of size (batch_size, num_channels, height, width)
        batch_images = real_batch[0]
        save_path = os.path.join(save_dir, 'trainingImages.png')
        save_images(batch_images, save_path)

    save_training_images(data_loader, run_dir)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    latent_vector_size = int(ini_config['CONFIGS']['latent_vector_size'])
    ngf = int(ini_config['CONFIGS']['ngf'])
    num_channels = int(ini_config['CONFIGS']['num_channels'])
    ndf = int(ini_config['CONFIGS']['ndf'])


    class Generator(nn.Module):
        def __init__(self, num_gpu):
            super(Generator, self).__init__()
            self.n_gpu = num_gpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(latent_vector_size, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, generator_input):
            return self.main(generator_input)


    class Discriminator(nn.Module):
        def __init__(self, num_gpu):
            super(Discriminator, self).__init__()
            self.ngpu = num_gpu
            self.main = nn.Sequential(
                # input is (num_channels) x 64 x 64
                nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, discriminator_input):
            return self.main(discriminator_input)

    # Create the generator and discriminator
    netG = Generator(n_gpu).to(device)
    netD = Discriminator(n_gpu).to(device)

    # Handle multi-gpu if desired, returns the new instance that is multi-gpu capable
    def handle_multiple_gpus(torch_obj, num_gpu):
        if (device.type == 'cuda') and (num_gpu > 1):
            return nn.DataParallel(torch_obj, list(range(num_gpu)))


    netG = handle_multiple_gpus(netG, n_gpu)
    netD = handle_multiple_gpus(netD, n_gpu)

    # Apply weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(weights_init)
    netG.apply(weights_init)

    # Writes architecture details onto file in run directory
    with open(os.path.join(run_dir, 'architecture.txt'), "w") as text_file:
        text_file.write(str(netG))
        text_file.write(str(netD))

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
        fake_img_output_path = run_dir + '/' + 'fakes_images_epoch_' + str(epoch) + '.png'
        print(fake_img_output_path)
        save_images(fake, fake_img_output_path)
