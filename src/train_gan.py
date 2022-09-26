# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 00:23:38 2020

@author: Gerardo Cervantes

Purpose: Train the GAN (Generative Adversarial Network) model
"""

from __future__ import print_function
import torch
from torch.profiler import profile, ProfilerActivity

from src import ini_parser, saver_and_loader, os_helper, create_model
from src.data_load import data_loader_from_config, color_transform, normalize, get_data_batch, unnormalize
from src.gan_model import GanModel
from src.metrics import score_metrics

import shutil
import logging
import os
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.cuda.empty_cache()


def train(config_file_path: str):

    if not os.path.exists(config_file_path):
        raise OSError('Configuration file path doesn\'t exist:' + config_file_path)

    config = ini_parser.read(config_file_path)

    # Creates the run directory in the output folder specified in the configuration file
    model_config = config['MODEL']

    models_dir, model_name = model_config['models_dir'], model_config['model_name']
    os_helper.is_valid_dir(models_dir, 'Model directory is invalid\nPath is not a directory: ' + models_dir)

    model_dir_name = 'models'
    images_dir_name = 'images'
    profiler_dir_name = 'profiler'

    run_dir = os.path.join(models_dir, model_name)
    will_restore_model = os.path.isdir(os.path.join(run_dir, model_dir_name))

    # Then restore existing model
    if will_restore_model:
        img_dir = os.path.join(run_dir, images_dir_name)
        model_dir = os.path.join(run_dir, model_dir_name)
        profiler_dir = os.path.join(run_dir, profiler_dir_name)
    else:
        run_dir, run_id = os_helper.create_run_dir(models_dir)
        img_dir = os_helper.create_dir(run_dir, images_dir_name)
        model_dir = os_helper.create_dir(run_dir, model_dir_name)
        profiler_dir = os_helper.create_dir(run_dir, profiler_dir_name)

    # Logs training information, everything logged will also be outputted to stdout (printed)
    log_path = os.path.join(run_dir, 'train.log')
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    if not will_restore_model:
        logging.info('Directory ' + run_dir + ' created, training output will be saved here')
        # Copies config and python model files
        shutil.copy(config_file_path, os.path.abspath(run_dir))
        logging.info('Copied config file!')
        saver_and_loader.save_gan_files(run_dir)
        logging.info('Copied the Generator and Discriminator files')
    else:
        logging.info('Directory ' + run_dir + ' loaded, training output will be saved here')

    # Set device
    n_gpus = int(config['MACHINE']['ngpu'])
    device = create_model.get_device(n_gpus)
    running_on_cpu = str(device) == 'cpu'

    # Creates data-loader
    data_config = config['DATA']
    data_config['image_height'] = str(int(data_config['base_height']) * (2 ** int(data_config['upsample_layers'])))
    data_config['image_width'] = str(int(data_config['base_width']) * (2 ** int(data_config['upsample_layers'])))
    data_loader = data_loader_from_config(data_config, using_gpu=not running_on_cpu)
    logging.info('Data size is ' + str(len(data_loader.dataset)) + ' images')

    # Save training images
    saver_and_loader.save_train_batch(data_loader, os.path.join(img_dir, 'train_batch.png'))
    num_classes = len(data_loader.dataset.classes)
    logging.info('Number of different image labels: ' + str(num_classes))
    real_images = get_data_batch(data_loader, device)

    n_color_channels = int(data_config['num_channels'])
    # Create model
    loaded_epoch_num = 0
    model_arch_config = config['MODEL ARCHITECTURE']
    # If it exists, then try to load the model
    if will_restore_model:
        generator_path, discriminator_path, loaded_epoch_num = os_helper.find_latest_generator_model(model_dir)
        logging.info('Loading model from epoch ' + str(loaded_epoch_num) + ' ...')
        logging.info(generator_path)
        logging.info(discriminator_path)
        netG, netD = saver_and_loader.load_discrim_and_generator(config, generator_path, discriminator_path, device)
        logging.info('Model loaded!')
    else:
        netG, netD = create_model.create_gan_instances(model_arch_config, data_config, device,
                                                       num_channels=n_color_channels, num_classes=num_classes,
                                                       n_gpus=n_gpus)
        saver_and_loader.save_architecture(netG, netD, run_dir, data_config, model_arch_config)
        # TODO Apply weight initialization to only DCGAN
        # netD.apply(create_model.weights_init)
        # netG.apply(create_model.weights_init)

    logging.info('Is GPU available? ' + str(torch.cuda.is_available()) + ' - Running on device:' + str(device))

    metrics_config = config['METRICS']
    compute_is = metrics_config.getboolean('is_metric')
    compute_fid = metrics_config.getboolean('fid_metric')

    if compute_is or compute_fid:
        logging.info('Computing metrics for real images ...')
        is_score, fid_score = score_metrics(get_data_batch(data_loader, device), compute_is, compute_fid,
                                            real_images=real_images, device=device)
        if compute_is:
            logging.info('Inception Score for real images: %.2f' % round(is_score, 2))
        if compute_fid:
            logging.info('FID Score for real images: %.2f' % round(fid_score, 2))
    train_config = config['TRAIN']
    gan_model = GanModel(netG, netD, num_classes, device, model_arch_config, train_config=train_config)
    latent_vector_size = int(model_arch_config['latent_vector_size'])
    fixed_noise = torch.randn(int(data_config['batch_size']), latent_vector_size, device=device,
                              requires_grad=False)

    truncation_value = float(model_arch_config['orthogonal_value'])
    if truncation_value != 0:
        # https://github.com/pytorch/pytorch/blob/a40812de534b42fcf0eb57a5cecbfdc7a70100cf/torch/nn/init.py#L153
        torch.nn.init.trunc_normal_(fixed_noise, a=(truncation_value * -1), b=truncation_value)

    fixed_labels = torch.randint(low=0, high=num_classes, size=[int(data_config['batch_size'])], device=device,
                                 dtype=torch.int64)

    n_epochs = int(train_config['num_epochs'])
    save_steps = int(train_config['save_steps'])
    save_steps = None if save_steps == 0 else save_steps
    eval_steps = int(metrics_config['steps_to_eval'])
    eval_steps = None if eval_steps == 0 else eval_steps
    log_steps = int(train_config['log_steps'])

    logging.info("Starting Training Loop...")
    profile_devices = [ProfilerActivity.CPU]
    if not running_on_cpu:
        profile_devices.append(ProfilerActivity.CUDA)
    profiler = profile(activities=profile_devices,
                       schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                       on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
                       record_shapes=True, profile_memory=True)
    profiler.start()
    n_steps = 0
    steps_in_epoch = len(data_loader)
    train_generator = True
    alternate_generator_training = train_config.getboolean('two_d_steps_per_g')
    total_g_error, total_d_error = 0.0, 0.0
    g_steps, d_steps = 0, 0
    for epoch in range(n_epochs):
        epoch_after_loading = loaded_epoch_num + epoch
        train_seq_start_time = time.time()
        # For each batch in the data-loader
        data_time, model_time = 0.0, 0.0
        data_start_time = time.time()

        for i, batch in enumerate(data_loader, 0):

            n_steps += 1
            real_data, labels = batch

            # Normalization can't be done on bloat16 operators
            if running_on_cpu:
                real_data = normalize(color_transform(real_data))
            if train_config.getboolean('mixed_precision'):
                real_data = real_data.to(torch.bfloat16) if running_on_cpu else real_data.to(torch.float16)

            real_data = real_data.to(device)  # Moving to GPU is a slow operation
            labels = labels.to(device)  # Moving to GPU is a slow operation
            if not running_on_cpu:
                real_data = normalize(color_transform(real_data))

            data_time += time.time() - data_start_time
            model_start_time = time.time()

            err_discriminator, err_generator = gan_model.update_minimax(real_data, labels,
                                                                        train_generator=train_generator)
            alternate_generator_training = not alternate_generator_training

            if err_generator:
                total_g_error += err_generator
                g_steps += 1

            if err_discriminator:
                total_d_error += err_discriminator
                d_steps += 1

            model_time += time.time() - model_start_time
            # Output training stats
            if n_steps % log_steps == 0:
                d_loss = '{:.4f}'.format((total_d_error / d_steps)) if d_steps else '0 steps'
                g_loss = '{:.4f}'.format((total_g_error / g_steps)) if g_steps else '0 steps'

                logging.info('[{}/{}][{}/{}]\tLoss_D: {}\tLoss_G: {}\tTime: {:.2f}s'.format(
                    epoch, n_epochs, n_steps % steps_in_epoch, steps_in_epoch, d_loss, g_loss,
                    time.time() - train_seq_start_time))
                logging.info(
                    'Data retrieve time: %.2fs Model updating time: %.2fs' % (data_time, model_time))

                data_time, model_time = 0, 0
                train_seq_start_time = time.time()

            # Save every save_steps or every epoch if save_steps is None
            if (save_steps is not None and n_steps % save_steps == 0) or \
                    (save_steps is None and n_steps % steps_in_epoch == 0):
                save_imgs_start_time = time.time()
                save_identifier = epoch_after_loading if save_steps is None else n_steps
                fake_img_output_path = os.path.join(img_dir, 'generated_image_' + str(save_identifier) + '.png')
                logging.info('Saving fake images: ' + fake_img_output_path)
                fake_images = gan_model.generate_images(fixed_noise, fixed_labels)
                saver_and_loader.save_images(fake_images, fake_img_output_path)
                del fake_images
                logging.info('Time to save images: %.2fs ' % (time.time() - save_imgs_start_time))
                gan_model.save(model_dir, save_identifier)

            if (eval_steps is not None and n_steps % eval_steps == 0) or \
                    (eval_steps is None and n_steps % steps_in_epoch == 0):
                if compute_is or compute_fid:
                    logging.info('Computing metrics for the saved images ...')
                    fake_images = gan_model.generate_images(fixed_noise, fixed_labels)
                    is_score, fid_score = score_metrics(unnormalize(fake_images), compute_is, compute_fid,
                                                        real_images=real_images, device=device)
                    if compute_is:
                        logging.info('Inception Score: %.2f' % round(is_score, 2))
                    if compute_fid:
                        logging.info('FID Score: %.2f' % round(fid_score, 2))
            data_start_time = time.time()
            profiler.step()

    profiler.stop()
    logging.info('Training complete! Models and output saved in the output directory:')
    logging.info(run_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Uses default config file
    train('model_config.ini')

