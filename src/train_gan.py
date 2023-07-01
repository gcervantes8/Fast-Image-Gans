# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 00:23:38 2020

@author: Gerardo Cervantes

Purpose: Train the GAN (Generative Adversarial Network) model
"""

import torch
from torch.profiler import profile, ProfilerActivity

from src import saver_and_loader, create_model, os_helper
from src.configs import ini_parser
from src.data_load import data_loader_from_config, color_transform, normalize, get_data_batch, unnormalize, \
    create_latent_vector, get_num_classes
from src.metrics import Metrics
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import shutil
import logging
import os
import time


def train(config_file_path: str):
    if not os.path.exists(config_file_path):
        raise OSError('Configuration file path doesn\'t exist:' + config_file_path)

    config = ini_parser.read_with_defaults(config_file_path)

    will_restore_model = saver_and_loader.is_loadable_model(config)
    if will_restore_model:
        run_dir, model_dir, img_dir, profiler_dir = saver_and_loader.get_run_directories(config)
    else:
        run_dir, model_dir, img_dir, profiler_dir = saver_and_loader.create_run_directories(config)

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
        logging.info('Directory "' + run_dir + '" created, training output will be saved here')
        # Copies config file
        shutil.copy(config_file_path, os.path.abspath(run_dir))
        logging.info('Copied config file!')
    else:
        logging.info('Directory ' + run_dir + ' loaded, training output will be saved here')

    # Set device
    n_gpus = int(config['MACHINE']['ngpu'])
    device = create_model.get_device(n_gpus)
    logging.info('Running on: ' + str(device))
    running_on_cpu = str(device) == 'cpu'

    # Creates data-loader
    data_config = config['DATA']
    train_config = config['TRAIN']
    data_config['image_height'] = str(int(data_config['base_height']) * (2 ** int(data_config['upsample_layers'])))
    data_config['image_width'] = str(int(data_config['base_width']) * (2 ** int(data_config['upsample_layers'])))
    is_mixed_precision = train_config.getboolean('mixed_precision')
    data_dtype = torch.float32
    if is_mixed_precision and not running_on_cpu:
        # Avoid if running on CPU since we would need to use bfloat16. Image processing is not supported for bfloat16
        data_dtype = torch.float16
    data_loader = data_loader_from_config(data_config, data_dtype=data_dtype, using_gpu=not running_on_cpu)
    logging.info('Data size is ' + str(len(data_loader.dataset)) + ' images')

    # Save training images
    saver_and_loader.save_train_batch(data_loader, os.path.join(img_dir, 'train_batch.png'))
    num_classes = get_num_classes(data_config)
    logging.info('Number of different image labels: ' + str(num_classes))
    real_images = get_data_batch(data_loader, device)
    model_arch_config = config['MODEL ARCHITECTURE']

    logging.info('Creating model...')

    gan_model = create_model.create_gan_model(model_arch_config, data_config, train_config, device, n_gpus)
    gan_model.save_architecture(run_dir, data_config)

    logging.info('Created GAN model')
    loaded_step_num = 0
    if will_restore_model:
        loaded_step_num = saver_and_loader.load_model(gan_model, model_dir)
        logging.info('Restored model from step ' + str(loaded_step_num))

    logging.info('Is GPU available? ' + str(torch.cuda.is_available()) + ' - Running on device:' + str(device))

    metrics_config = config['METRICS']
    compute_is = metrics_config.getboolean('is_metric')
    compute_fid = metrics_config.getboolean('fid_metric')

    metrics_scorer = None
    eval_bs = 5000
    if compute_is or compute_fid:
        logging.info('Computing metrics for real images ...')
        metrics_scorer = Metrics(compute_is, compute_fid, device=device)
        metrics_scorer.aggregate_data_loader_images(data_loader, eval_bs, device, real=True)
        metrics_scorer.aggregate_data_loader_images(data_loader, eval_bs, device, real=False)
        is_score, fid_score = metrics_scorer.score_metrics(compute_is, compute_fid)
        metrics_scorer.reset_metrics()
        metrics_scorer.log_scores(is_score, fid_score)

    # TODO Add option to be able to generate labels from dataset distribution, instead of sequentially
    def create_noise_and_labels():
        noise = create_latent_vector(data_config, model_arch_config, device)
        labels = torch.arange(start=0, end=int(data_config['batch_size']), device=device,
                                dtype=torch.int64) % num_classes
        return noise, labels
    fixed_noise, fixed_labels = create_noise_and_labels()

    def generate_images():
        noise, labels = create_noise_and_labels()
        fake_images = gan_model.generate_images(noise, labels)
        return fake_images
    n_epochs, log_steps = int(train_config['num_epochs']), int(train_config['log_steps'])
    save_steps = int(train_config['save_steps'])
    save_steps = None if save_steps == 0 else save_steps
    eval_steps = int(metrics_config['steps_to_eval'])
    eval_steps = None if eval_steps == 0 else eval_steps

    logging.info("Starting Training Loop...")

    def tensorboard_profiler():
        profile_devices = [ProfilerActivity.CPU]
        if not running_on_cpu:
            profile_devices.append(ProfilerActivity.CUDA)
        return profile(activities=profile_devices,
                       schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                       on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
                       record_shapes=True, profile_memory=True)

    profiler = tensorboard_profiler()
    if profiler:
        profiler.start()

    n_steps = 0
    steps_in_epoch = len(data_loader)
    total_g_error, total_d_error = 0.0, 0.0
    g_steps, d_steps = 0, 0
    data_time, model_time = 0.0, 0.0
    data_start_time = time.time()
    train_seq_start_time = time.time()
    if train_config.getboolean('compile'):
        logging.info('Compiling model with PyTorch 2.0')
        gan_model.optimize_models()
    for epoch in range(n_epochs):
        for i, batch in enumerate(data_loader, 0):

            n_steps += 1
            real_data, labels = batch

            # Normalization can't be done on bloat16 operators
            is_bfloat16_dtype = running_on_cpu and is_mixed_precision
            if is_bfloat16_dtype:
                real_data = normalize(color_transform(real_data))
                real_data = real_data.to(torch.bfloat16)

            real_data = real_data.to(device)  # Moving to GPU is a slow operation
            labels = labels.to(device)  # Moving to GPU is a slow operation
            if not is_bfloat16_dtype:
                real_data = normalize(color_transform(real_data))

            data_time += time.time() - data_start_time
            model_start_time = time.time()
            if train_config.getboolean('channels_last'):
                real_data = real_data.to(memory_format=torch.channels_last)  # Replace with your input
            err_discriminator, err_generator = gan_model.update_minimax(real_data, labels)

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

                logging.info('[{}/{}][{}/{}]\t Loss_D: {}\t Loss_G: {}\t Time: {:.2f}s'.format(
                    epoch, n_epochs, n_steps % steps_in_epoch, steps_in_epoch, d_loss, g_loss,
                    time.time() - train_seq_start_time))
                logging.info(
                    'Data retrieve time: %.2fs Model updating time: %.2fs' % (data_time, model_time))

                data_time, model_time = 0, 0
                total_g_error, total_d_error = 0.0, 0.0
                g_steps, d_steps = 0, 0
                train_seq_start_time = time.time()

            # Save every save_steps or every epoch if save_steps is None
            if n_steps % save_steps == 0:
                save_start_time = time.time()
                save_identifier = loaded_step_num + n_steps
                fake_img_output_path = os.path.join(img_dir, 'generated_image_' + str(save_identifier) + '.png')
                logging.info('Saving fake images: ' + fake_img_output_path)
                fake_images = gan_model.generate_images(fixed_noise, fixed_labels).cpu()
                saver_and_loader.save_images(fake_images.to(torch.float32), fake_img_output_path)
                del fake_images
                gan_model.save(model_dir, save_identifier)
                logging.info('Time to save images and model: %.2fs ' % (time.time() - save_start_time))

            if n_steps % eval_steps == 0:
                if metrics_scorer:
                    metric_start_time = time.time()
                    logging.info('Computing metrics for the saved images ...')

                    metrics_scorer.aggregate_images_from_fn(generate_images, eval_bs, real=False)
                    is_score, fid_score = metrics_scorer.score_metrics(compute_is, compute_fid)
                    metrics_scorer.reset_metrics()
                    metrics_scorer.log_scores(is_score, fid_score)
                    logging.info('Time to compute metrics: %.2fs ' % (time.time() - metric_start_time))
            data_start_time = time.time()
            if profiler:
                profiler.step()
    profiler.stop()
    logging.info('Training complete! Models and output saved in the output directory:')
    logging.info(run_dir)
    return run_dir


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Path to the configuration file')
    parser.add_argument('config_file', type=str,
                        help='The configuration file to train with, configuration files end with .ini extension.\n'
                        'Default config files are in the configs folder.')
    args = parser.parse_args()

    train(args.config_file)
