# -*- coding: utf-8 -*-
"""
Created on Thu May 08 2020

@author: Gerardo Cervantes

Purpose: Loads trained models and generates new images
Specify the run folder, reads the config file to recreate the model, and loads the latest model in the directory
"""

import torch
from torchvision import transforms
import math
import PIL
from src import saver_and_loader, os_helper
from src.configs import ini_parser
# from src.metrics import score_metrics
from src.data_load import unnormalize, get_num_classes, create_latent_vector
from src import create_model

import os
import logging
import argparse


def generate_batch_image(ini_config, gan_model, num_images: int):
    model_arch_config, data_config = ini_config['MODEL ARCHITECTURE'], ini_config['DATA']
    n_gpus = int(config['MACHINE']['ngpu'])
    device = create_model.get_device(n_gpus)
    num_classes = get_num_classes(data_config)

    # Generate images
    latent_vector = create_latent_vector(data_config, model_arch_config, device)

    fixed_labels = torch.arange(start=0, end=num_images, device=device, dtype=torch.int64) % num_classes
    fake_images = gan_model.generate_images(latent_vector, fixed_labels)
    return fake_images.to(torch.float32)


def generate_class_gif(ini_config, gan_model, num_images: int):
    model_arch_config, data_config = ini_config['MODEL ARCHITECTURE'], ini_config['DATA']
    n_gpus = int(config['MACHINE']['ngpu'])
    device = create_model.get_device(n_gpus)
    num_classes = get_num_classes(data_config)

    latent_vector = create_latent_vector(data_config, model_arch_config, device)
    images_between_classes = 32
    class_vectors = []
    for class_a in range(num_classes-1):
        class_b = class_a + 1

        # Gets the class embeddings
        embed_a = gan_model.netG.get_class_embedding(torch.tensor(class_a, dtype=torch.int64, device=device))
        embed_b = gan_model.netG.get_class_embedding(torch.tensor(class_b, dtype=torch.int64, device=device))

        weights = torch.arange(0, images_between_classes) / images_between_classes
        for weight in weights:
            class_vectors.append((embed_b * weight) + (embed_a * 1-weight))

    class_vectors = torch.stack(class_vectors)
    num_batches_to_run = math.ceil(class_vectors.size(dim=0) / num_images)

    images_for_gif = torch.tensor([], dtype=torch.float32, device=device)
    for batch_index in range(num_batches_to_run):
        class_vectors_batch = class_vectors[batch_index * num_images:(batch_index + 1) * num_images]
        if class_vectors_batch.size(dim=0) < num_images:
            continue
        images_generated = gan_model.generate_images(latent_vector, None, class_embeddings=class_vectors_batch)
        images_for_gif = torch.concat((images_for_gif, images_generated), axis=0)

    return images_for_gif


def to_unnormalized_pil_images(images_as_tensor):
    pil_images = []
    to_pil_transform = transforms.ToPILImage()
    for image in images_as_tensor:
        pil_image = to_pil_transform(unnormalize(image))
        pil_images.append(pil_image)
    return pil_images


def load_gan_model(ini_config, load_model_dir: str, step_count: int):
    # Get configs
    model_arch_config, data_config = ini_config['MODEL ARCHITECTURE'], ini_config['DATA']
    train_config, machine_config = ini_config['TRAIN'], ini_config['MACHINE']
    n_gpus = int(machine_config['ngpu'])
    device = create_model.get_device(n_gpus)

    # Create model
    gan_model = create_model.create_gan_model(model_arch_config, data_config, train_config, device, n_gpus)

    generator_path, _ = os_helper.get_step_model(load_model_dir, os_helper.ModelType.GENERATOR, step_count)
    discrim_path, _ = os_helper.get_step_model(load_model_dir, os_helper.ModelType.DISCRIMINATOR, step_count)
    ema_path = None
    if gan_model.ema:
        ema_path, _ = os_helper.get_step_model(load_model_dir, os_helper.ModelType.EMA, step_count)

    gan_model.load(generator_path, discrim_path, ema_path)
    return gan_model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Generate images from a trained model')
    parser.add_argument('config_file_path', type=str,
                        help='The configuration file to inference with, configuration files end with .ini extension.\n')
    parser.add_argument('num_images', type=int,
                        help='the amount of images it will generate, this is the batch size')
    parser.add_argument('step_count', type=int,
                        help='specifies the step count of the model you want to load')
    parser.add_argument('--class_gif', type=bool,
                        help='If true will save a gif of the classes going through each other.  Some models may not '
                             'support this')
    args = parser.parse_args()

    if not os.path.exists(args.config_file_path):
        raise OSError('Configuration file path doesn\'t exist:' + args.config_file_path)
    config = ini_parser.read_with_defaults(args.config_file_path)

    will_restore_model = saver_and_loader.is_loadable_model(config)
    if will_restore_model:
        _, model_dir, img_dir, _ = saver_and_loader.get_run_directories(config)
    else:
        raise ValueError("Missing models directory in the run directory")

    loaded_gan_model = load_gan_model(config, model_dir, args.step_count)

    if args.class_gif:
        fake_imgs = generate_class_gif(config, loaded_gan_model, args.num_images)
        pil_images = to_unnormalized_pil_images(fake_imgs)
        output_path = os.path.join(img_dir, 'gif_images.png')
        first_pil_img = pil_images[0]
        first_pil_img.save(fp=output_path, format='GIF', append_images=pil_images,
                           save_all=True, duration=16, loop=0)
    else:
        fake_imgs = generate_batch_image(config, loaded_gan_model, args.num_images)

        output_path = os.path.join(img_dir, 'fake_images.png')
        saver_and_loader.save_images(fake_imgs, output_path)
