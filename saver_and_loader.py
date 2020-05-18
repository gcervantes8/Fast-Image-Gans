import os
import shutil
import torch
import torchvision.utils as torch_utils

import Generator
import Discriminator
import create_model


# Writes architecture details onto file in run directory
def save_architecture(generator, discriminator, save_dir):
    with open(os.path.join(save_dir, 'architecture.txt'), "w") as text_file:
        text_file.write(str(generator))
        text_file.write(str(discriminator))


def save_gan_files(run_dir):
    shutil.copy(Generator.__name__ + '.py', os.path.abspath(run_dir))
    shutil.copy(Discriminator.__name__ + '.py', os.path.abspath(run_dir))


def save_model(generator, discriminator, generator_path, discriminator_path):
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)


def save_training_images(loader, save_dir):
    # Save training images
    real_batch = next(iter(loader))

    # batch_images is of size (batch_size, num_channels, height, width)
    batch_images = real_batch[0]
    save_path = os.path.join(save_dir, 'training batch.png')
    save_images(batch_images, save_path)


# tensor should be of shape (batch_size, num_channels, height, width) as outputted by the torch DataLoader
def save_images(tensor, save_path):
    torch_utils.save_image(tensor, save_path, normalize=True)


def load_model(config, generator_path, discriminator_path):
    generator, discriminator, device = create_model.create_gan_instances(config)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()
    discriminator.load_state_dict(torch.load(discriminator_path))
    discriminator.eval()
    return generator, discriminator, device
