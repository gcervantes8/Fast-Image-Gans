
# Loads trained models, and predicts

import torch
from src import ini_parser, os_helper

generator_path = 'output/P9MDIW/generator_epoch_3.pt'
discriminator_path = 'output/P9MDIW/discriminator_epoch_3.pt'
config_file_path = 'model_config.ini'

ini_config = ini_parser.read(config_file_path)
generator, discriminator, device = os_helper.load_model(ini_config, generator_path, discriminator_path)
print(generator)
latent_vector_size = int(ini_config['CONFIGS']['latent_vector_size'])

latent_vector = torch.randn(64, latent_vector_size, 1, 1, device=device)

fake_images = generator.forward(latent_vector)
os_helper.save_images(fake_images, 'output/P9MDIW/fake_images.png')

