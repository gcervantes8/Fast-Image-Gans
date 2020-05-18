import torch
import torch.nn as nn
import torchvision.datasets as torch_data_set
import torchvision.transforms as transforms

from src import Generator, Discriminator


# Create the data-set using an image folder
def create_data_loader(config, data_dir):

    image_size = int(config['CONFIGS']['image_size'])
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


def create_gan_instances(config):

    n_gpu = int(config['CONFIGS']['ngpu'])
    latent_vector_size = int(config['CONFIGS']['latent_vector_size'])
    ngf = int(config['CONFIGS']['ngf'])
    ndf = int(config['CONFIGS']['ndf'])
    num_channels = int(config['CONFIGS']['num_channels'])

    device = torch.device('cuda:0' if (torch.cuda.is_available() and n_gpu > 0) else 'cpu')

    # Create the generator and discriminator
    generator = Generator.Generator(n_gpu, latent_vector_size, ngf, num_channels).to(device)
    discriminator = Discriminator.Discriminator(n_gpu, ndf, num_channels).to(device)

    generator = _handle_multiple_gpus(generator, n_gpu, device)
    discriminator = _handle_multiple_gpus(discriminator, n_gpu, device)
    return generator, discriminator, device


# Handle multi-gpu if desired, returns the new instance that is multi-gpu capable
def _handle_multiple_gpus(torch_obj, num_gpu, device):
    if (device.type == 'cuda') and (num_gpu > 1):
        return nn.DataParallel(torch_obj, list(range(num_gpu)))
    else:
        return torch_obj


# custom weights initialization called on netG and netD
def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)