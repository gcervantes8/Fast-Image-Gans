import torch.nn as nn


class BaseGenerator(nn.Module):

    def __init__(self, num_gpu, base_width, base_height, upsample_layers, latent_vector_size, ngf, num_channels,
                 num_classes):
        self.n_gpu = num_gpu
        super(BaseGenerator, self).__init__()
