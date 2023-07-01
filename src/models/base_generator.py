import torch
import torch.nn as nn


class BaseGenerator(nn.Module):

    def __init__(self, base_width, base_height, upsample_layers, latent_vector_size, ngf, num_channels,
                 num_classes):
        super(BaseGenerator, self).__init__()

    def set_channels_last(self):
        self = self.to(memory_format=torch.channels_last)