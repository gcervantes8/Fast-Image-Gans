import torch
import torch.nn as nn


class BaseDiscriminator(nn.Module):

    def __init__(self, base_width, base_height, upsample_layers, ndf, num_channels, num_classes):
        super(BaseDiscriminator, self).__init__()

    def set_channels_last(self):
        self = self.to(memory_format=torch.channels_last)