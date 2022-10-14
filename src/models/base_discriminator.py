import torch.nn as nn


class BaseDiscriminator(nn.Module):

    def __init__(self, base_width, base_height, upsample_layers, ndf, num_channels, num_classes):
        super(BaseDiscriminator, self).__init__()
