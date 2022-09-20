import torch.nn as nn


class BaseDiscriminator(nn.Module):

    def __init__(self, num_gpu, base_width, base_height, upsample_layers, ndf, num_channels, num_classes):
        self.n_gpu = num_gpu
        super(BaseDiscriminator, self).__init__()
