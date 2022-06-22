import torch.nn as nn


class BaseDiscriminator(nn.Module):

    def __init__(self, num_gpu, ndf, num_channels):
        self.n_gpu = num_gpu
        super(BaseDiscriminator, self).__init__()
