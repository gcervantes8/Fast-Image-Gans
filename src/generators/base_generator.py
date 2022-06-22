import torch.nn as nn


class BaseGenerator(nn.Module):

    def __init__(self, num_gpu, latent_vector_size, ngf, num_channels):
        self.n_gpu = num_gpu
        super(BaseGenerator, self).__init__()
