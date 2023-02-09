import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from src.layers.batchnorm_conditional import ConditionalBatchNorm2d


class ResUp(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, latent_embed_vector_size: int):
        super(ResUp, self).__init__()

        self.upsample_a1 = nn.Upsample(scale_factor=2)
        self.conv_a2 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1), eps=1e-04)
        nn.init.orthogonal_(self.conv_a2.weight)

        self.cond_batch_norm_b1 = ConditionalBatchNorm2d(in_channels, latent_embed_vector_size)
        self.relu_b2 = nn.ReLU()
        self.upsample_b3 = nn.Upsample(scale_factor=2)
        # Set bias to False because it is followed by a batch norm (speed up)
        self.conv_b4 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False), eps=1e-04)
        nn.init.orthogonal_(self.conv_b4.weight)
        self.cond_batch_norm_b5 = ConditionalBatchNorm2d(out_channels, latent_embed_vector_size)
        self.relu_b6 = nn.ReLU()
        self.conv_b7 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'), eps=1e-04)
        nn.init.orthogonal_(self.conv_b7.weight)

    # res_input is of size (B, C, H, W)
    def forward(self, res_input, latent_embed_vector):
        out_a = self.upsample_a1(res_input)
        out_a = self.conv_a2(out_a)
        out_b = self.cond_batch_norm_b1(res_input, latent_embed_vector)
        out_b = self.relu_b2(out_b)
        out_b = self.upsample_b3(out_b)
        out_b = self.conv_b4(out_b)
        out_b = self.cond_batch_norm_b5(out_b, latent_embed_vector)
        out_b = self.relu_b6(out_b)
        out_b = self.conv_b7(out_b)

        return out_a + out_b
