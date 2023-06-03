import torch.nn as nn
from torch.nn.utils import spectral_norm
from src.layers.batchnorm_conditional import ConditionalBatchNorm2d


class DeepResUp(nn.Module):

    def __init__(self, in_channels, out_channels, latent_embed_vector_size, upsample=True):
        super(DeepResUp, self).__init__()

        self.upsample = upsample
        if in_channels < out_channels:
            raise ValueError("Out Channels should be smaller or equal to the In Channels in DeepResUp")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_a1 = nn.Upsample(scale_factor=2)

        self.cond_batch_norm_b1 = ConditionalBatchNorm2d(in_channels, latent_embed_vector_size)
        self.relu_b2 = nn.ReLU()
        # Set bias to False because it is followed by a batch norm (speed up)
        self.conv_b3 = spectral_norm(nn.Conv2d(in_channels, int(in_channels/4), kernel_size=1, padding='same', bias=False), eps=1e-04)
        nn.init.orthogonal_(self.conv_b3.weight)
        self.cond_batch_norm_b4 = ConditionalBatchNorm2d(int(in_channels/4), latent_embed_vector_size)
        self.relu_b5 = nn.ReLU()
        self.upsample_b6 = nn.Upsample(scale_factor=2)
        self.conv_b7 = spectral_norm(nn.Conv2d(int(in_channels/4), int(in_channels/4), kernel_size=3, padding='same', bias=False), eps=1e-04)
        nn.init.orthogonal_(self.conv_b7.weight)
        self.cond_batch_norm_b8 = ConditionalBatchNorm2d(int(in_channels/4), latent_embed_vector_size)
        self.relu_b9 = nn.ReLU()
        self.conv_b10 = spectral_norm(nn.Conv2d(int(in_channels/4), int(in_channels/4), kernel_size=3, padding='same', bias=False), eps=1e-04)
        nn.init.orthogonal_(self.conv_b10.weight)
        self.cond_batch_norm_b11 = ConditionalBatchNorm2d(int(in_channels/4), latent_embed_vector_size)
        self.relu_b12 = nn.ReLU()
        self.conv_b13 = spectral_norm(nn.Conv2d(int(in_channels/4), out_channels, kernel_size=1, padding='same'), eps=1e-04)
        nn.init.orthogonal_(self.conv_b13.weight)

    # res_input is of size (B, C, H, W)
    def forward(self, forward_input):

        res_input, latent_embed_vector = forward_input
        # (B, out_channels, H, W)
        out_a = res_input[:, :self.out_channels, :, :]

        if self.upsample:
            # (B, out_channels, H*2, W*2)
            out_a = self.upsample_a1(out_a)

        out_b = self.cond_batch_norm_b1(res_input, latent_embed_vector)
        out_b = self.relu_b2(out_b)
        out_b = self.conv_b3(out_b)
        out_b = self.cond_batch_norm_b4(out_b, latent_embed_vector)
        out_b = self.relu_b5(out_b)

        if self.upsample:
            # (B, out_channels, H*2, W*2)
            out_b = self.upsample_b6(out_b)
        out_b = self.conv_b7(out_b)
        out_b = self.cond_batch_norm_b8(out_b, latent_embed_vector)
        out_b = self.relu_b9(out_b)
        out_b = self.conv_b10(out_b)
        out_b = self.cond_batch_norm_b11(out_b, latent_embed_vector)
        out_b = self.relu_b12(out_b)
        out_b = self.conv_b13(out_b)

        output = out_a + out_b
        return output, latent_embed_vector
