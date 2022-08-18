import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

# Attention block based on https://arxiv.org/pdf/1805.08318.pdf
# And the original paper https://arxiv.org/pdf/1711.07971.pdf


class NonLocalBlock(nn.Module):

    def __init__(self, channels, block_channels_1=None, block_channels_2=None):
        super(NonLocalBlock, self).__init__()

        # By default, divides original channels by 8 and 2, similar to original biggan paper
        if block_channels_1 is None:
            block_channels_1 = int(channels/8)
        if block_channels_2 is None:
            block_channels_2 = int(channels/2)

        self.channels = channels
        self.block_channels_1 = block_channels_1
        self.block_channels_2 = block_channels_2
        self.conv_delta = spectral_norm(nn.Conv2d(channels, block_channels_1, kernel_size=1, padding='same'), eps=1e-04)
        nn.init.orthogonal_(self.conv_delta.weight)
        self.conv_phi = spectral_norm(nn.Conv2d(channels, block_channels_1, kernel_size=1, padding='same'), eps=1e-04)
        nn.init.orthogonal_(self.conv_phi.weight)
        self.pooling_phi = nn.MaxPool2d(kernel_size=2)
        self.conv_g = spectral_norm(nn.Conv2d(channels, block_channels_2, kernel_size=1, padding='same'), eps=1e-04)
        nn.init.orthogonal_(self.conv_g.weight)
        self.pooling_g = nn.MaxPool2d(kernel_size=2)
        self.softmax = nn.Softmax(dim=-1)  # Should not be dim 0, since that's batch
        self.conv_last = spectral_norm(nn.Conv2d(block_channels_2, channels, kernel_size=1, padding='same'), eps=1e-04)
        nn.init.orthogonal_(self.conv_last.weight)
        self.gamma = nn.Parameter(torch.zeros(1))

        # res_input is of size (B, C, H, W)
    def forward(self, nonlocal_input):
        batch_size = nonlocal_input.size(0)
        height = nonlocal_input.size(2)
        width = nonlocal_input.size(3)

        # conv delta out is [B, block_channels, H, W]
        delta_out = self.conv_delta(nonlocal_input)
        # [B, H, W, block_channels_1]
        delta_out = torch.permute(delta_out, (0, 2, 3, 1))
        # [B, H*W, block_channels_1]
        delta_out = torch.reshape(delta_out, [batch_size, -1, self.block_channels_1])

        # conv phi out is [B, block_channels, H, W]
        phi_out = self.conv_phi(nonlocal_input)
        # [B, block_channels, H/2, W/2]
        phi_out = self.pooling_phi(phi_out)

        # [block_channels_1, B, H/2, W/2]
        phi_out = torch.permute(phi_out, (1, 0, 2, 3))

        # [B, block_channels_1, H*W/4]
        phi_out = torch.reshape(phi_out, [batch_size, self.block_channels_1, -1])

        # [B, H*W, H*W/4]
        mult_out = torch.bmm(delta_out, phi_out)

        # [B, H*W, H*W/4]
        mult_out = self.softmax(mult_out)

        # conv g out is [B, block_channels_2, H, W]
        g_out = self.conv_g(nonlocal_input)

        # [B, block_channels_2, H/2, W/2]
        g_out = self.pooling_phi(g_out)

        # [B, H/2, W/2, block_channels_2]
        g_out = torch.permute(g_out, (0, 2, 3, 1))

        # [B, H*W/4, block_channels_2]
        g_out = torch.reshape(g_out, [batch_size, -1, self.block_channels_2])

        # [B, H*W, block_channels_2]
        mult_out = torch.bmm(mult_out, g_out)
        mult_out = torch.reshape(mult_out, [batch_size, height, width, self.block_channels_2])
        # [B, C, H, W]
        mult_out = torch.permute(mult_out, (0, 3, 1, 2))

        # [B, C, H, W]
        return torch.add(self.gamma * self.conv_last(mult_out), nonlocal_input)
