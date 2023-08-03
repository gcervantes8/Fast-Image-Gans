import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Decription in paper is incorrect (https://github.com/ajbrock/BigGAN-PyTorch/issues/17), corrected based on BigGAN Pytorch by ajbrock
# Previously implemented based on https://arxiv.org/pdf/1805.08318.pdf and original paper https://arxiv.org/pdf/1711.07971.pdf


class NonLocalBlock(nn.Module):

    def __init__(self, channels, block_channels_1=None, block_channels_2=None):
        super(NonLocalBlock, self).__init__()

        # By default, divides original channels by 8 and 2, similar to original biggan paper
        if block_channels_1 is None:
            block_channels_1 = int(channels/8)
            block_channels_1 = 1 if block_channels_1 < 1 else block_channels_1
        if block_channels_2 is None:
            block_channels_2 = int(channels/2)
            block_channels_2 = 1 if block_channels_2 < 1 else block_channels_2

        self.channels = channels
        self.block_channels_1 = block_channels_1
        self.block_channels_2 = block_channels_2
        self.conv_delta = spectral_norm(nn.Conv2d(channels, block_channels_1, kernel_size=1, padding='same', bias=False), eps=1e-04)
        nn.init.orthogonal_(self.conv_delta.weight)
        self.conv_phi = spectral_norm(nn.Conv2d(channels, block_channels_1, kernel_size=1, padding='same', bias=False), eps=1e-04)
        nn.init.orthogonal_(self.conv_phi.weight)
        self.pooling_phi = nn.MaxPool2d(kernel_size=2)
        self.conv_g = spectral_norm(nn.Conv2d(channels, block_channels_2, kernel_size=1, padding='same', bias=False), eps=1e-04)
        nn.init.orthogonal_(self.conv_g.weight)
        self.pooling_g = nn.MaxPool2d(kernel_size=2)
        self.softmax = nn.Softmax(dim=-1)  # Should not be dim 0, since that's batch
        self.conv_last = spectral_norm(nn.Conv2d(block_channels_2, channels, kernel_size=1, padding='same', bias=False), eps=1e-04)
        nn.init.orthogonal_(self.conv_last.weight)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

        # res_input is of size (B, C, H, W)
    def forward(self, nonlocal_input, *args):
        batch_size = nonlocal_input.size(0)
        height = nonlocal_input.size(2)
        width = nonlocal_input.size(3)

        # Convolutions and pooling
        # [B, block_channels_1, H, W]
        delta_out = self.conv_delta(nonlocal_input)
        # [B, block_channels_1, H/2, W/2]
        phi_out = self.pooling_phi(self.conv_phi(nonlocal_input))
        # [B, block_channels_2, H/2, W/2]
        g_out = self.pooling_g(self.conv_g(nonlocal_input))

        # Reshapes
        # [B, block_channels_1, H*W]
        delta_out = delta_out.view(batch_size, self.block_channels_1, height*width)
        # [B, block_channels_1, H*W/4]
        phi_out = phi_out.view(batch_size, self.block_channels_1, -1)
        # [B, block_channels_2, H*W/4]
        g_out = g_out.view(batch_size, self.block_channels_2, -1)
       
        # [B, H*W, H*W/4]
        mult_out = self.softmax(torch.bmm(torch.transpose(delta_out, 1, 2), phi_out))

        # [B, block_channels_2, H, W]
        mult_out = self.conv_last(torch.bmm(g_out, torch.transpose(mult_out, 1, 2)).view(batch_size, self.block_channels_2, height, width))

        # [B, C, H, W]
        return torch.add(self.gamma * mult_out, nonlocal_input)
