import torch
import torch.nn as nn

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
        self.conv_delta = nn.Conv2d(channels, block_channels_1, kernel_size=1, padding='same')
        self.conv_phi = nn.Conv2d(channels, block_channels_1, kernel_size=1, padding='same')
        self.conv_g = nn.Conv2d(channels, block_channels_2, kernel_size=1, padding='same')
        self.softmax = nn.Softmax(dim=1)  # Should not be dim 0, since that's batch
        self.conv_last = nn.Conv2d(block_channels_2, channels, kernel_size=1, padding='same')

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
        # [block_channels_1, B, H, W]
        phi_out = torch.permute(phi_out, (1, 0, 2, 3))
        # [B, block_channels_1, H*W]
        phi_out = torch.reshape(phi_out, [batch_size, self.block_channels_1, -1])

        # [B, H*W, H*W]
        first_mult_out = torch.bmm(delta_out, phi_out)

        first_mult_out = torch.reshape(first_mult_out, [batch_size, -1])

        softmax_mult = self.softmax(first_mult_out)
        softmax_mult = torch.reshape(softmax_mult, [batch_size, height * width, height * width])

        # conv g out is [B, block_channels_2, H, W]
        g_out = self.conv_g(nonlocal_input)

        # [B, H, W, block_channels_2]
        g_out = torch.permute(g_out, (0, 2, 3, 1))

        # [B, H*W, block_channels_2]
        g_out = torch.reshape(g_out, [batch_size, -1, self.block_channels_2])

        mult_out = torch.bmm(softmax_mult, g_out)
        mult_out = torch.reshape(mult_out, [batch_size, height, width, self.block_channels_2])
        # [B, C, H, W]
        mult_out = torch.permute(mult_out, (0, 3, 1, 2))

        # [B, C, H, W]
        return torch.add(nonlocal_input, self.conv_last(mult_out))
