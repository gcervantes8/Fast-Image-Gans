import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class DeepResDown(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, pooling=True):
        super(DeepResDown, self).__init__()
        self.same_dim = in_channels == out_channels
        if not pooling and not self.same_dim:
            raise ValueError("DeepResDown in channel and out channel must be equal if the pooling flag is false")
        if not (self.same_dim or int(out_channels/2) == in_channels):
            raise ValueError("DeepResDown out_channel should be half of in_channels or equal to in_channels")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.relu_b1 = nn.ReLU()
        self.conv_b2 = spectral_norm(nn.Conv2d(in_channels, int(out_channels/4), kernel_size=1, padding='same'),
                                     eps=1e-04)
        nn.init.orthogonal_(self.conv_b2.weight)
        self.relu_b3 = nn.ReLU()
        self.conv_b4 = spectral_norm(nn.Conv2d(int(out_channels/4), int(out_channels/4), kernel_size=3, padding='same'),
                                     eps=1e-04)
        nn.init.orthogonal_(self.conv_b4.weight)
        self.relu_b5 = nn.ReLU()
        self.conv_b6 = spectral_norm(nn.Conv2d(int(out_channels/4), int(out_channels/4), kernel_size=3, padding='same'),
                                     eps=1e-04)
        nn.init.orthogonal_(self.conv_b6.weight)
        self.relu_b7 = nn.ReLU()
        self.conv_b9 = spectral_norm(nn.Conv2d(int(out_channels/4), out_channels, kernel_size=1, padding='same'),
                                     eps=1e-04)
        nn.init.orthogonal_(self.conv_b9.weight)
        if pooling:
            self.avg_pooling_a1 = nn.AvgPool2d(kernel_size=2)
            self.avg_pooling_b8 = nn.AvgPool2d(kernel_size=2)
        if not self.same_dim:
            self.conv_a2 = spectral_norm(nn.Conv2d(in_channels, out_channels-in_channels, kernel_size=1, padding='same'),
                                         eps=1e-04)
            nn.init.orthogonal_(self.conv_a2.weight)

    # res_input is of size (B, C, H, W)
    def forward(self, res_input):

        if self.pooling:
            # out_a if pooling (B, C, H/2, W/2)
            out_a = self.avg_pooling_a1(res_input)

        else:
            out_a = res_input

        if not self.same_dim:
            out_a_temp = self.conv_a2(out_a)
            out_a = torch.cat((out_a, out_a_temp), dim=1)

        out_b = self.relu_b1(res_input)
        # out_b is (B, ndf, H, W)
        out_b = self.conv_b2(out_b)
        out_b = self.relu_b3(out_b)
        # out_b is (B, ndf, H, W)
        out_b = self.conv_b4(out_b)
        out_b = self.relu_b5(out_b)
        out_b = self.conv_b6(out_b)
        out_b = self.relu_b7(out_b)
        if self.pooling:
            out_b = self.avg_pooling_b8(out_b)
        out_b = self.conv_b9(out_b)
        output = out_a + out_b
        return output
