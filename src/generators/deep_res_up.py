import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class DeepResUp(nn.Module):

    def __init__(self, in_channels, out_channels, upsample=True):
        super(DeepResUp, self).__init__()

        self.upsample = upsample
        if in_channels < out_channels:
            raise ValueError("Out Channels should be smaller or equal to the In Channels in DeepResUp")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_a1 = nn.Upsample(scale_factor=2)

        self.batch_norm_b1 = nn.BatchNorm2d(num_features=in_channels, eps=1e-04)
        self.relu_b2 = nn.ReLU()
        self.conv_b3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        nn.init.orthogonal_(self.conv_b3.weight)
        self.batch_norm_b4 = nn.BatchNorm2d(num_features=out_channels, eps=1e-04)
        self.relu_b5 = nn.ReLU()
        self.upsample_b6 = nn.Upsample(scale_factor=2)
        self.conv_b7 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'), eps=1e-04)
        nn.init.orthogonal_(self.conv_b7.weight)
        self.batch_norm_b8 = nn.BatchNorm2d(num_features=out_channels, eps=1e-04)
        self.relu_b9 = nn.ReLU()
        self.conv_b10 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'), eps=1e-04)
        nn.init.orthogonal_(self.conv_b10.weight)
        self.batch_norm_b11 = nn.BatchNorm2d(num_features=out_channels, eps=1e-04)
        self.relu_b12 = nn.ReLU()
        self.conv_b13 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding='same')
        nn.init.orthogonal_(self.conv_b13.weight)

    # res_input is of size (B, C, H, W)
    def forward(self, res_input):
        # (B, out_channels, H, W)
        out_a = res_input[:, :self.out_channels, :, :]

        if self.upsample:
            # (B, out_channels, H*2, W*2)
            out_a = self.upsample_a1(out_a)

        out_b = self.batch_norm_b1(res_input)
        out_b = self.relu_b2(out_b)
        out_b = self.conv_b3(out_b)
        out_b = self.batch_norm_b4(out_b)
        out_b = self.relu_b5(out_b)

        if self.upsample:
            # (B, out_channels, H*2, W*2)
            out_b = self.upsample_b6(out_b)
        out_b = self.conv_b7(out_b)
        out_b = self.batch_norm_b8(out_b)
        out_b = self.relu_b9(out_b)
        out_b = self.conv_b10(out_b)
        out_b = self.batch_norm_b11(out_b)
        out_b = self.relu_b12(out_b)
        out_b = self.conv_b13(out_b)

        output = out_a + out_b
        return output
