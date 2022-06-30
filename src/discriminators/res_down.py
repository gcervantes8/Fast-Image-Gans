import torch.nn as nn


class ResDown(nn.Module):

    def __init__(self, in_channels, out_channels, pooling=True):
        super(ResDown, self).__init__()

        self.pooling = pooling
        self.conv_a1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        nn.init.orthogonal_(self.conv_a1.weight)
        self.relu_b1 = nn.ReLU()
        self.conv_b2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        nn.init.orthogonal_(self.conv_b2.weight)
        self.relu_b3 = nn.ReLU()
        self.conv_b4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        nn.init.orthogonal_(self.conv_b4.weight)
        if pooling:
            self.avg_pooling_a2 = nn.AvgPool2d(kernel_size=2)
            self.avg_pooling_b5 = nn.AvgPool2d(kernel_size=2)

    # res_input is of size (B, C, H, W)
    def forward(self, res_input):

        out_a = self.conv_a1(res_input)
        # out_a is (B, ndf, H, W)

        if self.pooling:
            # out_a is (B, ndf, H/2, W/2)
            out_a = self.avg_pooling_a2(out_a)

        out_b = self.relu_b1(res_input)
        # out_b is (B, ndf, H, W)
        out_b = self.conv_b2(out_b)
        out_b = self.relu_b3(out_b)
        # out_b is (B, ndf, H, W)
        out_b = self.conv_b4(out_b)

        if self.pooling:
            # out_b is (B, ndf, H/2, W/2)
            out_b = self.avg_pooling_b5(out_b)

        output = out_a + out_b
        return output
