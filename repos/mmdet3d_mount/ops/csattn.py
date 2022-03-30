import torch
import torch.nn as nn


class ChannelAttn(nn.Module):

    def __init__(self, inplanes, ratio=4):
        super(ChannelAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_comm = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // ratio, inplanes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.mlp_comm(self.avg_pool(x))
        maxout = self.mlp_comm(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttn(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttn, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CSAttn(nn.Module):

    def __init__(self, planes):
        super(CSAttn, self).__init__()
        self.ca = ChannelAttn(planes)
        self.sa = SpatialAttn()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


if __name__ == '__main__':
    inputs = torch.rand(2, 128, 320, 96)
    csattn = CSAttn(planes=128)
    print(csattn(inputs).shape)
