import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SiamFC', 'SiamConvFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out


class SiamConvFC(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 num_convs=1,
                 kernel_size=1,
                 out_scale=0.001):
        super(SiamConvFC, self).__init__()
        self.out_scale = out_scale
        z_convs = []
        x_convs = []
        last_channels = in_channels
        for i in range(num_convs):
            z_convs.append(nn.Conv2d(last_channels, channels, kernel_size))
            x_convs.append(nn.Conv2d(last_channels, channels, kernel_size))
            last_channels = channels
        self.z_convs = nn.Sequential(*z_convs)
        self.x_convs = nn.Sequential(*x_convs)

    def forward(self, z, x):
        z = self.z_convs(z)
        x = self.x_convs(x)
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out
