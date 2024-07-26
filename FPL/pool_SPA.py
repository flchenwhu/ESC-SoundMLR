import torch
import torch.nn as nn
import torch.nn.functional as F
from FPL.specpool2d import SpectralPool2d






class SpectralPoolingAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(SpectralPoolingAttention, self).__init__()
        mid_channels = max(out_channels // reduction, 1)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=-1)  # mean over frequency
        y = self.conv1(y.unsqueeze(-1))
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x

class Pooling_Layer(nn.Module):
    def __init__(self, in_channels, factor=0.75, reduction=16):
        super(Pooling_Layer, self).__init__()
        self.factor = factor
        self.spectral_pool = SpectralPool2d(scale_factor=(factor, 1))  # 仅在时间轴上缩放
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.attention = SpectralPoolingAttention(in_channels, in_channels, reduction)

    def forward(self, x):
        identity = x
        x = self.spectral_pool(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.attention(x)

        # 如果identity与x的尺寸不同，需要调整identity的尺寸
        if x.size(2) != identity.size(2):  # 检查时间维度是否匹配
            identity = F.interpolate(identity, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        x += identity
        return x






