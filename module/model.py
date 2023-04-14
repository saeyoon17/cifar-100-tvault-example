"""mobilenetv2 in pytorch
[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381

code source: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/mobilenetv2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                in_channels * t,
                in_channels * t,
                3,
                stride=stride,
                padding=1,
                groups=in_channels * t,
            ),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2(nn.Module):
    def __init__(self, class_num=100):
        super().__init__()
        scale = 3
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32 * scale, 1, padding=1),
            nn.BatchNorm2d(32 * scale),
            nn.ReLU6(inplace=True),
        )
        self.stage1 = LinearBottleNeck(32 * scale, 16 * scale, 1, 1)
        self.stage2 = self._make_stage(2, 16 * scale, 24 * scale, 2, 6)
        self.stage3 = self._make_stage(3, 24 * scale, 32 * scale, 2, 6)
        self.stage4 = self._make_stage(4, 32 * scale, 64 * scale, 2, 6)
        self.stage5 = self._make_stage(3, 64 * scale, 96 * scale, 1, 6)
        self.stage6 = self._make_stage(3, 96 * scale, 160 * scale, 1, 6)
        self.stage7 = LinearBottleNeck(160 * scale, 320 * scale, 1, 6)
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(320 * scale, 1280 * scale, 1),
            nn.BatchNorm2d(1280 * scale),
            nn.ReLU6(inplace=True),
        )

        self.conv2 = nn.Conv2d(1280 * scale, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.dropout(self.stage3(x))
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.dropout(self.stage6(x))
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)


def mobilenetv2():
    return MobileNetV2()
