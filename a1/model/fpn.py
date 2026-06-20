# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

"""Modified from https://github.com/silkylove/ObjectDetection/blob/master/models/retinanet/fpn.py """


def conv3x3(input_dim, output_dim, stride=1):
    return nn.Conv2d(
        input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(input_dim, output_dim, stride=1):
    return nn.Conv2d(
        input_dim, output_dim, kernel_size=1, stride=stride, padding=0, bias=False
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_dim, hidden_dim, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(input_dim, hidden_dim, stride)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = conv3x3(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = conv1x1(hidden_dim, self.expansion * hidden_dim)
        self.bn3 = nn.BatchNorm2d(self.expansion * hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or input_dim != self.expansion * hidden_dim:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    self.expansion * hidden_dim,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * hidden_dim),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        ## no relu here
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_plane = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ↑ layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # ↓ layers
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # ↑
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(self.relu(p6))

        # ↓
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        return p3, p4, p5, p6, p7

    def _make_layer(self, block, hidden_dim, num_blocks, stride):
        strides = [stride] + [1] * (
            num_blocks - 1
        )  # List of strides: e.g. [stride, 1, 1, 1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_plane, hidden_dim, stride))
            self.in_plane = hidden_dim * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y


def FPN50():
    return FPN(Bottleneck, [3, 4, 6, 3])


def FPN101():
    return FPN(Bottleneck, [3, 4, 23, 3])


def FPN152():
    return FPN(Bottleneck, [3, 8, 36, 3])


def test():
    net = FPN50()
    fms = net(torch.randn(1, 3, 640, 640))
    for fm in fms:
        print(fm.size())
