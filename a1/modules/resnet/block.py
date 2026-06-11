import torch
import torch.nn as nn


def conv3x3(input_dim, output_dim, stride=1):
    return nn.Conv2d(
        input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(input_dim, output_dim, stride=1):
    return nn.Conv2d(
        input_dim, output_dim, kernel_size=1, stride=stride, padding=0, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_dim, hidden_dim, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.residual = nn.Sequential(
            conv3x3(input_dim, hidden_dim, stride),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            conv3x3(hidden_dim, hidden_dim * BasicBlock.expansion),
            nn.BatchNorm2d(hidden_dim * BasicBlock.expansion),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)
        if self.downsample is not None:
            skip_connection = self.downsample(x)
        else:
            skip_connection = x
        out = self.relu(residual + skip_connection)
        return out


class BottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(self, input_dim, hidden_dim, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.residual = nn.Sequential(
            conv1x1(input_dim, hidden_dim, stride),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            conv3x3(hidden_dim, hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            conv1x1(hidden_dim, hidden_dim * BottleNeckBlock.expansion),
            nn.BatchNorm2d(hidden_dim * BottleNeckBlock.expansion),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)
        if self.downsample is not None:
            skip_connection = self.downsample(x)
        else:
            skip_connection = x
        # print('residual.shape', residual.shape)
        # print('skip_connection.shape', skip_connection.shape)
        out = self.relu(residual + skip_connection)
        return out
