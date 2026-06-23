import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, padding):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv2d(x)


class DeConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.ConvTranspose2d(
                input_dim,
                output_dim,
                kernel_size,
                stride=stride,
                padding=1,
                output_padding=0,
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv2d(x)


class SPP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_1 = nn.functional.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        x_2 = nn.functional.max_pool2d(x, kernel_size=9, stride=1, padding=4)
        x_3 = nn.functional.max_pool2d(x, kernel_size=13, stride=1, padding=6)
        return torch.cat([x, x_1, x_2, x_3], dim=1)
