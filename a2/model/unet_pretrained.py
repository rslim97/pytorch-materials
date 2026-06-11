import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def conv3x3(input_dim, output_dim, stride=1, padding=0, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        input_dim, output_dim, kernel_size=3, stride=stride, padding=padding, bias=bias
    )


class twoConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # todo
        # initialize the block
        self.block = nn.Sequential(
            conv3x3(input_dim, output_dim, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            conv3x3(output_dim, output_dim, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # todo
        # implement the forward path
        return self.block(x)


class twoConvBlockUp(twoConvBlock):
    def __init__(self, input_dim, output_dim):
        # output_dim = input_dim // 2
        super().__init__(input_dim, output_dim)

    def forward(self, x, x_skip):
        # concat
        # x_skip: input from skip connection
        # x: input from upsample
        diff_y = x_skip.shape[2] - x.shape[2]
        diff_x = x_skip.shape[3] - x.shape[3]
        # print('diff_y', diff_y)
        # print('diff_x', diff_x)
        x = torch.nn.functional.pad(
            x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x_out = torch.cat([x, x_skip], dim=1)
        # x_out = x + x_skip

        return self.block(x_out)


class upBlock(nn.Module):
    def __init__(
        self, upsample_input_dim, upsample_output_dim, input_dim=None, output_dim=None
    ):
        super().__init__()
        if not input_dim:
            input_dim = upsample_input_dim
        if not output_dim:
            output_dim = upsample_output_dim
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv3x3(upsample_input_dim, upsample_output_dim),
        )
        # self.upsample = nn.Upsample(scale_factor=2)
        self.twoconvblock = twoConvBlockUp(input_dim, output_dim)

    def forward(self, x, x_skip):
        # print('x_lower', x_lower.shape)
        x1 = self.upsample(x)
        # print('x_upsampled', x1.shape)
        # print('x_skip', x_skip.shape)
        x2 = self.twoconvblock(x1, x_skip)
        # print('x_out.shape', x2.shape)
        # print('\n')
        return x2


class Unet_pretrained_encoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.input_block = nn.Sequential(*list(self.encoder.children()))[:3]  # for RGB images
        self.input_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=64),
        )  # for grayscale images
        self.input_pool = list(self.encoder.children())[3]
        down_blocks = []
        up_blocks = []
        for bottleneck in list(self.encoder.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = conv3x3(2048, 2048)
        up_blocks.append(upBlock(2048, 1024))
        up_blocks.append(upBlock(1024, 512))
        up_blocks.append(upBlock(512, 256))
        up_blocks.append(upBlock(256, 128, input_dim=128 + 64, output_dim=64))
        # up_blocks.append(upBlock(64, 32, input_dim=32+3, output_dim=3))  # for RGB images
        up_blocks.append(
            upBlock(64, 32, input_dim=32 + 1, output_dim=3)
        )  # for grayscale images
        # up_blocks.append(upBlock(128))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.final_block = nn.Sequential(
            nn.Conv2d(3, n_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        fmaps = dict()
        fmaps["layer_0"] = x  # 64 channels
        x = self.input_block(x)  # out: 64 channels
        fmaps["layer_1"] = x  # 64 channels
        x = self.input_pool(x)  # out: 64 channels
        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            # No need to save fmap in semantic pathway
            if i == 5:
                continue
            fmaps[f"layer_{i}"] = x

        # fmaps[f'layer_{5}'] = x
        # print(fmaps.keys())
        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 2):
            skip_connection = fmaps[f"layer_{5 - i + 1}"]
            # print('skip_connection.shape', skip_connection.shape)
            # print('x.shape', x.shape)
            x = block(x, skip_connection)

        x = self.final_block(x)
        del fmaps
        return x


if __name__ == "__main__":
    # # weights = ResNet50_Weights.DEFAULT
    # model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # # print(model)

    # down_blocks = []
    # input_block = nn.Sequential(*list(model.children()))[:3]
    # # print('model.children', model.children)
    # # model.children()
    # print('input_block', input_block)

    # for bottleneck in list(model.children()):
    #     if isinstance(bottleneck, nn.Sequential):
    #         down_blocks.append(bottleneck)

    # down_blocks = nn.ModuleList(down_blocks)

    # # for i, block in enumerate(down_blocks, 1000):
    # #     print(f'block {i}: {block}')

    # # print('model', model)

    # # for m in model.modules():
    # #     print('m ', m)

    # for c in model.children():
    #     print('c', c)

    model = Unet_pretrained_encoder(n_classes=2).cuda()
    print(model)
    # x = torch.rand(4, 3, 256, 256).cuda()
    x = torch.rand(4, 1, 256, 256).cuda()
    out = model(x)
    print(out.shape)
