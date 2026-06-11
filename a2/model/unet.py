import torch
import torch.nn as nn
import torch.nn.functional as F


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


class twoConvBlockDown(twoConvBlock):
    expansion = 2

    def __init__(self, input_dim):
        output_dim = input_dim * twoConvBlockDown.expansion
        super().__init__(input_dim, output_dim)
        # self.block = nn.Sequential(
        #     conv3x3(input_dim, input_dim * twoConvBlockM.expansion,
        #             stride=1, padding=0, bias=False),
        #     nn.ReLU(inplace=True),
        #     conv3x3(input_dim, input_dim * twoConvBlockM.expansion,
        #             stride=1, padding=0, bias=False),
        #     nn.ReLU(inplace=True),
        # )


class twoConvBlockUp(twoConvBlock):
    def __init__(self, input_dim):
        output_dim = input_dim // 2
        super().__init__(input_dim, output_dim)

    def forward(self, x1, x2):
        # concat
        # x1: input from skip connection
        # x2: input from upsample
        # print('x1', x1.shape)
        # print('x2', x2.shape)
        diff_y = x1.shape[2] - x2.shape[2]
        diff_x = x1.shape[3] - x2.shape[3]
        x2 = torch.nn.functional.pad(
            x2, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([x1, x2], dim=1)
        return self.block(x)


class downBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            twoConvBlockDown(input_dim),
        )

    def forward(self, x):
        return self.block(x)


class upBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv3x3(input_dim, input_dim // 2),
        )
        self.twoconvblock = twoConvBlockUp(input_dim)

    def forward(self, x_skip, x_lower):
        # print('x_lower', x_lower.shape)
        x_upsample = self.upsample(x_lower)
        # print('x_upsample', x_upsample.shape)
        # print('x_skip', x_skip.shape)
        x1 = self.twoconvblock(x_skip, x_upsample)
        # print('x1.shape', x1.shape)
        return x1


class downStep(nn.Module):
    def __init__(self):
        super().__init__()
        # todo
        # initialize the down path
        layers = [twoConvBlock(1, 64)]
        in_planes = [64, 128, 256, 512]
        self.hooks = {}
        for i in range(len(in_planes)):
            layers.append(downBlock(in_planes[i]))

        for i in range(len(layers) - 1):
            layers[i].register_forward_hook(self.forward_hooks(f"down.{i}"))

        self.down = nn.Sequential(*layers)

    def forward_hooks(self, name):
        # Helper function to store features from the hook
        def hook(model, input, output):
            self.hooks[name] = output

        return hook

    def forward(self, x):
        # todo
        # implement the forward path
        # if len(self.hooks) > 0:
        #     print('self.hooks', self.hooks['down.0'].shape)
        #     print('self.hooks', self.hooks['down.1'].shape)
        #     print('self.hooks', self.hooks['down.2'].shape)
        #     print('self.hooks', self.hooks['down.3'].shape)
        return self.down(x), self.hooks


class upStep(nn.Module):
    def __init__(self):
        super().__init__()
        # todo
        # initialize the up path
        self.layers = nn.ModuleList([])
        in_planes = [1024, 512, 256, 128]
        for i in range(len(in_planes)):
            self.layers.append(upBlock(in_planes[i]))

    def forward(self, x, skip_connections):
        # todo
        # implement the forward path
        # x: feature tensor of semantic pathway
        for i in range(len(self.layers)):
            x_skip = skip_connections[f"down.{len(self.layers)-i-1}"]
            x = self.layers[i](x_skip, x)

        return x


class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # todo
        # initialize the complete model
        self.down = downStep()
        self.up = upStep()
        self.final_block = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        # todo
        # implement the forward path
        # Semantic pathway
        x = F.pad(x, [2, 2, 2, 2], "constant")
        x_semantic, skip_connections = self.down(x)
        x = self.up(x_semantic, skip_connections)
        out = F.pad(x, [2, 2, 2, 2], "constant")
        return self.final_block(out)


if __name__ == "__main__":
    model = UNet(n_classes=2).cuda()
    print(model)
    # x = torch.rand(4, 3, 256, 256).cuda()
    x = torch.rand(4, 1, 256, 256).cuda()
    out = model(x)
    print(out.shape)
