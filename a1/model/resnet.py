import torch.utils.model_zoo as model_zoo

from modules.resnet.block import *


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes):
        super().__init__()

        # Conv_1 x 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_plane = 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # conv_2
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # conv_3
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # conv_4
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # conv_5

        self.cls_head = self._make_cls_head(512 * block.expansion, num_classes)

        def _weights_init(m):
            """kaiming init (https://arxiv.org/abs/1502.01852v1)"""
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)

    def _make_layer(self, block, hidden_dim, num_blocks, stride=1):
        downsample = None
        if stride != 1 or hidden_dim * block.expansion != self.in_plane:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_plane,
                    hidden_dim * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim * block.expansion),
            )
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(self.in_plane, hidden_dim, stride, downsample))
                self.in_plane = hidden_dim * block.expansion
            else:
                layers.append(block(self.in_plane, hidden_dim))

        return nn.Sequential(*layers)

    def _make_cls_head(self, input_dim, num_classes):
        """cls: number of classes"""
        layers = [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_dim, num_classes),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.cls_head(x)

        return x


"""
    address to load pretrained model
"""
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
}


def resnet18(num_classes, pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    resnet_state_dict = model_zoo.load_url(model_urls["resnet18"])

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    if pretrained:
        print("Loading pretrained ...")
        model_state_dict = model.state_dict()
        for k in resnet_state_dict.keys():
            if k in model_state_dict.keys() and not k.startswith("fc"):
                model_state_dict[k] = resnet_state_dict[k]
        model.load_state_dict(model_state_dict)

    return model


def resnet34(num_classes, pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    resnet_state_dict = model_zoo.load_url(model_urls["resnet34"])

    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    if pretrained:
        print("Loading pretrained ...")
        model_state_dict = model.state_dict()
        for k in resnet_state_dict.keys():
            if k in model_state_dict.keys() and not k.startswith("fc"):
                model_state_dict[k] = resnet_state_dict[k]
        model.load_state_dict(model_state_dict)

    return model


def resnet50(num_classes, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    resnet_state_dict = model_zoo.load_url(model_urls["resnet50"])

    model = ResNet(BottleNeckBlock, [3, 4, 6, 3], num_classes)
    if pretrained:
        print("Loading pretrained ...")
        model_state_dict = model.state_dict()
        for k in resnet_state_dict.keys():
            if k in model_state_dict.keys() and not k.startswith("fc"):
                model_state_dict[k] = resnet_state_dict[k]
        model.load_state_dict(model_state_dict)

    return model


def resnet101(num_classes, pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    resnet_state_dict = model_zoo.load_url(model_urls["resnet101"])

    model = ResNet(BottleNeckBlock, [3, 4, 23, 3], num_classes)
    if pretrained:
        print("Loading pretrained ...")
        model_state_dict = model.state_dict()
        for k in resnet_state_dict.keys():
            if k in model_state_dict.keys() and not k.startswith("fc"):
                model_state_dict[k] = resnet_state_dict[k]
        model.load_state_dict(model_state_dict)

    return model
