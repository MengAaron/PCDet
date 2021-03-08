# RSN feature extractor
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Down(nn.Module):
    """Downsample module for U-Net"""

    def __init__(self, layer_num, in_channels, out_channels):
        assert layer_num > 0
        assert type(layer_num) == int
        super().__init__()

        layers = [BasicBlock(in_channels=in_channels, out_channels=out_channels, stride=2)]
        for i in range(layer_num - 1):
            layers.append(BasicBlock(in_channels=in_channels, out_channels=out_channels, stride=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):
    """Upsample module for U-Net"""

    def __init__(self, layer_num, in_channels, out_channels):
        assert layer_num > 0
        assert type(layer_num) == int
        super().__init__()

        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)]
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        for i in range(layer_num):
            layers.append(BasicBlock(in_channels=out_channels, out_channels=out_channels, stride=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.Down1 = Down(1, in_channels=in_channels, out_channels=16)
        self.Down2 = Down(2, in_channels=16, out_channels=64)
        self.Down3 = Down(2, in_channels=64, out_channels=128)
        self.Down4 = Down(2, in_channels=128, out_channels=128)
        self.Up3 = Up(2, in_channels=128, out_channels=128)
        self.Up2 = Up(2, in_channels=128, out_channels=64)
        self.Up1 = Up(1, in_channels=64, out_channels=16)
        self.final = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        conv1 = self.Down1(x)
        conv2 = self.Down2(conv1)
        conv3 = self.Down3(conv2)
        conv4 = self.Down4(conv3)
        up3 = torch.cat((self.Up3(conv4), conv3), dim=1)
        up2 = torch.cat((self.Up2(up3), conv2), dim=1)
        up1 = torch.cat((self.Up1(up2), conv1), dim=1)
        output = self.final(up1)

        return output