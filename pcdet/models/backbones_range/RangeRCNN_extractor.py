# Range RCNN feature extractor
import pudb
import torch
import torch.nn as nn
import torch.nn.functional as F


class DRBlock(nn.Module):
    # Dilated Residual Block
    def __init__(self, in_channels, out_channels):
        super(DRBlock, self).__init__()
        self.out_channels = out_channels
        # residual function
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, bias=False, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv_fuse = torch.cat([conv0, conv1, conv2], dim=1)
        output = self.conv_fuse(conv_fuse)

        return nn.ReLU(inplace=True)(output + self.shortcut(x))


class DDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DDBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.DRBlock = DRBlock(out_channels, out_channels)
        self.post = nn.Sequential(
            nn.Dropout(), nn.MaxPool2d(stride))

    def forward(self, x):
        x = self.conv(x)
        x = self.DRBlock(x)
        x = self.post(x)
        return x


class RangeRCNNBackbone(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(RangeRCNNBackbone, self).__init__()
        self.Down0 = DDBlock(in_channels, out_channels=32, stride=1)
        self.Down1 = DDBlock(32, 64, stride=1)
        self.Down2 = DDBlock(64, 128)
        self.Down3 = DDBlock(128, 128)
        self.Down4 = DDBlock(128, 256)
        self.Down5 = DDBlock(256, 256)
        self.Up = nn.UpsamplingBilinear2d()
        self.Up4 = DRBlock(512, 128)
        self.Up3 = DRBlock(256, 128)
        self.Up2 = DRBlock(256, 64)
        self.Up1 = DRBlock(128, 64)
        self.out_channels = self.Up1.out_channels

    def forward(self, batch_dict):
        pudb.set_trace()
        x = batch_dict['range_image']
        conv0 = self.Down0(x)
        conv1 = self.Down1(conv0)
        conv2 = self.Down2(conv1)
        conv3 = self.Down3(conv2)
        conv4 = self.Down4(conv3)
        conv5 = self.Down5(conv4)
        up4 = torch.cat([self.Up(conv5), conv4], dim=1)
        up4 = self.Up4(up4)
        up3 = torch.cat([self.Up(up4), conv3], dim=1)
        up3 = self.Up4(up3)
        up2 = torch.cat([self.Up(up3), conv2], dim=1)
        up2 = self.Up4(up2)
        up1 = torch.cat([self.Up(up2), conv1], dim=1)
        up1 = self.Up4(up1)
        batch_dict.pop('range_image', None)
        batch_dict['range_features'] = up1
        return batch_dict

    def get_output_feature_dim(self):
        return self.out_channels
