import collections
import torch
from .model import Model
from .modules import Conv2dAct


class SqueezeNet(Model):
    class BasicBlock(torch.nn.Module):
        def __init__(self, in_channels, squeeze_channels, expand_channels):
            super().__init__()
            self.conv0 = Conv2dAct(in_channels, squeeze_channels, kernel_size=1)
            self.conv1 = Conv2dAct(squeeze_channels, expand_channels // 2, kernel_size=1)
            self.conv2 = Conv2dAct(squeeze_channels, expand_channels // 2, kernel_size=3, padding=1)

        def forward(self, input):
            x = self.conv0(input)
            return torch.cat([self.conv1(x), self.conv2(x)], 1)

    def __init__(self, squeeze_ratio=0.125, dropout_ratio=0.5):
        super().__init__(512)

        r = squeeze_ratio
        basic_block = SqueezeNet.BasicBlock

        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dAct(3, 64, kernel_size=3, stride=2, padding=1)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('block0_0', basic_block(64, int(128 * r), 128)),
            ('block0_1', basic_block(128, int(128 * r), 128)),
            ('pool1', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('block1_0', basic_block(128, int(256 * r), 256)),
            ('block1_1', basic_block(256, int(256 * r), 256)),
            ('pool2', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('block2_0', basic_block(256, int(384 * r), 384)),
            ('block2_1', basic_block(384, int(384 * r), 384)),
            ('block2_2', basic_block(384, int(512 * r), 512)),
            ('block2_3', basic_block(512, int(512 * r), 512)),
            ('dropout', torch.nn.Dropout(p=dropout_ratio)),
            ('pool3', torch.nn.AdaptiveAvgPool2d(1)),
            ('flatten', torch.nn.Flatten())
        ]))
