"""MobileNetV3 model (https://arxiv.org/abs/1905.02244)"""
import collections
import torch
from .model import Model
from .modules import Add, Conv2dBNRelu, Conv2dBN, Conv2dRelu, HardSwish, SEBlock

class MobileNetV3Base(Model):
    class BasicBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, expansion_channels, stride=1, use_se=True, use_hswish=True, kernel_size=3):
            super(MobileNetV3Base.BasicBlock, self).__init__()
            self.conv0 = Conv2dBNRelu(in_channels, expansion_channels, kernel_size=1, use_hswish=use_hswish)
            self.conv1 = Conv2dBN(expansion_channels, expansion_channels,
                                  kernel_size=kernel_size, padding=kernel_size//2,
                                  stride=stride, groups=expansion_channels)
            self.activation1 = HardSwish() if use_hswish else torch.nn.ReLU()
            self.conv2 = Conv2dBN(expansion_channels, out_channels, kernel_size=1)

            self.se = SEBlock(expansion_channels, expansion_channels // 4, use_hsigmoid=True) if use_se else None
            self.residual = Add() if stride == 1 and in_channels == out_channels else None

        def forward(self, input):
            x = self.conv0(input)
            x = self.conv1(x)

            if self.se:
                x = self.se(x)

            x = self.activation1(x)
            x = self.conv2(x)

            if self.residual:
                x = self.residual(x, input)

            return x

class MobileNetV3(MobileNetV3Base):
    def __init__(self, width_multiplier = 1, use_hswish = True):
        m = width_multiplier
        super(MobileNetV3, self).__init__(int(1280 * m), use_hswish=use_hswish)

        basic_block = self.BasicBlock


        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dBNRelu(3, int(16 * m), kernel_size=3, padding=1, stride=2, use_hswish=True)),
            ('block0_0', basic_block(int(16 * m), int(16 * m), int(16 * m), use_se=False, use_hswish=False)),
            ('block1_0', basic_block(int(16 * m), int(24 * m), int(64 * m), use_se=False, use_hswish=False, stride=2)),
            ('block1_1', basic_block(int(24 * m), int(24 * m), int(72 * m), use_se=False, use_hswish=False)),
            ('block2_0', basic_block(int(24 * m), int(40 * m), int(72 * m), use_hswish=False, stride=2, kernel_size=5)),
            ('block2_1', basic_block(int(40 * m), int(40 * m), int(120 * m), use_hswish=False, kernel_size=5)),
            ('block2_2', basic_block(int(40 * m), int(40 * m), int(120 * m), use_hswish=False, kernel_size=5)),
            ('block3_0', basic_block(int(40 * m), int(80 * m), int(240 * m), use_se=False, stride=2)),
            ('block3_1', basic_block(int(80 * m), int(80 * m), int(200 * m), use_se=False)),
            ('block3_2', basic_block(int(80 * m), int(80 * m), int(184 * m), use_se=False)),
            ('block3_3', basic_block(int(80 * m), int(80 * m), int(184 * m), use_se=False)),
            ('block3_4', basic_block(int(80 * m), int(112 * m), int(480 * m))),
            ('block3_5', basic_block(int(112 * m), int(112 * m), int(672 * m))),
            ('block4_0', basic_block(int(112 * m), int(160 * m), int(672 * m), stride=2, kernel_size=5)),
            ('block4_1', basic_block(int(160 * m), int(160 * m), int(960 * m), kernel_size=5)),
            ('block4_2', basic_block(int(160 * m), int(160 * m), int(960 * m), kernel_size=5)),
            ('conv1', Conv2dBNRelu(int(160 * m), int(960 * m), kernel_size=1, use_hswish=True)),
            ('pool0', torch.nn.AdaptiveAvgPool2d(1)),
            ('conv2', Conv2dRelu(int(960 * m), int(1280 * m), kernel_size=1, use_hswish=True)),
            ('flatten', torch.nn.Flatten())
        ]))

    def forward(self, input):
        return self.features(input)

class MobileNetV3Small(MobileNetV3Base):
    def __init__(self, width_multiplier = 1, use_hswish = True):
        m = width_multiplier
        super(MobileNetV3Small, self).__init__(int(1024 * m), use_hswish=use_hswish)

        basic_block = self.BasicBlock

        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dBNRelu(3, int(16 * m), kernel_size=3, padding=1, stride=2, use_hswish=True)),
            ('block0_0', basic_block(int(16 * m), int(16 * m), int(16 * m), use_se=False, use_hswish=False)),
            ('block1_0', basic_block(int(16 * m), int(24 * m), int(72 * m), use_se=False, use_hswish=False, stride=2)),
            ('block2_0', basic_block(int(24 * m), int(24 * m), int(88 * m), use_se=False, use_hswish=False)),
            ('block2_0', basic_block(int(24 * m), int(40 * m), int(96 * m), use_hswish=False, stride=2, kernel_size=5)),
            ('block2_1', basic_block(int(40 * m), int(40 * m), int(240 * m), use_hswish=False, kernel_size=5)),
            ('block2_2', basic_block(int(40 * m), int(40 * m), int(240 * m), use_hswish=False, kernel_size=5)),
            ('block3_0', basic_block(int(40 * m), int(48 * m), int(120 * m), use_se=False, stride=2)),
            ('block3_1', basic_block(int(48 * m), int(48 * m), int(144 * m), use_se=False)),
            ('block3_2', basic_block(int(48 * m), int(96 * m), int(288 * m), use_se=False)),
            ('block3_3', basic_block(int(96 * m), int(96 * m), int(576 * m), use_se=False)),
            ('block3_4', basic_block(int(96 * m), int(96 * m), int(576 * m))),
            ('conv1', Conv2dBNRelu(int(96 * m), int(576 * m), kernel_size=1, use_hswish=True)),
            ('pool0', torch.nn.AdaptiveAvgPool2d(1)),
            ('conv2', Conv2dRelu(int(576 * m), int(1024 * m), kernel_size=1, use_hswish=True)),
            ('flatten', torch.nn.Flatten())
        ]))

    def forward(self, input):
        return self.features(input)
