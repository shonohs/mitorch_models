"""MobileNetV2 model (https://arxiv.org/pdf/1801.04381.pdf)"""
import collections
import torch
from .model import Model
from .modules import Add, Conv2dAct, DepthwiseSeparableConv2d, default_module_settings


class MobileNetV2(Model):
    class BasicBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, expansion_factor, stride=1):
            super().__init__()
            intermediate_channels = in_channels * expansion_factor
            self.conv0 = Conv2dAct(in_channels, intermediate_channels, kernel_size=1)
            self.conv1 = DepthwiseSeparableConv2d(intermediate_channels, out_channels, kernel_size=3, stride=stride, padding=1, activation2='none')
            self.residual = Add() if stride == 1 and in_channels == out_channels else None

        def forward(self, input):
            x = self.conv0(input)
            x = self.conv1(x)
            if self.residual:
                x = self.residual(x, input)
            return x

    @default_module_settings(activation='relu6')
    def __init__(self, width_multiplier=1):
        super().__init__(output_dim=1280)

        m = width_multiplier
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dAct(3, int(32 * m), kernel_size=3, padding=1, stride=2)),
            ('block0_0', MobileNetV2.BasicBlock(int(32 * m), int(16 * m), expansion_factor=1)),
            ('block1_0', MobileNetV2.BasicBlock(int(16 * m), int(24 * m), expansion_factor=6, stride=2)),
            ('block1_1', MobileNetV2.BasicBlock(int(24 * m), int(24 * m), expansion_factor=6)),
            ('block2_0', MobileNetV2.BasicBlock(int(24 * m), int(32 * m), expansion_factor=6, stride=2)),
            ('block2_1', MobileNetV2.BasicBlock(int(32 * m), int(32 * m), expansion_factor=6)),
            ('block2_2', MobileNetV2.BasicBlock(int(32 * m), int(32 * m), expansion_factor=6)),
            ('block3_0', MobileNetV2.BasicBlock(int(32 * m), int(64 * m), expansion_factor=6, stride=2)),
            ('block3_1', MobileNetV2.BasicBlock(int(64 * m), int(64 * m), expansion_factor=6)),
            ('block3_2', MobileNetV2.BasicBlock(int(64 * m), int(64 * m), expansion_factor=6)),
            ('block3_3', MobileNetV2.BasicBlock(int(64 * m), int(64 * m), expansion_factor=6)),
            ('block4_0', MobileNetV2.BasicBlock(int(64 * m), int(96 * m), expansion_factor=6)),
            ('block4_1', MobileNetV2.BasicBlock(int(96 * m), int(96 * m), expansion_factor=6)),
            ('block4_2', MobileNetV2.BasicBlock(int(96 * m), int(96 * m), expansion_factor=6)),
            ('block5_0', MobileNetV2.BasicBlock(int(96 * m), int(160 * m), expansion_factor=6, stride=2)),
            ('block5_1', MobileNetV2.BasicBlock(int(160 * m), int(160 * m), expansion_factor=6)),
            ('block5_2', MobileNetV2.BasicBlock(int(160 * m), int(160 * m), expansion_factor=6)),
            ('block6_0', MobileNetV2.BasicBlock(int(160 * m), int(320 * m), expansion_factor=6)),
            ('conv1', Conv2dAct(int(320 * m), int(1280 * m), kernel_size=1)),
            ('pool0', torch.nn.AdaptiveAvgPool2d(1)),
            ('flatten', torch.nn.Flatten())
        ]))
