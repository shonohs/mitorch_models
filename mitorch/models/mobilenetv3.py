"""MobileNetV3 model (https://arxiv.org/abs/1905.02244)"""
import collections
import torch
from .model import Model
from .modules import Conv2dAct, MBConv, default_module_settings


class MobileNetV3(Model):
    @default_module_settings(activation='relu6')
    def __init__(self, width_multiplier=1, dropout_ratio=0.2, **kwargs):
        m = width_multiplier
        super().__init__(int(1280 * m), **kwargs)

        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dAct(3, int(16 * m), kernel_size=3, padding=1, stride=2, activation='hswish')),
            ('block0_0', MBConv(int(16 * m), int(16 * m), int(16 * m), use_se=False, activation='relu6')),
            ('block1_0', MBConv(int(16 * m), int(24 * m), int(64 * m), use_se=False, activation='relu6', stride=2)),
            ('block1_1', MBConv(int(24 * m), int(24 * m), int(72 * m), use_se=False, activation='relu6')),
            ('block2_0', MBConv(int(24 * m), int(40 * m), int(72 * m), activation='relu6', stride=2, kernel_size=5)),
            ('block2_1', MBConv(int(40 * m), int(40 * m), int(120 * m), activation='relu6', kernel_size=5)),
            ('block2_2', MBConv(int(40 * m), int(40 * m), int(120 * m), activation='relu6', kernel_size=5)),
            ('block3_0', MBConv(int(40 * m), int(80 * m), int(240 * m), use_se=False, stride=2)),
            ('block3_1', MBConv(int(80 * m), int(80 * m), int(200 * m), use_se=False)),
            ('block3_2', MBConv(int(80 * m), int(80 * m), int(184 * m), use_se=False)),
            ('block3_3', MBConv(int(80 * m), int(80 * m), int(184 * m), use_se=False)),
            ('block3_4', MBConv(int(80 * m), int(112 * m), int(480 * m))),
            ('block3_5', MBConv(int(112 * m), int(112 * m), int(672 * m))),
            ('block4_0', MBConv(int(112 * m), int(160 * m), int(672 * m), stride=2, kernel_size=5)),
            ('block4_1', MBConv(int(160 * m), int(160 * m), int(960 * m), kernel_size=5)),
            ('block4_2', MBConv(int(160 * m), int(160 * m), int(960 * m), kernel_size=5)),
            ('conv1', Conv2dAct(int(160 * m), int(960 * m), kernel_size=1, activation='hswish')),
            ('pool0', torch.nn.AdaptiveAvgPool2d(1)),
            ('conv2', Conv2dAct(int(960 * m), int(1280 * m), kernel_size=1, use_bn=False, activation='hswish')),
            ('dropout', torch.nn.Dropout(p=dropout_ratio)),
            ('flatten', torch.nn.Flatten())
        ]))


class MobileNetV3Small(Model):
    @default_module_settings(activation='relu6')
    def __init__(self, width_multiplier=1, dropout_ratio=0.2, **kwargs):
        m = width_multiplier
        super().__init__(int(1024 * m), **kwargs)

        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dAct(3, int(16 * m), kernel_size=3, padding=1, stride=2, activation='hswish')),
            ('block0_0', MBConv(int(16 * m), int(16 * m), int(16 * m), activation='relu6', stride=2)),
            ('block1_0', MBConv(int(16 * m), int(24 * m), int(72 * m), use_se=False, activation='relu6', stride=2)),
            ('block1_1', MBConv(int(24 * m), int(24 * m), int(88 * m), use_se=False, activation='relu6')),
            ('block2_0', MBConv(int(24 * m), int(40 * m), int(96 * m), stride=2, kernel_size=5)),
            ('block2_1', MBConv(int(40 * m), int(40 * m), int(240 * m), kernel_size=5)),
            ('block2_2', MBConv(int(40 * m), int(40 * m), int(240 * m), kernel_size=5)),
            ('block2_3', MBConv(int(40 * m), int(48 * m), int(120 * m), kernel_size=5)),
            ('block2_4', MBConv(int(48 * m), int(48 * m), int(144 * m), kernel_size=5)),
            ('block3_0', MBConv(int(48 * m), int(96 * m), int(288 * m), kernel_size=5, stride=2)),
            ('block3_1', MBConv(int(96 * m), int(96 * m), int(576 * m), kernel_size=5)),
            ('block3_2', MBConv(int(96 * m), int(96 * m), int(576 * m), kernel_size=5)),
            ('conv1', Conv2dAct(int(96 * m), int(576 * m), kernel_size=1, activation='hswish')),
            ('pool0', torch.nn.AdaptiveAvgPool2d(1)),
            ('conv2', Conv2dAct(int(576 * m), int(1024 * m), kernel_size=1, use_bn=False, activation='hswish')),
            ('dropout', torch.nn.Dropout(p=dropout_ratio)),
            ('flatten', torch.nn.Flatten())
        ]))
