"""EfficientNet (https://arxiv.org/abs/1905.11946)"""
import collections
import torch
from .model import Model
from .modules import Add, Conv2dBN, Conv2dBNRelu, SEBlock


class EfficientNet(Model):
    BASIC_CONFIG = [
        # Expansion factor, kernel_size, in channels, out channels, stride, #layers
        [1, 3, 32, 16, 1, 1],
        [6, 3, 16, 24, 2, 2], # 112x112 -> 56x56
        [6, 5, 24, 40, 2, 2], # 56x56 -> 28x28
        [6, 3, 40, 80, 2, 3], # 28x28 -> 14x14
        [6, 5, 80, 112, 1, 3],
        [6, 5, 112, 192, 2, 4],
        [6, 3, 192, 320, 1, 1]
    ]

    class BasicBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, expansion_factor, stride=1, kernel_size=3, se_reduction_ratio=4):
            super(EfficientNet.BasicBlock, self).__init__()
            intermediate_channels = in_channels * expansion_factor
            self.conv0 = Conv2dBNRelu(in_channels, intermediate_channels, kernel_size=1, use_swish=True)
            self.conv1 = Conv2dBNRelu(intermediate_channels, intermediate_channels, kernel_size=kernel_size, padding=kernel_size//2,
                                      stride=stride, groups=intermediate_channels, use_swish=True)
            self.se = SEBlock(intermediate_channels, se_reduction_ratio, use_swish=True)
            self.conv2 = Conv2dBN(intermediate_channels, out_channels, kernel_size=1)
            self.residual = Add() if stride == 1 and in_channels == out_channels else None

        def forward(self, input):
            x = self.conv0(input)
            x = self.conv1(x)
            x = self.se(x)
            x = self.conv2(x)

            if self.residual:
                x = self.residual(x, input)

            return x

    def __init__(self, width_multiplier = 1, depth_multiplier = 1, dropout_ratio = 0.2):
        m = width_multiplier
        super(EfficientNet, self).__init__(int(1280 * width_multiplier))

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier

        stages = [s for i in range(len(self.BASIC_CONFIG)) for s in self._make_stage(i)]

        self.features = torch.nn.Sequential(collections.OrderedDict([('conv0', Conv2dBNRelu(3, int(32 * m), kernel_size=3, padding=1, stride=2, use_swish=True))]
                                                                   + stages
                                                                   + [('conv1', Conv2dBNRelu(int(320 * m), int(1280 * m), kernel_size=1, use_swish=True)),
                                                                      ('pool0', torch.nn.AdaptiveAvgPool2d(1)),
                                                                      ('dropout0', torch.nn.Dropout(dropout_ratio))]))

    def _make_stage(self, index):
        expansion_factor, kernel_size, in_planes, out_planes, stride, num_layers = self.BASIC_CONFIG[index]
        in_planes = int(in_planes * self.width_multiplier)
        out_planes = int(out_planes * self.width_multiplier)
        blocks = []
        for i in range(int(num_layers * self.depth_multiplier)):
            blocks.append((f'block{index}_{i}', EfficientNet.BasicBlock(in_planes, out_planes, expansion_factor, stride, kernel_size)))
            stride = 1
            in_planes = out_planes

        return blocks


class EfficientNetB0(EfficientNet):
    INPUT_SIZE = 224


class EfficientNetB1(EfficientNet):
    INPUT_SIZE = 240
    def __init__(self):
        super(EfficientNetB1, self).__init__(1.0, 1.1, 0.2)


class EfficientNetB2(EfficientNet):
    INPUT_SIZE = 260
    def __init__(self):
        super(EfficientNetB2, self).__init__(1.1, 1.2, 0.3)


class EfficientNetB3(EfficientNet):
    INPUT_SIZE = 300
    def __init__(self):
        super(EfficientNetB3, self).__init__(1.2, 1.4, 0.3)


class EfficientNetB4(EfficientNet):
    INPUT_SIZE = 380
    def __init__(self):
        super(EfficientNetB4, self).__init__(1.4, 1.8, 0.4)


class EfficientNetB5(EfficientNet):
    INPUT_SIZE = 456
    def __init__(self):
        super(EfficientNetB5, self).__init__(1.6, 2.2, 0.4)


class EfficientNetB6(EfficientNet):
    INPUT_SIZE = 528
    def __init__(self):
        super(EfficientNetB6, self).__init__(1.8, 2.6, 0.5)


class EfficientNetB7(EfficientNet):
    INPUT_SIZE = 600
    def __init__(self):
        super(EfficientNetB7, self).__init__(2.0, 3.1, 0.5)
