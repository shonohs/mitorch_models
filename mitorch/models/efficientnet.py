"""EfficientNet (https://arxiv.org/abs/1905.11946)"""
import collections
import math
import torch
from .model import Model
from .modules import Conv2dAct, MBConv


class EfficientNet(Model):
    BASIC_CONFIG = [
        # Expansion factor, kernel_size, in channels, out channels, stride, #layers
        [1, 3, 32, 16, 1, 1],
        [6, 3, 16, 24, 2, 2],  # 112x112 -> 56x56
        [6, 5, 24, 40, 2, 2],  # 56x56 -> 28x28
        [6, 3, 40, 80, 2, 3],  # 28x28 -> 14x14
        [6, 5, 80, 112, 1, 3],
        [6, 5, 112, 192, 2, 4],
        [6, 3, 192, 320, 1, 1]
    ]

    def __init__(self, width_multiplier=1, depth_multiplier=1, dropout_ratio=0.2):
        m = width_multiplier
        super().__init__(round(1280 * m / 8) * 8)

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier

        stages = [s for i in range(len(self.BASIC_CONFIG)) for s in self._make_stage(i)]

        self.features = torch.nn.Sequential(collections.OrderedDict([('conv0', Conv2dAct(3, round(32 * m / 8) * 8, kernel_size=3, padding=1, stride=2, activation='swish'))]
                                                                    + stages
                                                                    + [('conv1', Conv2dAct(round(320 * m / 8) * 8, round(1280 * m / 8) * 8, kernel_size=1, activation='swish')),
                                                                       ('pool0', torch.nn.AdaptiveAvgPool2d(1)),
                                                                       ('dropout0', torch.nn.Dropout(dropout_ratio)),
                                                                       ('flatten', torch.nn.Flatten())]))

    def _make_stage(self, index):
        expansion_factor, kernel_size, in_planes, out_planes, stride, num_layers = self.BASIC_CONFIG[index]
        in_planes = round(in_planes * self.width_multiplier / 8) * 8
        out_planes = round(out_planes * self.width_multiplier / 8) * 8
        expansion_planes = round(in_planes * expansion_factor / 8) * 8
        blocks = []
        for i in range(int(math.ceil(num_layers * self.depth_multiplier))):
            blocks.append((f'block{index}_{i}', MBConv(in_planes, out_planes, expansion_planes, kernel_size=kernel_size, stride=stride, use_se_swish=True, activation='swish')))
            stride = 1
            in_planes = out_planes

        return blocks


class EfficientNetB0(EfficientNet):
    pass


class EfficientNetB1(EfficientNet):
    def __init__(self):
        super().__init__(1.0, 1.1, 0.2)


class EfficientNetB2(EfficientNet):
    def __init__(self):
        super().__init__(1.1, 1.2, 0.3)


class EfficientNetB3(EfficientNet):
    def __init__(self):
        super().__init__(1.2, 1.4, 0.3)


class EfficientNetB4(EfficientNet):
    def __init__(self):
        super().__init__(1.4, 1.8, 0.4)


class EfficientNetB5(EfficientNet):
    def __init__(self):
        super().__init__(1.6, 2.2, 0.4)


class EfficientNetB6(EfficientNet):
    def __init__(self):
        super().__init__(1.8, 2.6, 0.5)


class EfficientNetB7(EfficientNet):
    def __init__(self):
        super().__init__(2.0, 3.1, 0.5)
