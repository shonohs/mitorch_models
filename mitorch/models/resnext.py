"""ResNext (https://arxiv.org/abs/1611.05431)"""
import collections
import torch
from .model import Model
from .modules import Add, Conv2dAct, Conv2dBN, default_module_settings


class ResNext(Model):
    class BasicBlock(torch.nn.Module):
        """Depth=3 building block"""
        def __init__(self, in_channels, out_channels, cardinality, stride=1):
            super().__init__()
            self.conv0 = Conv2dAct(in_channels, out_channels // 2, kernel_size=1)
            self.conv1 = Conv2dAct(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, stride=stride, groups=cardinality)
            self.conv2 = Conv2dBN(out_channels // 2, out_channels, kernel_size=1)

            self.conv_shortcut = Conv2dBN(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels or stride != 1 else None
            self.add = Add()
            self.activation = torch.nn.ReLU()

        def forward(self, input):
            x = self.conv0(input)
            x = self.conv1(x)
            x = self.conv2(x)

            x = self.add(x, self.conv_shortcut(input) if self.conv_shortcut else input)
            return self.activation(x)

    @default_module_settings(use_bn=True)
    def __init__(self, num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4):
        in_channels = 64
        out_channels = bottleneck_width * cardinality * 2
        feature_channels = out_channels * (2 ** (len(num_blocks) - 1))
        super().__init__(feature_channels)

        first_block_stride = 1
        blocks = []
        for i, n in enumerate(num_blocks):
            blocks.extend(self._make_stage(in_channels, out_channels, n, first_block_stride, cardinality, i))
            in_channels = out_channels
            out_channels *= 2
            first_block_stride = 2

        self.features = torch.nn.Sequential(collections.OrderedDict([('conv0', Conv2dAct(3, 64, kernel_size=7, stride=2, padding=3)),
                                                                     ('pool0', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]
                                                                    + blocks
                                                                    + [('pool1', torch.nn.AdaptiveAvgPool2d(1)),
                                                                       ('flatten', torch.nn.Flatten())]))

    def _make_stage(self, in_channels, out_channels, num_blocks, first_block_stride, cardinality, index):
        blocks = [(f'block{index}_0', ResNext.BasicBlock(in_channels, out_channels, cardinality, first_block_stride))]
        for i in range(num_blocks - 1):
            blocks.append((f'block{index}_{i+1}', ResNext.BasicBlock(out_channels, out_channels, cardinality)))
        return blocks


class ResNext14(ResNext):
    def __init__(self):
        super().__init__([1, 1, 1, 1])


class ResNext26(ResNext):
    def __init__(self):
        super().__init__([2, 2, 2, 2])


class ResNext50(ResNext):
    def __init__(self):
        super().__init__([3, 4, 6, 3])


class ResNext101(ResNext):
    def __init__(self):
        super().__init__([3, 4, 23, 3])
