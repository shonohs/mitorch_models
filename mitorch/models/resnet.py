"""Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)"""
import collections
import torch
from .model import Model
from .modules import Activation, Add, Conv2dAct, Conv2dBN


class ResNet(Model):
    class BasicBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv0 = Conv2dAct(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
            self.conv1 = Conv2dBN(out_channels, out_channels, kernel_size=3, padding=1)
            self.conv_shortcut = Conv2dBN(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels or stride != 1 else None
            self.add = Add()
            self.activation = Activation()

        def forward(self, inputs):
            x = self.conv0(inputs)
            x = self.conv1(x)
            x = self.add(x, self.conv_shortcut(inputs) if self.conv_shortcut else inputs)
            return self.activation(x)

    class BottleneckBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv0 = Conv2dAct(in_channels, out_channels // 4, kernel_size=1)
            self.conv1 = Conv2dAct(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, stride=stride)
            self.conv2 = Conv2dBN(out_channels // 4, out_channels, kernel_size=1)

            self.conv_shortcut = Conv2dBN(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels or stride != 1 else None
            self.add = Add()
            self.activation = Activation()

        def forward(self, input):
            x = self.conv0(input)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.add(x, self.conv_shortcut(input) if self.conv_shortcut else input)
            return self.activation(x)

    def __init__(self, num_blocks=[3, 4, 6, 3], use_bottleneck=True):
        in_channels = 64
        out_channels = 256 if use_bottleneck else 64
        feature_channels = out_channels * (2 ** (len(num_blocks) - 1))
        super().__init__(feature_channels)

        basic_block = ResNet.BottleneckBlock if use_bottleneck else ResNet.BasicBlock
        first_block_stride = 1
        blocks = []
        for i, n in enumerate(num_blocks):
            blocks.extend(self._make_stage(basic_block, in_channels, out_channels, n, first_block_stride, i))
            in_channels = out_channels
            out_channels *= 2
            first_block_stride = 2

        self.features = torch.nn.Sequential(collections.OrderedDict([('conv0', Conv2dAct(3, 64, kernel_size=7, stride=2, padding=3)),
                                                                     ('pool0', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]
                                                                    + blocks
                                                                    + [('pool1', torch.nn.AdaptiveAvgPool2d(1)),
                                                                       ('flatten', torch.nn.Flatten())]))

    def _make_stage(self, basic_block, in_channels, out_channels, num_blocks, first_block_stride, index):
        blocks = [(f'block{index}_0', basic_block(in_channels, out_channels, first_block_stride))]
        for i in range(num_blocks - 1):
            blocks.append((f'block{index}_{i+1}', basic_block(out_channels, out_channels)))
        return blocks


class ResNet18(ResNet):
    def __init__(self):
        super().__init__([2, 2, 2, 2], use_bottleneck=False)


class ResNet34(ResNet):
    def __init__(self):
        super().__init__([3, 4, 6, 3], use_bottleneck=False)


class ResNet50(ResNet):
    def __init__(self):
        super().__init__([3, 4, 6, 3])


class ResNet101(ResNet):
    def __init__(self):
        super().__init__([3, 4, 23, 3])


class ResNet152(ResNet):
    def __init__(self):
        super().__init__([3, 8, 36, 3])
