"""ConvMixer
Patches Are All You Need? (https://openreview.net/forum?id=TVHS5Y4dNvM)
"""
import collections
import torch
from .model import Model
from .modules import Add, Conv2dAct


class ConvMixer(Model):
    class BasicBlock(torch.nn.Module):
        def __init__(self, channels, kernel_size, activation):
            super().__init__()
            self.conv0 = Conv2dAct(channels, channels, kernel_size=kernel_size, groups=channels, padding=kernel_size//2, activation=activation)
            self.conv1 = Conv2dAct(channels, channels, kernel_size=1, activation=activation)
            self.add = Add()

        def forward(self, input_tensor):
            x = self.add(self.conv0(input_tensor) + input_tensor)
            return self.conv1(x)

    def __init__(self, embedding_channels, depth, patch_size, kernel_size, activation='relu'):
        super().__init__(embedding_channels)

        # TODO: Check the impact of Conv->BN->Act
        self.features = torch.nn.Sequential(collections.OrderedDict([('conv0', Conv2dAct(3, embedding_channels, kernel_size=patch_size, stride=patch_size, activation=activation))]
                                                                    + [(f'block{i}', self.BasicBlock(embedding_channels, kernel_size, activation)) for i in range(depth)]
                                                                    + [('pool0', torch.nn.AdaptiveAvgPool2d(1)),
                                                                       ('flatten', torch.nn.Flatten())]))


class ConvMixer1536_20(ConvMixer):
    def __init__(self):
        super().__init__(1536, 20, 7, 9, 'gelu')


class ConvMixer768_32(ConvMixer):
    def __init__(self):
        super().__init__(768, 32, 7, 7)
