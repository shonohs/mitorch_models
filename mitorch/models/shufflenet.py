"""ShuffleNet (https://arxiv.org/abs/1707.01083)"""
import collections
import torch
from .model import Model
from .modules import Add, Conv2dAct, ChannelShuffle, Conv2dBN


class ShuffleNet(Model):
    FIRST_STAGE_CHANNELS = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}

    class BasicBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, num_groups):
            super().__init__()
            assert in_channels == out_channels

            bottleneck_channels = out_channels // 4

            self.conv0 = Conv2dAct(in_channels, bottleneck_channels, kernel_size=1, groups=num_groups)
            self.shuffle = ChannelShuffle(num_groups)
            self.conv1 = Conv2dBN(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, groups=bottleneck_channels)
            self.conv2 = Conv2dBN(bottleneck_channels, out_channels, kernel_size=1, groups=num_groups)

            self.add = Add()
            self.activation = torch.nn.ReLU()

        def forward(self, input):
            x = self.conv0(input)
            x = self.shuffle(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.add(x, input)
            return self.activation(x)

    class DownsampleBasicBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, num_groups, stride, skip_first_group_conv=False):
            super().__init__()
            assert stride > 1

            out_channels = out_channels - in_channels
            bottleneck_channels = out_channels // 4
            first_conv_num_groups = 1 if skip_first_group_conv else num_groups

            self.conv0 = Conv2dAct(in_channels, bottleneck_channels, kernel_size=1, groups=first_conv_num_groups)
            self.shuffle = ChannelShuffle(num_groups)
            self.conv1 = Conv2dBN(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, groups=bottleneck_channels, stride=stride)
            self.conv2 = Conv2dBN(bottleneck_channels, out_channels, kernel_size=1, groups=num_groups)
            self.pool = torch.nn.AvgPool2d(kernel_size=3, padding=1, stride=stride)
            self.activation = torch.nn.ReLU()

        def forward(self, input):
            x = self.conv0(input)
            x = self.shuffle(x)
            x = self.conv1(x)
            x = self.conv2(x)
            shortcut = self.pool(input)
            x = torch.cat((shortcut, x), 1)
            return self.activation(x)

    def __init__(self, width_multiplier=1, num_groups=3, num_blocks=[4, 8, 4]):
        assert num_groups in ShuffleNet.FIRST_STAGE_CHANNELS, "Unexpected number of groups"
        first_stage_channels = int(ShuffleNet.FIRST_STAGE_CHANNELS[num_groups] * width_multiplier)
        feature_planes = first_stage_channels * (2 ** (len(num_blocks) - 1))
        super().__init__(feature_planes)

        in_planes = 24
        out_planes = first_stage_channels
        blocks = []
        for i, n in enumerate(num_blocks):
            blocks.extend(self._make_stage(in_planes, out_planes, n, num_groups, i, skip_first_group_conv=(i == 0)))
            in_planes = out_planes
            out_planes *= 2

        self.features = torch.nn.Sequential(collections.OrderedDict([('conv0', Conv2dAct(3, 24, kernel_size=3, stride=2, padding=1)),
                                                                     ('pool0', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]
                                                                    + blocks
                                                                    + [('pool1', torch.nn.AdaptiveAvgPool2d(1)),
                                                                       ('flatten', torch.nn.Flatten())]))

    def _make_stage(self, in_channels, out_channels, num_blocks, num_groups, index, skip_first_group_conv=False):
        blocks = [(f'block{index}_0', ShuffleNet.DownsampleBasicBlock(in_channels, out_channels, num_groups, stride=2, skip_first_group_conv=skip_first_group_conv))]
        for i in range(num_blocks - 1):
            blocks.append((f'block{index}_{i+1}', ShuffleNet.BasicBlock(out_channels, out_channels, num_groups)))
        return blocks
