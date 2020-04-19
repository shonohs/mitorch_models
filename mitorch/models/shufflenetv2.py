"""ShuffleNetV2"""
import collections
import torch
from .model import Model
from .modules import Conv2dAct, Conv2dBN, SEBlock, ChannelShuffle, default_module_settings


class ShuffleNetV2(Model):
    FIRST_STAGE_CHANNELS = {0.5: 48, 1.0: 116, 1.5: 176, 2.0: 244}

    class BasicBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, use_se=False):
            super().__init__()
            in_channels = in_channels // 2
            out_channels = out_channels // 2

            self.conv0 = Conv2dAct(in_channels, out_channels, kernel_size=1)
            self.conv1 = Conv2dBN(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
            self.conv2 = Conv2dAct(out_channels, out_channels, kernel_size=1)

            self.se = SEBlock if use_se else None
            self.shuffle = ChannelShuffle(2)

        def forward(self, input):
            x1, x2 = torch.chunk(input, 2, dim=1)
            x2 = self.conv0(x2)
            x2 = self.conv1(x2)
            x2 = self.conv2(x2)
            if self.se:
                x2 = self.se(x2)

            x = torch.cat((x1, x2), 1)
            return self.shuffle(x)

    class DownsampleBasicBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, use_se=False):
            super().__init__()
            out_channels = out_channels // 2

            self.conv0 = Conv2dAct(in_channels, out_channels, kernel_size=1)
            self.conv1 = Conv2dBN(out_channels, out_channels, kernel_size=3, padding=1, stride=2, groups=out_channels)
            self.conv2 = Conv2dAct(out_channels, out_channels, kernel_size=1)
            self.se = SEBlock(out_channels, 4) if use_se else None
            self.conv3 = Conv2dBN(in_channels, in_channels, kernel_size=3, padding=1, stride=2, groups=in_channels)
            self.conv4 = Conv2dAct(in_channels, out_channels, kernel_size=1)
            self.shuffle = ChannelShuffle(2)

        def forward(self, input):
            x1 = self.conv0(input)
            x1 = self.conv1(x1)
            x1 = self.conv2(x1)

            if self.se:
                x1 = self.se(x1)

            x2 = self.conv3(input)
            x2 = self.conv4(x2)

            x = torch.cat((x2, x1), 1)
            return self.shuffle(x)

    @default_module_settings(use_bn=True)
    def __init__(self, channels_scaler=1.0, num_blocks=[4, 8, 4], use_se=False):
        first_in_channels = ShuffleNetV2.FIRST_STAGE_CHANNELS[channels_scaler]
        last_stage_out_channels = first_in_channels * (2 ** (len(num_blocks) - 1))
        final_out_channels = 1024 if last_stage_out_channels < 800 else 2048

        super().__init__(final_out_channels)

        blocks = []
        for i, n in enumerate(num_blocks):
            out_channels = first_in_channels * (2 ** i)
            in_channels = 24 if i == 0 else out_channels // 2
            blocks.extend(self._make_stage(in_channels, out_channels, i, n, use_se))

        self.features = torch.nn.Sequential(collections.OrderedDict([('conv0', Conv2dAct(3, 24, kernel_size=3, padding=1, stride=2)),
                                                                     ('pool0', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]
                                                                    + blocks
                                                                    + [('conv1', Conv2dAct(last_stage_out_channels, final_out_channels, kernel_size=1)),
                                                                       ('pool1', torch.nn.AdaptiveAvgPool2d(1)),
                                                                       ('flatten', torch.nn.Flatten())]))

    def _make_stage(self, in_channels, out_channels, index, num_blocks, use_se):
        blocks = [(f'block{index}_0', ShuffleNetV2.DownsampleBasicBlock(in_channels, out_channels, use_se))]
        for i in range(num_blocks - 1):
            blocks.append((f'block{index}_{i+1}', ShuffleNetV2.BasicBlock(out_channels, out_channels, use_se)))
        return blocks
