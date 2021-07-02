from .addition import Add
from .base import ModuleBase
from .convolution import Conv2dAct, Conv2dBN
from .se_block import SEBlock


class MBConv(ModuleBase):
    def __init__(self, in_channels, out_channels, expansion_channels, kernel_size=3, stride=1, use_se=True, use_se_hsigmoid=True, activation='hswish', se_activation='relu6'):
        super().__init__()
        self.conv0 = Conv2dAct(in_channels, expansion_channels, kernel_size=1, activation=activation) if in_channels != expansion_channels else None
        self.conv1 = Conv2dAct(expansion_channels, expansion_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                               stride=stride, groups=expansion_channels, activation=activation)
        self.conv2 = Conv2dBN(expansion_channels, out_channels, kernel_size=1)

        self.se = SEBlock(expansion_channels, reduction_ratio=4, use_hsigmoid=use_se_hsigmoid, activation=se_activation) if use_se else None
        self.residual = Add() if stride == 1 and in_channels == out_channels else None

    def forward(self, input):
        x = self.conv0(input) if self.conv0 else input
        x = self.conv1(x)

        if self.se:
            x = self.se(x)

        x = self.conv2(x)

        if self.residual:
            x = self.residual(x, input)

        return x
