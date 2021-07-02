from .base import ModuleBase
from .convolution import Conv2dAct


class DepthwiseSeparableConv2d(ModuleBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_bn=True, use_bn2=True, activation='relu', activation2='relu'):
        super().__init__()
        self.depthwise_conv = Conv2dAct(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, use_bn=use_bn, activation=activation)
        self.pointwise_conv = Conv2dAct(in_channels, out_channels, kernel_size=1, use_bn=use_bn2, activation=activation2)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)
