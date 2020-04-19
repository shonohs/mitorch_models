from .base import ModuleBase, default_module_settings
from .convolution import Conv2dAct


class DepthwiseSeparableConv2d(ModuleBase):
    @default_module_settings(use_bn=True, use_bn2=True, activation='relu', activation2='relu')
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, **kwargs):
        super().__init__(**kwargs)
        self.depthwise_conv = Conv2dAct(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                        use_bn=self.module_settings['use_bn'], activation=self.module_settings['activation'])
        self.pointwise_conv = Conv2dAct(in_channels, out_channels, kernel_size=1, use_bn=self.module_settings['use_bn2'], activation=self.module_settings['activation2'])

    def forward(self, input):
        x = self.depthwise_conv(input)
        return self.pointwise_conv(x)
