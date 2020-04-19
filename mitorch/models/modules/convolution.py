import torch
from .base import ModuleBase, default_module_settings
from .activation import HardSwish, Swish


class Conv2dAct(ModuleBase):
    @default_module_settings(use_bn=True, activation='relu')
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, **kwargs):
        super().__init__(**kwargs)
        use_bn = self.module_settings['use_bn']
        activation = self.module_settings['activation']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=(not use_bn))
        self.bn = torch.nn.BatchNorm2d(out_channels) if use_bn else None
        self._set_activation(activation)

    def forward(self, input):
        x = self.conv(input)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

    def _set_activation(self, act):
        assert act in ['relu', 'hswish', 'swish', 'relu6', 'none']

        if act == 'relu':
            self.activation = torch.nn.ReLU(inplace=True)
        elif act == 'hswish':
            self.activation = HardSwish()
        elif act == 'swish':
            self.activation = Swish()
        elif act == 'relu6':
            self.activation = torch.nn.ReLU6(inplace=True)
        elif act == 'none':
            self.activation = None


class Conv2dBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, input):
        return self.bn(self.conv(input))
