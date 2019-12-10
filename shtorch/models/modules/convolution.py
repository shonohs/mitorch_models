import torch
from .base import ModuleBase
from .activation import HardSwish, Swish


class Conv2dBNRelu(ModuleBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_hswish=False, use_swish=False):
        super(Conv2dBNRelu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.activation = HardSwish() if use_hswish else (Swish() if use_swish else torch.nn.ReLU(inplace=True))

    def forward(self, input):
        return self.activation(self.bn(self.conv(input)))

    def apply_settings(self, kwargs):
        if kwargs.get('use_relu6'):
            self.relu = torch.nn.ReLU6(inplace=True)


class Conv2dBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(Conv2dBN, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, input):
        return self.bn(self.conv(input))


class Conv2dRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_hswish=False):
        super(Conv2dRelu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.activation = HardSwish() if use_hswish else torch.nn.ReLU(inplace=True)

    def forward(self, input):
        return self.activation(self.conv(input))

    def apply_settings(self, kwargs):
        if kwargs.get('use_relu6'):
            self.activation = torch.nn.ReLU6(inplace=True)
