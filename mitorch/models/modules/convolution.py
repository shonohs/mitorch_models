import torch
from .base import ModuleBase
from .activation import HardSwish, Swish


class Conv2dAct(ModuleBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bn=True, sync_bn=False, activation='relu'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=(not use_bn))
        self.bn = (torch.nn.SyncBatchNorm if sync_bn else torch.nn.BatchNorm2d)(out_channels) if use_bn else None
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

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            torch.nn.init.zeros_(self.conv.bias)


class Conv2dBN(ModuleBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, sync_bn=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = (torch.nn.SyncBatchNorm if sync_bn else torch.nn.BatchNorm2d)(out_channels)

    def forward(self, input):
        return self.bn(self.conv(input))

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')


class Conv2d(ModuleBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)

    def forward(self, input):
        return self.conv(input)

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            torch.nn.init.zeros_(self.conv.bias)
