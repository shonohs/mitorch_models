import torch
from .base import ModuleBase
from .activation import HardSwish, Swish


class Conv2dAct(ModuleBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bn=True, activation=None):
        super(Conv2dAct, self).__init__()
        self.explicit_settings = {'use_bn': use_bn, 'activation': activation}

        use_bn = use_bn if use_bn is not None else True
        activation = activation if activation is not None else 'relu'

        self.out_channels = out_channels
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=(not use_bn))
        self.bn = torch.nn.BatchNorm2d(out_channels) if use_bn else None
        self._set_activation(activation)

    def forward(self, input):
        x = self.conv(input)
        if self.bn:
            x = self.bn(x)
        return self.activation(x)

    def apply_settings(self, kwargs):
        if self.explicit_settings['use_bn'] is None and kwargs.get('use_bn') is not None:
            use_bn = kwargs.get('use_bn')
            self.bn = torch.nn.BatchNorm2d(self.out_channels) if use_bn else None
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=(not use_bn))

        if self.explicit_settings['activation'] is None and kwargs.get('activation') is not None:
            self._set_activation(kwargs.get('activation'))

    def _set_activation(self, act):
        assert act in ['relu', 'hswish', 'swish', 'relu6']

        if act == 'relu':
            self.activation = torch.nn.ReLU(inplace=True)
        elif act == 'hswish':
            self.activation = HardSwish()
        elif act == 'swish':
            self.activation = Swish()
        elif act == 'relu6':
            self.activation = torch.nn.ReLU6(inplace=True)


class Conv2dBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(Conv2dBN, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, input):
        return self.bn(self.conv(input))
