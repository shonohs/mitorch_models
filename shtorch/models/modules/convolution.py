import torch
from .base import ModuleBase

class Conv2dBNRelu(ModuleBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(Conv2dBNRelu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, input):
        return self.relu(self.bn(self.conv(input)))

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
