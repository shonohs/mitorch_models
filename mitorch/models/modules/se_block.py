"""Squeeze-and-Excitation (SE) block (https://arxiv.org/abs/1709.01507)"""
import torch
from .base import ModuleBase
from .activation import HardSigmoid, Swish


class SEBlock(ModuleBase):
    def __init__(self, in_channels, reduction_ratio, use_hsigmoid=False, use_swish=False):
        super(SEBlock, self).__init__()

        self.in_channels = in_channels
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv0 = torch.nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.activation0 = Swish() if use_swish else torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = HardSigmoid() if use_hsigmoid else torch.nn.Sigmoid()

    def forward(self, input):
        x = self.pool(input)
        x = self.activation0(self.conv0(x))
        x = self.sigmoid(self.conv1(x))
        return input * x
