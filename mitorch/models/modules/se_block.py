"""Squeeze-and-Excitation (SE) block (https://arxiv.org/abs/1709.01507)"""
import torch
from .base import ModuleBase
from .activation import HardSigmoid, Swish


class SEBlock(ModuleBase):
    def __init__(self, in_channels, reduction_ratio, use_hsigmoid=False, use_swish=False):
        super(SEBlock, self).__init__()

        self.in_channels = in_channels
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc0 = torch.nn.Linear(in_channels, in_channels // reduction_ratio)
        self.activation0 = Swish() if use_swish else torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = HardSigmoid() if use_hsigmoid else torch.nn.Sigmoid()

    def forward(self, input):
        x = self.pool(input)
        x = x.view(-1, self.in_channels)
        x = self.activation0(self.fc0(x))
        x = self.sigmoid(self.fc1(x))
        x = x.view(-1, self.in_channels, 1, 1)
        return input * x
