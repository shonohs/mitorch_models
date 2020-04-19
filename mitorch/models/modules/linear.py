import torch
from .base import ModuleBase


class LinearAct(ModuleBase):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, input):
        return self.activation(self.fc(input))
