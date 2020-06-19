import torch
from .base import ModuleBase


class LinearAct(ModuleBase):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, input):
        return self.activation(self.fc(input))

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(self.fc.bias)


class Linear(ModuleBase):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)

    def reset_parameters(self):
        torch.nn.init.normal_(self.fc.weight, 0, 0.01)
        torch.nn.init.zeros_(self.fc.bias)
