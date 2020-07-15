import torch
from .base import ModuleBase


class LinearAct(ModuleBase):
    VERSION = (0, 1)

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, input):
        return self.activation(self.fc(input))

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(self.fc.bias)


class Linear(ModuleBase):
    VERSION = (0, 1)

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.fc.weight, model='fan_in', nonlinearity='linear')
        torch.nn.init.zeros_(self.fc.bias)
