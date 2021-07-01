import torch
from .base import ModuleBase


class LinearAct(ModuleBase):
    def __init__(self, in_channels, out_channels, use_bn=True, sync_bn=False):
        super().__init__()

        self.fc = torch.nn.Linear(in_channels, out_channels)
        self.bn = (torch.nn.SyncBatchNorm if sync_bn else torch.nn.BatchNorm2d)(out_channels) if use_bn else None
        self.activation = torch.nn.ReLU()

    def forward(self, input):
        x = self.fc(input)
        if self.bn:
            x = self.bn(x)
        return self.activation(x)

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(self.fc.bias)


class Linear(ModuleBase):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='linear')
        torch.nn.init.zeros_(self.fc.bias)
