import torch
from .base import ModuleBase


class Swish(ModuleBase):
    def forward(self, input):
        return input * torch.sigmoid(input)


class HardSigmoid(ModuleBase):
    def forward(self, input):
        return torch.nn.functional.relu6(input + 3) / 6


class HardSwish(ModuleBase):
    def forward(self, input):
        return input * torch.nn.functional.relu6(input + 3) / 6
