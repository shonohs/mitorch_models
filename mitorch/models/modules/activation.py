import torch
from .base import ModuleBase, default_module_settings


class Activation(ModuleBase):
    @default_module_settings(activation='relu')
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        activation = self.module_settings['activation']

        act_class = {'relu': torch.nn.ReLU,
                     'relu6': torch.nn.ReLU6,
                     'hswish': HardSwish,
                     'swish': Swish,
                     'none': None}[activation]
        self.activation = act_class()

    def forward(self, x):
        return self.activation(x)


class Swish(ModuleBase):
    def forward(self, input):
        return input * torch.sigmoid(input)


class HardSigmoid(ModuleBase):
    def forward(self, input):
        return torch.nn.functional.relu6(input + 3) / 6


class HardSwish(ModuleBase):
    def forward(self, input):
        return input * torch.nn.functional.relu6(input + 3) / 6
