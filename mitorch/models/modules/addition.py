from .base import ModuleBase


class Add(ModuleBase):
    def forward(self, *inputs):
        x = inputs[0]
        for i in inputs[1:]:
            x = x + i
        return x
