from .base import ModuleBase


class Add(ModuleBase):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, *inputs):
        x = inputs[0]
        for i in inputs[1:]:
            x = x + i
        return x
