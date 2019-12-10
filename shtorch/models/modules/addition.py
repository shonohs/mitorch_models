from .base import ModuleBase


class Add(ModuleBase):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, input0, input1):
        return input0 + input1
