from .base import ModuleBase


class Add(ModuleBase):
    def __init__(self, enable):
        super(Add, self).__init__()
        self.enable = enable

    def forward(self, input0, input1):
        return (input0 + input1) if self.enable else input0
