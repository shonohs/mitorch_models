import torch
from .modules import ModuleBase

class Model(torch.nn.Module):
    def __init__(self, output_dim, **kwargs):
        super(Model, self).__init__()
        self.output_dim = output_dim
        self.modules_kwargs = kwargs

    def post_init(self):
        for m in self.modules():
            if isinstance(m, ModuleBase):
                m.apply_settings(self.modules_kwargs)
