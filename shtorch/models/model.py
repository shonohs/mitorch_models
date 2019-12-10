import torch
from .modules import ModuleBase

class Model(torch.nn.Module):
    MAJOR_VERSION = 0 # Major updates where the results can be changed.
    MINOR_VERSION = 0 # Minor updates that doesn't have impact on model outputs. e.g. module name changes.

    def __init__(self, output_dim, **kwargs):
        super(Model, self).__init__()
        self.output_dim = output_dim
        self.modules_kwargs = kwargs

    def __setattr__(self, name, value):
        if isinstance(value, torch.nn.Module):
            for m in value.modules():
                apply_settings = getattr(m, 'apply_settings', None)
                if apply_settings and callable(apply_settings):
                    m.apply_settings(self.modules_kwargs)

        super(Model, self).__setattr__(name, value)
