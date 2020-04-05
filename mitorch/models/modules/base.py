import torch


class ModuleBase(torch.nn.Module):
    VERSION = (0, 0)

    def apply_settings(self, args):
        pass

    def reset_parameters(self):
        pass
