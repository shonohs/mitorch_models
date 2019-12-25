import torch

class ModuleBase(torch.nn.Module):
    def apply_settings(self, args):
        pass

    def reset_parameters(self):
        pass
