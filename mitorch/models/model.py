import functools
import torch
from .modules.base import ModuleBase


class Model(torch.nn.Module):
    # (Major version, Minor version).
    # Major version updates can change the training results.
    # Minor version updates doesn't have impact on model outputs. e.g. module name changes.
    VERSION = (0, 0)

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def get_output_shapes(self, output_names):
        input = torch.randn(1, 3, 224, 224)
        outputs = self.forward(input, output_names)
        return [o.shape[1] for o in outputs]

    @property
    def version(self):
        versions = {}
        for m in self.modules():
            if isinstance(m, (Model, ModuleBase)):
                module_name = type(m).__name__
                module_version = m.VERSION
                versions[module_name] = module_version
        return versions

    def forward(self, input, output_names=None):
        if not output_names:
            if hasattr(self, 'features'):
                return self.features(input)
            else:
                raise NotImplementedError
        else:
            # TODO: Is there more efficient way to extract values?
            forward_hooks = []
            for i, name in enumerate(output_names):
                m = self._find_module_by_name(name)
                forward_hooks.append(m.register_forward_hook(functools.partial(self._extract_outputs_hook, index=i)))
            self._outputs = [None] * len(output_names)

            self.forward(input)

            for hook in forward_hooks:
                hook.remove()
            assert all([o is not None for o in self._outputs])
            return self._outputs

    def _find_module_by_name(self, name):
        paths = name.split('.')
        current = self
        for p in paths:
            children = current.named_children()
            for name, c in children:
                if name == p:
                    current = c
                    break

        return current

    def _extract_outputs_hook(self, module, input, output, index):
        self._outputs[index] = output

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, ModuleBase):
                m.reset_parameters()
