import collections
import functools
import torch


class Model(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    @torch.no_grad()
    def get_output_shapes(self, output_names):
        input = torch.randn(1, 3, 224, 224)
        outputs = self.forward(input, output_names)
        return [o.shape[1] for o in outputs]

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
            results = self._outputs
            self._outputs = None
            return results

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
        assert self._outputs[index] is None
        self._outputs[index] = output

    def reset_parameters(self):
        for m in self.children():
            Model._call_reset_parameters(m)

    @staticmethod
    def _call_reset_parameters(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        elif isinstance(m, collections.abc.Iterable):
            for c in m:
                Model._call_reset_parameters(c)
