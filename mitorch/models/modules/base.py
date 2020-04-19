import functools
import torch

module_settings = {}


def default_module_settings(**settings_kwargs):
    """
    Module configurations priority.
    1. config key name with '!' prefix.
    2. configurations given to the module constructor.
    3. First configs given to default_module_settings().
    """
    def settings_wrapper2(f):
        @functools.wraps(f)
        def settings_wrapper(*args, **kwargs):
            global module_settings
            old_module_settings = module_settings.copy()
            for key in settings_kwargs:
                if key not in module_settings:
                    module_settings[key] = settings_kwargs[key]
            result = f(*args, **kwargs)
            module_settings = old_module_settings
            return result
        return settings_wrapper
    return settings_wrapper2


class ModuleBase(torch.nn.Module):
    VERSION = (0, 0)

    def __init__(self, **kwargs):
        super().__init__()
        global module_settings
        values = kwargs
        for key in module_settings:
            if key[0] == '!':
                values[key[1:]] = module_settings[key]
            elif key not in values:
                values[key] = module_settings[key]

        self.module_settings = values

    def reset_parameters(self):
        pass
