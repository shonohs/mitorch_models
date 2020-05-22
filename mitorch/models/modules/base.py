import contextlib
import functools
import torch

_module_settings = {}


@contextlib.contextmanager
def set_module_settings(**kwargs):
    global _module_settings
    old_module_settings = _module_settings.copy()
    try:
        for key in kwargs:
            if key not in _module_settings:
                _module_settings[key] = kwargs[key]
        yield
    finally:
        _module_settings = old_module_settings


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
            global _module_settings
            old_module_settings = _module_settings.copy()
            for key in settings_kwargs:
                if key not in _module_settings:
                    _module_settings[key] = settings_kwargs[key]
            result = f(*args, **kwargs)
            _module_settings = old_module_settings
            return result
        return settings_wrapper
    return settings_wrapper2


class ModuleBase(torch.nn.Module):
    VERSION = (0, 0)

    # If 'none' activation is specified by kwargs, it has the highest priority.
    # This is because some architectures need blocks without activation.
    PRIORITY_KWARGS_SETTINGS = {'activation': 'none', 'activation2': 'none'}

    def __init__(self, **kwargs):
        super().__init__()
        global _module_settings
        values = kwargs
        for key in _module_settings:
            if key[0] == '!':
                if not self._has_kwargs_high_priority(kwargs, key[1:]):
                    values[key[1:]] = _module_settings[key]
            elif key not in values:
                values[key] = _module_settings[key]

        self.module_settings = values

    def reset_parameters(self):
        pass

    @staticmethod
    def _has_kwargs_high_priority(kwargs, name):
        return name in kwargs and name in ModuleBase.PRIORITY_KWARGS_SETTINGS and kwargs[name] == ModuleBase.PRIORITY_KWARGS_SETTINGS[name]
