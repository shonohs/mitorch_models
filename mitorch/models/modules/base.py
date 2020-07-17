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
            with set_module_settings(**settings_kwargs):
                return f(*args, **kwargs)
        return settings_wrapper
    return settings_wrapper2


# If 'none' activation is specified by kwargs, it has the highest priority.
# This is because some architectures need blocks without activation.
_PRIORITY_KWARGS_SETTINGS = {'activation': 'none', 'activation2': 'none'}


def _has_kwargs_high_priority(kwargs, name):
    return name in kwargs and kwargs[name] == _PRIORITY_KWARGS_SETTINGS.get(name, 'NOTFOUND')


def get_module_settings(**kwargs):
    global _module_settings
    values = kwargs
    for key in _module_settings:
        if key[0] == '!':
            if not _has_kwargs_high_priority(kwargs, key[1:]):
                values[key[1:]] = _module_settings[key]
        elif key not in values:
            values[key] = _module_settings[key]
    return values


class ModuleBase(torch.nn.Module):
    VERSION = (0, 0)

    def __init__(self, **kwargs):
        super().__init__()
        self.module_settings = get_module_settings(**kwargs)

    def reset_parameters(self):
        pass
