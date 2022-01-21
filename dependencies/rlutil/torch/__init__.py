import rlutil.torch.nn
from rlutil.torch.pytorch_util import set_gpu, default_device, to_numpy
from types import ModuleType
import torch.optim as optim
import pdb

def _replace_funcs(global_dict):
    import torch as th
    class DeviceWrapped(object): 
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args, device=None, **kwargs):
            if device is None:
                device = default_device()
            return self.fn(*args, device=device, **kwargs)

        def __repr__(self):
            return '<device-wrapped function %s>' % self.fn.__name__

    for _key in dir(th):
        _value = getattr(th, _key)
       # print(_key)
        if _key in ['ScriptClass']:
            global_dict[_key] = _value
            continue
        else: 
        # pdb.set_trace()       
        # hacky way of determining if device is an argument
        # (cannot use inspect because torch functions are builtins)   
           # print(_key)
            if isinstance(_value, ModuleType):
                pass
            elif _value.__doc__ and 'device (:class:`torch.device`, optional):' in _value.__doc__:
                global_dict[_key] = DeviceWrapped(_value)
               # print('In',_key)
            else:
                global_dict[_key] = _value

_replace_funcs(globals())
