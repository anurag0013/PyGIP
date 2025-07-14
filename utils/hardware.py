import os

import torch

_DEFAULT_DEVICE_STR = os.getenv("PYGIP_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
_default_device = torch.device(_DEFAULT_DEVICE_STR)


def get_device():
    return _default_device


def set_device(device_str):
    global _default_device
    _default_device = torch.device(device_str)
