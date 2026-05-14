"""MRVP package."""

import os as _os

try:
    import torch as _torch
    _torch.set_num_threads(max(1, min(4, _os.cpu_count() or 1)))
except Exception:
    _MRVP_THREAD_SETUP = False
else:
    _MRVP_THREAD_SETUP = True
