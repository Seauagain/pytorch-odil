import os
import sys

import numpy as np
import torch
from .backend import ModTorch

if not int(os.environ.get("ODIL_MT", 0)):
    os.environ["OMP_NUM_THREADS"] = "1"

enable_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ["", "-1"]

if not enable_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if not int(os.environ.get("ODIL_WARN", 0)):
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=Warning)

enable_jit = bool(int(os.environ.get("ODIL_JIT", 0)))

backend_name = os.environ.get("ODIL_BACKEND", "")


backend_name = "torch"

DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda:0"
    torch.cuda.empty_cache()
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            DEVICE = "npu:0"
            torch.npu.empty_cache()
            torch.npu.set_device(DEVICE)
        else:
            DEVICE = "cpu"
    except ImportError:
        DEVICE = "cpu"

mod = ModTorch(torch)
tf = None
jax = None


# Default data type.
dtype_name = os.environ.get("ODIL_DTYPE", "float32")

if dtype_name in ["float32", "float64"]:
    # dtype = numpy.dtype(dtype_name)
    DTYPE = getattr(torch, dtype_name)
else:
    sys.stderr.write(f"Expected ODIL_DTYPE=float32 or float64, got '{dtype}' \n")
    exit(1)



cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
width = 50
print("=" * width)
print(f"Available device: {DEVICE}".center(width))
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}".center(width))
print("=" * width)






