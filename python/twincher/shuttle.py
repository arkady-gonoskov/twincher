# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

import json
import os
import struct
import numpy as np
from . import _core

# Format of twincher files (.twc), version 1:
#   8 bytes    magic string b"TWINCHER"
#   4 bytes    format version (uint32, little-endian)
#   8 bytes    size of the metadata in bytes (uint64, little-endian)
#   metadata   JSON, UTF-8
#   core data  structure and parameters of the twincher, see twincher::save() in twincher.h
# Files written before the introduction of format versions (version 0) start directly with
# the size of the metadata.
_TWC_MAGIC = b"TWINCHER"
_TWC_VERSION = 1

def _read_header(file_name):
    """Read the header of a twincher file; return (metadata, offset of the core data)."""
    with open(file_name, "rb") as f:
        head = f.read(8)
        if head == _TWC_MAGIC:
            header = f.read(12)
            if len(header) != 12:
                raise ValueError(f"{file_name} is corrupted (incomplete header)")
            version, json_size = struct.unpack("<IQ", header)
            if version > _TWC_VERSION:
                raise ValueError(
                    f"{file_name}: file format version {version} is not supported by this version "
                    f"of twincher (supported: up to {_TWC_VERSION}); please update twincher"
                )
            offset = 20 + json_size
        elif len(head) == 8: # version 0
            json_size = struct.unpack("<Q", head)[0]
            offset = 8 + json_size
        else:
            raise ValueError(f"{file_name} is not a twincher file")
        if offset > os.fstat(f.fileno()).st_size:
            raise ValueError(f"{file_name} is not a twincher file or is corrupted (invalid size of metadata)")
        json_data = f.read(json_size)
    try:
        metadata = json.loads(json_data.decode("utf-8"))
    except ValueError as e: # includes JSON and UTF-8 decoding errors
        raise ValueError(f"{file_name} is not a twincher file or is corrupted ({e})") from None
    return metadata, offset

def allocate_array(shape, data_type):
    if(data_type == "np"):
        return np.empty(shape, dtype=np.float64)
    elif (data_type == "cp"):
        import cupy as cp
        return cp.empty(shape, dtype=cp.float32)
    elif (data_type == "torch"):
        import torch
        return torch.empty(shape, dtype=torch.float32, device="cuda")
    elif (data_type == "torch_cpu"):
        import torch
        return torch.empty(shape, dtype=torch.float64, device="cpu")
    else:
        raise TypeError("Unsupported dtype=", data_type)

def get_ptr(array, data_type, shape=None):
    """
    Return the memory address of the data of an array to be bound to the computational core.

    The core accesses the data through this raw address, assuming a C-contiguous array with
    the element type and device given by data_type. These properties, and the shape if
    given, are therefore checked here.
    """
    if data_type == "np":
        expected = "a C-contiguous numpy.ndarray of dtype float64"
        valid = isinstance(array, np.ndarray) and array.dtype == np.float64 and array.flags.c_contiguous
    elif data_type == "cp":
        import cupy as cp
        expected = "a C-contiguous cupy.ndarray of dtype float32 on the current CUDA device"
        valid = (isinstance(array, cp.ndarray) and array.dtype == cp.float32 and array.flags.c_contiguous
                 and array.device.id == cp.cuda.runtime.getDevice())
    elif data_type == "torch":
        import torch
        expected = "a contiguous torch.Tensor of dtype float32 on the current CUDA device"
        valid = (isinstance(array, torch.Tensor) and array.dtype == torch.float32 and array.is_contiguous()
                 and array.is_cuda and array.device.index == torch.cuda.current_device())
    elif data_type == "torch_cpu":
        import torch
        expected = "a contiguous torch.Tensor of dtype float64 on the CPU"
        valid = (isinstance(array, torch.Tensor) and array.dtype == torch.float64 and array.is_contiguous()
                 and array.device.type == "cpu")
    else:
        raise ValueError(f"unsupported data_type={data_type!r}")
    if not valid:
        description = f"{type(array).__module__}.{type(array).__name__}"
        for attribute in ("dtype", "device"):
            if hasattr(array, attribute):
                description += f", {attribute}={getattr(array, attribute)}"
        raise TypeError(f"data_type={data_type!r} requires {expected}, got {description}")
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError(f"expected an array of shape {tuple(shape)}, got {tuple(array.shape)}")
    if data_type == "np":
        return array.ctypes.data
    if data_type == "cp":
        return array.data.ptr
    return array.data_ptr()

def info():
    """Print the version, license and GPU support of twincher."""
    _core.info()
    print(
        f"CUDA backend: {'built' if _core.cuda_enabled else 'not built'}, "
        f"GPU: {'available' if _core.gpu_available() else 'not available'}",
        flush=True,
    )

_CPU_DATA_TYPES = ("np", "torch_cpu")
_GPU_DATA_TYPES = ("cp", "torch")

def _core_class(data_type):
    """Return the class of the computational core that operates arrays of the given data type."""
    if data_type in _CPU_DATA_TYPES:
        return _core.ShuttleCPU
    if data_type in _GPU_DATA_TYPES:
        if not _core.cuda_enabled:
            raise RuntimeError(
                f"data_type={data_type!r} requires the GPU backend, but twincher was built "
                "without CUDA support (no CUDA compiler was found when it was installed)"
            )
        if not _core.gpu_available():
            raise RuntimeError(f"data_type={data_type!r} requires a CUDA GPU, but none is available")
        return _core.ShuttleGPU
    raise ValueError(
        f"unsupported data_type={data_type!r}; expected one of "
        f"{', '.join(repr(t) for t in _CPU_DATA_TYPES + _GPU_DATA_TYPES)}"
    )

class Shuttle:
    def __init__(self, data_type="default", n_s=2, n_l=1, n_c=1, n_p=1, n_m=1, tw_type=1, rng_seed=42):

        if(data_type == "default"):
            # PyTorch tensors: float32 on GPU if possible, otherwise float64 on CPU
            from .tw_util import select_device
            data_type = "torch" if select_device("auto") == "cuda" else "torch_cpu"
        self.dtype = data_type

        # main parameters:
        self.n_s = n_s
        self.n_l = n_l
        self.n_c = n_c
        self.n_p = n_p
        self.n_m = n_m
        self.tw_type = tw_type

        # core initiation:
        self.core = _core_class(data_type)(n_s, n_l, n_c, n_p, n_m, tw_type, rng_seed)

        self.n_t = self.core.n_t

        self.allocate_init()

        self.time_monitor = None

        self.metadata = {"twincher_type": "untrained"}

    def enable_time_monitor(self, time_monitor):
        self.time_monitor = time_monitor
        if self.dtype == "torch":
            print(
                "WARNING: twincher::Shuttle::enable_time_monitor(): time monitoring synchronizes the GPU "
                "around each operation, which can affect performance; to disable it, pass "
                "time_monitor=None (e.g. to twincher.Learner)"
            )
    
    def tw_load(self, file_name, n_c, n_p, n_m):
        # core initiation:
        self.core = _core_class(self.dtype)(file_name, n_c, n_p, n_m)

        self.n_t = self.core.n_t
        
        # main parameters:
        self.n_s = self.core.n_s
        self.n_l = self.core.n_l
        self.tw_type = self.core.tw_type
        n_s, n_l = self.n_s, self.n_l
        self.n_c = n_c
        self.n_p = n_p
        self.n_m = n_m

        self.allocate_init()

    def allocate_init(self):
        n_s, n_l, n_c, n_p, n_m, n_t, dtype = self.n_s, self.n_l, self.n_c, self.n_p, self.n_m, self.n_t, self.dtype

        # memory allocation:
        self._s = allocate_array((n_s, n_c), dtype) # s, field
        self._V = allocate_array((n_p, n_s, n_c), dtype) # dr_ds
        self._Q = allocate_array((n_s, n_m, n_c), dtype) # ds_dm
        self._B = allocate_array((n_s, n_m, n_c), dtype) # dL_du, u = ds_dm
        self._b = allocate_array((n_s, n_c), dtype) # dL_ds
        if(dtype == "torch"):
            import torch
            self.param = torch.empty((4*n_t), dtype=torch.float32, device="cuda", requires_grad=True)
            self.param.grad = torch.empty_like(self.param)
            self._a = self.param.detach() # a, param
            self._g = self.param.grad
            self.core.cuda_stream = torch.cuda.current_stream().cuda_stream
        elif(dtype == "torch_cpu"):
            import torch
            self.param = torch.empty((4*n_t), dtype=torch.float64, device="cpu", requires_grad=True)
            self.param.grad = torch.empty_like(self.param)
            self._a = self.param.detach() # a, param
            self._g = self.param.grad # dL_da, grad
        else:
            self._a = allocate_array((4*n_t), dtype) # a, param
            self._g = allocate_array((4*n_t), dtype) # dL_da, grad

        # arrays of full size (see resize_n_m)
        self._Q_original = self._Q
        self._B_original = self._B

        # bind arrays and initiate the state of core
        self.core.init(
            s_ptr=get_ptr(self._s, dtype, (n_s, n_c)),
            a_ptr=get_ptr(self._a, dtype, (4*n_t,)),
            V_ptr=get_ptr(self._V, dtype, (n_p, n_s, n_c)),
            Q_ptr=get_ptr(self._Q, dtype, (n_s, n_m, n_c)),
            B_ptr=get_ptr(self._B, dtype, (n_s, n_m, n_c)),
            b_ptr=get_ptr(self._b, dtype, (n_s, n_c)),
            g_ptr=get_ptr(self._g, dtype, (4*n_t,)),
        )
    

    def set_rnd_t(self, rng_seed = 42):
        self.core.set_rnd_t(rng_seed=rng_seed)

    def forward_inference(self):
        if self.time_monitor is not None: self.time_monitor.start("twinch::forward_inference", self.n_c*self.n_t, torch_sync=(self.dtype == "torch"))
        self.core.forward_inference()
        if self.time_monitor is not None: self.time_monitor.stop("twinch::forward_inference", torch_sync=(self.dtype == "torch"))

    def backward_inference(self):
        if self.time_monitor is not None: self.time_monitor.start("twinch::backward_inference", self.n_c*self.n_t, torch_sync=(self.dtype == "torch"))
        self.core.backward_inference()
        if self.time_monitor is not None: self.time_monitor.stop("twinch::backward_inference", torch_sync=(self.dtype == "torch"))

    def compute_V(self):
        if self.time_monitor is not None: self.time_monitor.start("twinch::compute_V", self.n_c*self.n_t, torch_sync=(self.dtype == "torch"))
        self.core.compute_V()
        if self.time_monitor is not None: self.time_monitor.stop("twinch::compute_V", torch_sync=(self.dtype == "torch"))

    def tw_save(self, file_name):
        self.core.tw_save(file_name)

    def forward_query(self):
        if self.time_monitor is not None: self.time_monitor.start("twinch::forward_query", self.n_c*self.n_t, torch_sync=(self.dtype == "torch"))
        self.core.forward_query()
        if self.time_monitor is not None: self.time_monitor.stop("twinch::forward_query", torch_sync=(self.dtype == "torch"))

    def backward_query(self):
        if self.time_monitor is not None: self.time_monitor.start("twinch::backward_query", self.n_c*self.n_t, torch_sync=(self.dtype == "torch"))
        self.core.backward_query()
        if self.time_monitor is not None: self.time_monitor.stop("twinch::backward_query", torch_sync=(self.dtype == "torch"))

    def tw_update(self):
        if self.time_monitor is not None: self.time_monitor.start("twinch::tw_update", self.n_t, torch_sync=(self.dtype == "torch"))
        self.core.tw_update()
        if self.time_monitor is not None: self.time_monitor.stop("twinch::tw_update", torch_sync=(self.dtype == "torch"))

    @property 
    def s(self): return self._s
    @property
    def a(self): return self._a
    @property
    def V(self): return self._V
    @property
    def Q(self): return self._Q
    @property
    def B(self): return self._B
    @property
    def b(self): return self._b
    @property
    def g(self): return self._g

    def resize_n_m(self, n_m_new):
        if n_m_new > self._Q_original.shape[1]:
            raise ValueError("n_m_new must be <= n_m at initiation")
        new_size = self.n_s*n_m_new*self.n_c
        # views of the beginning of the arrays bound to the core (reshape(-1) of a contiguous
        # array is a view for NumPy, CuPy and PyTorch, whereas flatten() copies in NumPy/CuPy)
        self._Q = self._Q_original.reshape(-1)[:new_size].reshape(self.n_s, n_m_new, self.n_c)
        self._B = self._B_original.reshape(-1)[:new_size].reshape(self.n_s, n_m_new, self.n_c)
        self.n_m = n_m_new
        self.core.n_m = n_m_new

    def save(self, file_name):
        metadata = json.dumps(self.metadata, indent=4).encode("utf-8")
        with open(file_name, "wb") as f:
            f.write(_TWC_MAGIC)
            f.write(struct.pack("<I", _TWC_VERSION))
            f.write(struct.pack("<Q", len(metadata)))
            f.write(metadata)
        self.core.tw_save(file_name, True)

    def load(self, file_name, n_c, n_p, n_m):
        self.metadata, offset = _read_header(file_name)

        # core initiation:
        self.core = _core_class(self.dtype)(file_name, n_c, n_p, n_m, offset)

        self.n_t = self.core.n_t
        
        # main parameters:
        self.n_s = self.core.n_s
        self.n_l = self.core.n_l
        self.tw_type = self.core.tw_type
        n_s, n_l = self.n_s, self.n_l
        self.n_c = n_c
        self.n_p = n_p
        self.n_m = n_m

        self.allocate_init()

def load_metadata(file_name):
    return _read_header(file_name)[0]


