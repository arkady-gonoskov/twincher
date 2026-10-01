# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

"""Unit tests of the Python wrapper Shuttle, for all data types."""

import numpy as np
import pytest
import twincher
from twincher.shuttle import get_ptr
from helpers import set_rnd, to_np, copy

def make_shuttle(data_type, n_m=2, **kwargs):
    SH = twincher.Shuttle(data_type=data_type, n_s=8, n_l=4, n_c=4, n_p=2, n_m=n_m, rng_seed=1, **kwargs)
    SH.set_rnd_t(5)
    return SH

def test_default_data_type():
    import torch
    SH = twincher.Shuttle(n_s=4, n_l=2)
    gpu = twincher._core.gpu_available() and torch.cuda.is_available()
    assert SH.dtype == ("torch" if gpu else "torch_cpu")

def test_unknown_data_type():
    with pytest.raises(ValueError, match="data_type"):
        twincher.Shuttle(data_type="float64")

@pytest.mark.parametrize("gpu_type", ["torch", "cp"])
@pytest.mark.parametrize("cuda_enabled, message", [
    (True, "requires a CUDA GPU, but none is available"),
    (False, "twincher was built without CUDA support"),
], ids=["no-gpu", "no-cuda-backend"])
def test_gpu_data_type_without_gpu(gpu_type, cuda_enabled, message, monkeypatch):
    # simulates systems on which twincher cannot use a GPU
    monkeypatch.setattr(twincher._core, "cuda_enabled", cuda_enabled)
    monkeypatch.setattr(twincher._core, "gpu_available", lambda: False)
    with pytest.raises(RuntimeError, match=message):
        twincher.Shuttle(data_type=gpu_type)

def test_resize_n_m(data_type):
    # after the reduction of n_m, the arrays Q and B must remain those bound to the core
    SH = make_shuttle(data_type, n_m=3)
    SH.resize_n_m(2)
    reference = make_shuttle(data_type, n_m=2)
    results = []
    for S in (SH, reference):
        assert tuple(S.Q.shape) == (8, 2, 4) and tuple(S.B.shape) == (8, 2, 4)
        set_rnd(S.s)
        set_rnd(S.Q)
        S.forward_query()
        results.append(to_np(S.Q))
    assert np.abs(results[0]).max() > 0
    assert np.array_equal(results[0], results[1])

def test_resize_n_m_cannot_grow(data_type):
    SH = make_shuttle(data_type, n_m=2)
    with pytest.raises(ValueError):
        SH.resize_n_m(3)

def test_backward_inference_keeps_g(data_type):
    SH = make_shuttle(data_type)
    set_rnd(SH.s)
    s_in = copy(SH.s)
    SH.g[:] = 7.0
    SH.forward_inference()
    SH.backward_inference()
    assert (to_np(SH.g) == 7.0).all()
    assert np.abs(to_np(SH.s) - to_np(s_in)).max() < 1e-5

def test_set_tw_config_updates_twinches(data_type):
    # the gradient right after set_tw_config must equal that after an explicit tw_update
    gradients = []
    for explicit_update in (False, True):
        SH = make_shuttle(data_type)
        SH.core.set_tw_config(0.02, 0.2, 1.5)
        if explicit_update:
            SH.tw_update()
        SH.s[:] = 0.3
        SH.Q[:] = 0.5
        SH.forward_query()
        SH.B[:] = 1.0
        SH.b[:] = 0.1
        SH.backward_query()
        gradients.append(to_np(SH.g))
    assert np.abs(gradients[0]).max() > 0
    assert np.array_equal(gradients[0], gradients[1])

def test_cpu_threads():
    SH = twincher.Shuttle(data_type="np", n_s=16, n_l=8, n_c=600, n_p=1, n_m=1)
    SH.set_rnd_t()
    assert SH.core.n_cores == 0 # OpenMP default
    results = []
    for n_cores in (1, 2, 0):
        SH.core.n_cores = n_cores
        set_rnd(SH.s)
        SH.forward_inference()
        results.append(SH.s.copy())
    assert np.array_equal(results[0], results[1]) and np.array_equal(results[0], results[2])

# ------------------ validation of arrays bound to the core (get_ptr) ------------------

def test_get_ptr_accepts_valid_arrays():
    array = np.zeros((4, 3))
    assert get_ptr(array, "np", (4, 3)) == array.ctypes.data

@pytest.mark.parametrize("array, error", [
    (np.zeros((4, 3), dtype=np.float32), TypeError), # wrong dtype
    (np.zeros((4, 6))[:, ::2], TypeError),           # not contiguous
    (np.zeros((3, 4)), ValueError),                  # wrong shape
    ([[0.0]*3]*4, TypeError),                        # not an array
], ids=["dtype", "non-contiguous", "shape", "list"])
def test_get_ptr_rejects_invalid_numpy_arrays(array, error):
    with pytest.raises(error):
        get_ptr(array, "np", (4, 3))

def test_get_ptr_rejects_invalid_torch_tensors():
    import torch
    with pytest.raises(TypeError):
        get_ptr(torch.zeros(4, 3, dtype=torch.float32), "torch_cpu", (4, 3)) # wrong dtype
    with pytest.raises(TypeError):
        get_ptr(np.zeros((4, 3)), "torch_cpu", (4, 3)) # wrong library
    with pytest.raises(TypeError):
        get_ptr(torch.zeros(4, 3, dtype=torch.float32), "torch", (4, 3)) # not on GPU
