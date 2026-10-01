# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

"""Helpers shared by the tests: operations that work for NumPy, CuPy and PyTorch arrays."""

import numpy as np

def set_rnd(arr, rng_seed=42):
    """Fill an array with random values from [-1, 1)."""
    module_name = arr.__class__.__module__.split(".")[0]
    if module_name == "numpy":
        rng = np.random.default_rng(rng_seed)
        arr[:] = rng.uniform(-1, 1, size=arr.shape)
    elif module_name == "cupy":
        import cupy as cp
        rng = cp.random.default_rng(rng_seed)
        arr[:] = rng.random(arr.shape, dtype=arr.dtype)*2 - 1
    elif module_name == "torch":
        import torch
        generator = torch.Generator(device=arr.device)
        generator.manual_seed(rng_seed)
        arr[:] = torch.rand(arr.shape, dtype=arr.dtype, device=arr.device, generator=generator)*2 - 1
    else:
        raise TypeError(f"Unsupported array type: {type(arr)}")

def to_np(arr):
    """Return a NumPy copy of an array."""
    backend = type(arr).__module__.split(".")[0]
    if backend == "numpy":
        return arr.copy()
    elif backend == "cupy":
        import cupy as cp
        return cp.asnumpy(arr)
    elif backend == "torch":
        return arr.cpu().numpy()
    else:
        raise TypeError(f"Unsupported backend: {backend}")

def copy(arr):
    """Return a copy of an array in the same library and on the same device."""
    backend = type(arr).__module__.split(".")[0]
    if backend == "numpy":
        return arr.copy()
    elif backend == "cupy":
        return arr.copy()
    elif backend == "torch":
        return arr.clone()
    else:
        raise TypeError(f"Unsupported backend: {backend}")

def check(error, tolerance):
    """
    Assert that an error is below a tolerance. Both values are printed; pytest shows this
    output for failed tests, and for all tests with the option -rP.
    """
    print(f"error/tolerance: {error:.3e}/{tolerance:.3e}")
    assert error < tolerance, f"error {error:.3e} is not below the tolerance {tolerance:.3e}"
