# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

import contextlib
import gc
import warnings
from pathlib import Path

def output_dir(dir=None):
    """
    Return the directory for output files as a Path, creating it if necessary.
    dir can be a str or a Path; by default, "twincher_output" in the current working directory.
    """
    path = Path.cwd() / "twincher_output" if dir is None else Path(dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

def plot_style(style=None):
    """
    Return a context manager that applies a matplotlib style while a figure is created or
    saved. The global matplotlib settings of the user are restored afterwards. With
    style=None, the current settings are used.
    """
    if style is None:
        return contextlib.nullcontext()
    import matplotlib.pyplot as plt
    return plt.style.context(style)

def save_figure(fig, file_name, style=None):
    """Save a matplotlib figure to an image file, using the given style (see plot_style)."""
    with plot_style(style):
        fig.savefig(file_name, dpi=300, bbox_inches="tight")
    # Each savefig leaves its rendering buffer (about 50 MB at this resolution) in reference
    # cycles, which are freed only when the cyclic garbage collector of Python runs. Without
    # an explicit collection, repeated saving can accumulate gigabytes of memory.
    gc.collect()

def select_device(device="auto"):
    """
    Resolve the requested device ("auto", "cpu" or "cuda") into "cpu" or "cuda".

    "cuda" is used only if both PyTorch and twincher can use a CUDA GPU. Otherwise the
    computations fall back to the CPU, with a warning explaining the reason if a GPU was
    explicitly requested or appears to be present.
    """
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be 'auto', 'cpu' or 'cuda', got {device!r}")
    if device == "cpu":
        return "cpu"

    import torch
    from . import _core
    torch_cuda = torch.cuda.is_available()
    twincher_cuda = _core.gpu_available()
    if torch_cuda and twincher_cuda:
        return "cuda"

    reason = None
    if torch_cuda and not _core.cuda_enabled:
        reason = (
            "twincher was built without CUDA support (no CUDA compiler was found when it was "
            "installed). To enable the GPU, make nvcc available and reinstall twincher: "
            "pip install --force-reinstall --no-deps --no-cache-dir twincher"
        )
    elif twincher_cuda and not torch_cuda:
        reason = "the installed PyTorch has no CUDA support; install a CUDA-enabled PyTorch"
    elif device == "cuda":
        reason = "no CUDA GPU is available"
    if reason is not None:
        warnings.warn(
            f"Computations will run on the CPU: {reason}.", RuntimeWarning, stacklevel=3
        )
    return "cpu"

def _get_backend(arr):
    return type(arr).__module__.split(".")[0]

def to_np(arr):
    backend = _get_backend(arr)
    if backend == "numpy":
        return arr.copy()
    elif backend == "cupy":
        import cupy as cp
        return cp.asnumpy(arr)
    elif backend == "torch":
        # return arr.detach().cpu().numpy()
        return arr.cpu().numpy()
    else:
        raise TypeError(f"Unsupported backend: {backend}")
    
def copy_to(dest, source):
    dest_backend = _get_backend(dest)
    source_backend = _get_backend(source)

    # NumPy destination
    if dest_backend == "numpy":
        if source_backend == "numpy":
            dest[...] = source

        elif source_backend == "cupy":
            import cupy as cp
            dest[...] = cp.asnumpy(source)

        elif source_backend == "torch":
            dest[...] = source.detach().cpu().numpy()

        else:
            raise TypeError(
                f"Cannot copy from {source_backend} to numpy"
            )

    # CuPy destination
    elif dest_backend == "cupy":
        import cupy as cp

        if source_backend == "numpy":
            dest[...] = cp.asarray(source)

        elif source_backend == "cupy":
            dest[...] = source

        elif source_backend == "torch":
            dest[...] = cp.asarray(source)

        else:
            raise TypeError(
                f"Cannot copy from {source_backend} to cupy"
            )

    # PyTorch destination
    elif dest_backend == "torch":
        import torch

        if source_backend == "numpy":
            dest.copy_(torch.as_tensor(source, device=dest.device))

        elif source_backend == "cupy":
            # Zero-copy view of the CuPy array, followed by
            # a GPU-to-GPU copy.
            dest.copy_(torch.as_tensor(source, device=dest.device))

        elif source_backend == "torch":
            dest.copy_(source)

        else:
            raise TypeError(
                f"Cannot copy from {source_backend} to torch"
            )

    else:
        raise TypeError(
            f"Unsupported destination backend: {dest_backend}"
        )

def estimate_J( # estimates Jacobian matrix using finite differences
        # input:
        stencil, # callable stencil(p, y) that defines y = y(p)
        p, # point at which J is estimated
        # output:
        y, # y(p)
        dy_dp, # estimated Jacobian
        # auxiliary:  
        d_eval=1e-8
    ):
    backend = _get_backend(y)
    if backend != "numpy" and d_eval == 1e-8:
        print("ERROR: estimate_J (d_eval=1e-8) is called with non-numpy arrays")
        return

    stencil(p, y)
    for i_p in range(p.shape[0]):
        p[i_p] += d_eval
        stencil(p, dy_dp[:, i_p])
        dy_dp[:, i_p] -= y[:]
        dy_dp[:, i_p] /= d_eval
        p[i_p] -= d_eval


def validate_parameters(params, requirements, raise_err = False):
    """
    Validate and fill a parameter dictionary.

    requirements:
        {
            "parameter_name": (min_value, max_value, default_value),
            ...
        }

    Rules:
    - params is None: returns None without checks
    - min_value=None: no lower-bound check
    - max_value=None: no upper-bound check
    - default_value=None: parameter is required
    - otherwise, missing parameters are silently added with their default

    Returns:
        if raise_err and errors are detected, raise ValueError(statement)
        otherwise
        a completed copy of params, "ok" if everything is valid, otherwise a concise error string.
    """
    if params is None:
        return None, ""

    params = dict(params) # copy: the caller's dictionary (possibly a default argument) is not modified
    errors = []

    # Check for unrecognized parameters
    unknown = set(params) - set(requirements)
    if unknown:
        errors.append(f"unrecognized: {', '.join(sorted(unknown))}")

    # Check required parameters, add defaults, and validate bounds
    for name, (min_value, max_value, default_value) in requirements.items():

        if name not in params:
            if default_value is None:
                errors.append(f"missing required: {name}")
            else:
                params[name] = default_value
            continue

        value = params[name]

        if min_value is not None and value < min_value:
            errors.append(f"{name} < {min_value}")

        if max_value is not None and value > max_value:
            errors.append(f"{name} > {max_value}")

    statement = "ok" if not errors else "; ".join(errors)

    if raise_err and errors:
        raise ValueError(statement)

    return params, statement
