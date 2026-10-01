# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

"""Configuration of pytest for the tests of twincher."""

import pytest
import twincher

def pytest_report_header(config):
    core = twincher._core
    return (
        f"twincher {twincher.__version__}: CUDA backend {'built' if core.cuda_enabled else 'not built'}, "
        f"GPU {'available' if core.gpu_available() else 'not available'}"
    )

def pytest_collection_modifyitems(config, items):
    """Skip the tests marked with 'gpu' if twincher cannot use a GPU on this system."""
    if twincher._core.gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="twincher cannot use a CUDA GPU on this system")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)

@pytest.fixture(params=["torch", "cp"])
def gpu_data_type(request):
    """Data types of Shuttle on GPU: PyTorch tensors and CuPy arrays (if CuPy is installed)."""
    if request.param == "cp":
        pytest.importorskip("cupy")
    return request.param

@pytest.fixture(params=["np", "torch_cpu", pytest.param("torch", marks=pytest.mark.gpu), pytest.param("cp", marks=pytest.mark.gpu)])
def data_type(request):
    """All data types of Shuttle; those on GPU are skipped if no GPU (or no CuPy) is available."""
    if request.param == "cp":
        pytest.importorskip("cupy")
    return request.param

@pytest.fixture(params=["ShuttleCPU", pytest.param("ShuttleGPU", marks=pytest.mark.gpu)])
def core_class(request):
    """Classes of the compiled core: ShuttleCPU, and ShuttleGPU if a GPU is available."""
    return getattr(twincher._core, request.param)
