# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

"""Tests of the package as a whole: version, imports, registry and utilities."""

import subprocess
import sys
import warnings
import pytest
import twincher
from twincher import registry, tw_util

def test_version():
    assert isinstance(twincher.__version__, str)
    assert twincher.__version__[0].isdigit()

def test_info(capfd):
    twincher.info()
    text = capfd.readouterr().out
    assert "Twincher" in text and "GNU Affero General Public License" in text
    assert "CUDA backend: " in text and "GPU: " in text

def test_public_names():
    for name in twincher.__all__:
        assert hasattr(twincher, name), name
    assert "Learner" in dir(twincher)
    with pytest.raises(AttributeError):
        twincher.no_such_name

def test_import_does_not_load_torch_and_matplotlib():
    # in a new process, because other tests import PyTorch
    code = (
        "import sys, twincher\n"
        "SH = twincher.Shuttle(data_type='np', n_s=4, n_l=2)\n"
        "SH.forward_inference()\n"
        "loaded = [name for name in ('torch', 'matplotlib') if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

# ------------------------------------ registry ------------------------------------

def test_builtin_components_are_registered():
    assert "hs" in registry.components("arch_defaults")
    assert {"hs", "hs_noise"} <= set(registry.components("loss_terms"))
    assert "hs" in registry.components("solvers")
    assert "hs_monitor_p2" in registry.components("monitors")

def test_register():
    with pytest.raises(RuntimeError, match="already been registered"):
        twincher.register("loss_terms", "hs", object)
    with pytest.raises(RuntimeError, match="unknown component category"):
        twincher.register("no_such_category", "name", object)

def test_unknown_components():
    with pytest.raises(ValueError):
        twincher.Learner(arch="no_such_arch", n_s=4, n_l=2)
    with pytest.raises(RuntimeError):
        twincher.monitor(name="no_such_monitor")

# ------------------------------------ utilities ------------------------------------

def test_validate_parameters():
    requirements = {"factor": (0.0, 1.0, 0.5), "required": (0, None, None)}
    given = {"required": 3}
    params, statement = tw_util.validate_parameters(given, requirements)
    assert params == {"required": 3, "factor": 0.5} and statement == "ok"
    assert given == {"required": 3} # the given dictionary is not modified
    assert tw_util.validate_parameters(None, requirements) == (None, "")
    for invalid in ({}, {"required": 3, "factor": 2.0}, {"required": 3, "unknown": 1}):
        with pytest.raises(ValueError):
            tw_util.validate_parameters(invalid, requirements, raise_err=True)

def test_select_device():
    assert tw_util.select_device("cpu") == "cpu"
    assert tw_util.select_device("auto") in ("cpu", "cuda")
    with pytest.raises(ValueError):
        tw_util.select_device("gpu")

@pytest.mark.parametrize("device, torch_cuda, cuda_enabled, gpu_available, selected, warning", [
    ("auto", True, True, True, "cuda", None),
    ("auto", True, False, False, "cpu", "twincher was built without CUDA support"),
    ("auto", False, True, True, "cpu", "PyTorch has no CUDA support"),
    ("cuda", False, True, False, "cpu", "no CUDA GPU is available"),
    ("auto", False, True, False, "cpu", None), # no GPU at all: the CPU is used without a warning
], ids=["gpu-usable", "twincher-without-cuda", "pytorch-without-cuda", "no-gpu-requested", "no-gpu-auto"])
def test_select_device_cases(monkeypatch, device, torch_cuda, cuda_enabled, gpu_available, selected, warning):
    # simulates the combinations of GPU support in PyTorch and in twincher
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: torch_cuda)
    monkeypatch.setattr(twincher._core, "cuda_enabled", cuda_enabled)
    monkeypatch.setattr(twincher._core, "gpu_available", lambda: gpu_available)
    if warning is None:
        with warnings.catch_warnings():
            warnings.simplefilter("error") # any warning makes the test fail
            assert tw_util.select_device(device) == selected
    else:
        with pytest.warns(RuntimeWarning, match=warning):
            assert tw_util.select_device(device) == selected

def test_output_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert tw_util.output_dir() == tmp_path / "twincher_output"
    assert (tmp_path / "twincher_output").is_dir()
    nested = tw_util.output_dir(str(tmp_path / "a" / "b")) # str is accepted, parents are created
    assert nested == tmp_path / "a" / "b" and nested.is_dir()
