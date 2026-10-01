# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

"""Errors of the compiled core must be raised as Python exceptions (not printed, no crashes)."""

import os
import pytest
import twincher

VALID = (8, 3, 4, 2, 2) # n_s, n_l, n_c, n_p, n_m

def shuttle_for(core_class, **kwargs):
    """An initialized Shuttle that uses the given class of the core."""
    data_type = "np" if core_class.__name__ == "ShuttleCPU" else "torch"
    return twincher.Shuttle(data_type=data_type, n_s=8, n_l=3, n_c=4, n_p=2, n_m=2, **kwargs)

@pytest.mark.parametrize("args", [
    (1, 3, 4, 2, 2),             # n_s < 2
    (8, 0, 4, 2, 2),             # n_l < 1
    (8, 3, 0, 2, 2),             # n_c < 1
    (8, 3, 4, -1, 2),            # n_p < 0
    (8, 3, 4, 2, -1),            # n_m < 0
    (8, 3, 4, 2, 2, 3),          # unknown tw_type
    (4096, 3, 1 << 20, 4, 4),    # tensors too large for int indexing
], ids=["n_s", "n_l", "n_c", "n_p", "n_m", "tw_type", "too-large"])
def test_invalid_arguments(core_class, args):
    with pytest.raises(ValueError):
        core_class(*args)

def test_operations_require_init(core_class, tmp_path):
    core = core_class(*VALID)
    operations = [
        core.forward_inference, core.backward_inference, core.forward_query, core.backward_query,
        core.compute_V, core.tw_update, core.set_rnd_t, core.set_tw_config,
        lambda: core.tw_save(str(tmp_path / "x.tw")),
    ]
    for operation in operations:
        with pytest.raises(RuntimeError, match="init"):
            operation()

def test_init_twice(core_class):
    SH = shuttle_for(core_class)
    with pytest.raises(RuntimeError, match="already initialized"):
        SH.allocate_init()

def test_invalid_twinch_config(core_class):
    SH = shuttle_for(core_class)
    with pytest.raises(ValueError, match="wl < ws < wu"):
        SH.core.set_tw_config(0.5, 0.1, 1.0)

def test_n_m_limits(core_class):
    SH = shuttle_for(core_class)
    SH.core.n_m = 1
    assert SH.core.n_m == 1
    for invalid in (3, -1):
        with pytest.raises(ValueError):
            SH.core.n_m = invalid

@pytest.mark.gpu
def test_invalid_kernel_launch():
    SH = twincher.Shuttle(data_type="torch", n_s=8)
    SH.core.n_threads = 4096 # more than the maximal number of threads per block
    with pytest.raises(RuntimeError, match="CUDA error"):
        SH.forward_inference()

def test_file_errors(core_class, tmp_path):
    good = str(tmp_path / "good.tw")
    twincher.Shuttle(data_type="np", n_s=8, n_l=3, n_c=4, n_p=2, n_m=2).tw_save(good)
    core_class(good, 4, 2, 2) # a valid file can be loaded

    with pytest.raises(RuntimeError, match="cannot open"):
        core_class(str(tmp_path / "missing.tw"), 4, 2, 2)

    garbage = tmp_path / "garbage.tw"
    garbage.write_bytes(os.urandom(200))
    with pytest.raises(RuntimeError, match="not a twincher file"):
        core_class(str(garbage), 4, 2, 2)

    truncated = tmp_path / "truncated.tw"
    truncated.write_bytes(open(good, "rb").read()[:60])
    with pytest.raises(RuntimeError, match="truncated"):
        core_class(str(truncated), 4, 2, 2)

    SH = shuttle_for(core_class)
    with pytest.raises(RuntimeError, match="cannot open"):
        SH.tw_save(str(tmp_path / "no_such_directory" / "x.tw"))
