# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

"""Unit tests of Learner, solver and Verifier with a tiny problem (a few steps on the CPU)."""

import json
import numpy as np
import pytest
import twincher

def make_learner(dir, **kwargs):
    kwargs.setdefault("plot_progress", False)
    learner = twincher.Learner(arch="hs", n_s=4, n_l=2, n_c=8, device="cpu", dir=dir, **kwargs)
    p = learner.get_p_request(n_p=1, n_grid=4)
    y = np.concatenate([p, p**2], axis=1)
    dy_dp = np.stack([np.ones_like(p), 2*p], axis=1)
    learner.set_data(y, dy_dp, stencil_name="parabola")
    return learner

def test_step(tmp_path):
    learner = make_learner(tmp_path)
    for _ in range(3):
        learner.step()
    assert learner.iter == 3
    assert np.isfinite(float(learner.loss))
    assert (tmp_path / "learner.json").exists()

def test_init(tmp_path):
    learner = make_learner(tmp_path)
    with pytest.raises(AttributeError):
        learner.SH
    learner.init() # gives access to the twincher before the first step
    assert learner.SH.n_s == 4
    with pytest.raises(RuntimeError, match="already been called"):
        learner.init()
    learner.step()
    assert learner.iter == 1

    learner = make_learner(tmp_path)
    learner._init() # former name, kept for compatibility
    assert learner.initiated

    learner = twincher.Learner(arch="hs", n_s=4, n_l=2, device="cpu", dir=tmp_path)
    with pytest.raises(RuntimeError, match="missing data"):
        learner.init()

def test_without_time_monitor(tmp_path):
    # loss terms call the time monitor unconditionally: a disabled monitor is used
    learner = make_learner(tmp_path, time_monitor=None)
    learner.step()
    assert learner.time_monitor.disable
    assert not (tmp_path / "learner_tm.log").exists()

def test_str(tmp_path):
    learner = make_learner(tmp_path)
    assert json.loads(str(learner))["arch"] == "hs"
    learner.step()
    config = json.loads(str(learner)) # after initialization: the configuration at initialization
    assert config["arch"] == "hs" and config["n_s"] == 4

def test_output_directory(tmp_path, monkeypatch):
    nested = tmp_path / "a" / "b"
    learner = make_learner(str(nested)) # str is accepted, parents are created
    learner.step()
    assert (nested / "learner.json").exists()

    monkeypatch.chdir(tmp_path)
    learner = make_learner(None) # default: twincher_output in the working directory
    learner.step()
    assert (tmp_path / "twincher_output" / "learner.json").exists()

def test_modify_params(tmp_path):
    learner = make_learner(tmp_path)
    assert [term.name for term in learner.loss_terms] == ["hs", "hs_noise"]
    learner.modify_params({"loss_terms": {"hs_noise": None, "hs": {"det_margin": 0.7}}})
    assert [term.name for term in learner.loss_terms] == ["hs"]
    assert learner.loss_terms[0].det_margin == 0.7
    with pytest.raises(ValueError, match="unknown parameter"):
        learner.modify_params({"no_such_parameter": 1})
    learner.step()
    with pytest.raises(RuntimeError):
        learner.modify_params({"lr": 1e-4}) # not possible after initialization

def test_invalid_device(tmp_path):
    with pytest.raises(ValueError, match="device"):
        twincher.Learner(arch="hs", n_s=4, n_l=2, device="gpu", dir=tmp_path)

def test_learning_curve_keeps_matplotlib_settings(tmp_path):
    import matplotlib
    settings = dict(matplotlib.rcParams)
    learner = make_learner(tmp_path, plot_progress=True, output_stride=1)
    learner.step()
    assert (tmp_path / "lc.png").stat().st_size > 0
    assert dict(matplotlib.rcParams) == settings

def test_solver_and_verifier(tmp_path):
    learner = make_learner(tmp_path)
    learner.step()
    file_name = str(tmp_path / "final.twc")
    learner.SH.save(file_name)
    assert twincher.load_metadata(file_name) == {
        "twincher_type": "hs", "stencil_name": "parabola", "n_p": 1, "n_y": 2,
    }

    # each solver gets its own damper with defaults derived from its own parameters
    solver_1 = twincher.solver(file_name, n_c=4, device="cpu", dp_cap=1.0)
    solver_2 = twincher.solver(file_name, n_c=4, device="cpu", dp_cap=0.1)
    assert solver_1.damper["dp_min"] == pytest.approx(1e-3)
    assert solver_2.damper["dp_min"] == pytest.approx(1e-4)
    assert solver_1.damper is not solver_2.damper
    assert twincher.solver(file_name, n_c=4, device="cpu", damper=None).damper is None

    class Stencil:
        n_p, n_y, name = 1, 2, "another stencil"
    with pytest.raises(ValueError, match="stencil mismatch"):
        twincher.Verifier(stencil=Stencil(), file_name=file_name, device="cpu")
    with pytest.raises(ValueError, match="either SH or file_name"):
        twincher.Verifier(stencil=Stencil())
