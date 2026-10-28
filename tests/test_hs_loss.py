# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

"""Unit tests of HSLoss options: ramp modes, trimmer over the whole domain, momentum reset."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import twincher
from twincher.hs_loss import MomentumReset


def make_learner(dir, hs_params, sign=1.0, **kwargs):
    # n_p = 2, n_y = 3: y = (sign * p0, p1, p0 * p1)
    learner = twincher.Learner(
        arch="hs",
        n_s=4,
        n_l=3,
        n_c=64,
        device="cpu",
        dir=dir,
        plot_progress=False,
        **kwargs,
    )
    p = learner.get_p_request(n_p=2, n_grid=5)
    p0, p1, zero, one = p[:, 0], p[:, 1], np.zeros(len(p)), np.ones(len(p))
    y = np.stack([sign * p0, p1, p0 * p1], axis=1)
    dy_dp = np.stack([[sign * one, zero], [zero, one], [p1, p0]])
    learner.set_data(y, dy_dp.transpose(2, 0, 1))
    learner.modify_params({"loss_terms": {"hs_noise": None, "hs": hs_params}})
    return learner


def train(learner, n_steps):
    for _ in range(n_steps):
        learner.step()
    return learner.SH.a.clone()


def test_ramp_modes(tmp_path):
    # without a boost at the border, the ramp of version 0.1.0 ("sum") changes nothing
    # (det_margin 1.5: the determinant loss acts on all cases)
    a = {}
    for mode in ("max", "sum"):
        for boost in (1.0, 2.0):
            params = {"ramp_mode": mode, "ramp_boost": boost, "det_margin": 1.5}
            a[mode, boost] = train(make_learner(tmp_path, params), 5)
    assert torch.equal(a["max", 1.0], a["sum", 1.0])
    assert not torch.equal(a["max", 2.0], a["sum", 2.0])
    assert not torch.equal(a["max", 1.0], a["max", 2.0])
    with pytest.raises(ValueError, match="unknown ramp_mode"):
        make_learner(tmp_path, {"ramp_mode": "linear"}).init()


def test_trimmer_domain_gradient(tmp_path):
    # the gradient of both passes against finite differences of the loss
    learner = make_learner(
        tmp_path, {"trimmer_threshold": 0.6, "trimmer_domain_amp": 2.0}
    )
    train(learner, 20)  # away from the identity
    hs = learner.loss_terms[0]
    learner.SH.core.sga = 0.0  # (the s-gate regularization is not part of hs.loss)
    rng_state = learner.rng.get_state()

    def loss_and_grad(a):
        learner.SH.a.copy_(a)
        learner.SH.tw_update()
        learner.rng.set_state(rng_state)  # the same cases over the whole domain
        hs.compute_grad()
        return float(hs.loss), learner.SH.g.clone()

    a0 = learner.SH.a.clone()
    _, g = loss_and_grad(a0)
    assert float(hs.domain_loss) > 0 and float(hs.trimmer_loss) > 0
    e = torch.as_tensor(np.random.default_rng(1).normal(size=a0.shape))
    h = 1e-6
    fd = (loss_and_grad(a0 + h * e)[0] - loss_and_grad(a0 - h * e)[0]) / (2 * h)
    assert fd == pytest.approx(float(g @ e), rel=1e-5)


def test_momentum_reset():
    param = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    learner = SimpleNamespace(
        SH=SimpleNamespace(param=param, a=param.detach()),
        optimizer=torch.optim.Adam([param], lr=0.1),
        lr=0.1,
        lr_dynamic=0.1,
    )
    reset = MomentumReset(neg_frac=0.1)

    def step(n_neg):
        reset.before(learner)
        lr = learner.lr  # the learning rate during this step
        reset.judge(torch.tensor([-1.0] * n_neg + [1.0] * (20 - n_neg)))
        param.grad = torch.ones(3, dtype=torch.float64)
        learner.optimizer.param_groups[0]["lr"] = learner.lr_dynamic  # (as the damper)
        learner.optimizer.step()
        return lr

    a0 = learner.SH.a.clone()
    assert step(2) == 0.1  # accepted (2 of 20 det < 0)
    assert step(3) == 0.1  # rejected: back to the state before the update that led to
    state = learner.optimizer.state[param]  # this batch, first moment of Adam zeroed
    assert torch.equal(learner.SH.a, a0) and reset.n == 1
    assert not state["exp_avg"].any() and state["exp_avg_sq"].all()
    assert reset.boost == 0.5 and learner.lr == 0.1  # nominal again after the step
    assert step(0) == pytest.approx(0.05)  # learning rate x 0.5, then x 1.1 per step
    assert learner.lr_dynamic == pytest.approx(0.05)
    assert not torch.equal(learner.SH.a, a0)
    reset.paused = True
    assert step(20) == pytest.approx(0.055) and reset.n == 1  # not judged
    reset.load_state({"boost": 0.25, "n": 7, "ema": 1.0})
    assert reset.state() == {"boost": 0.25, "n": 7, "ema": 1.0} and not reset.good


def test_momentum_reset_in_learner(tmp_path):
    # y0 = -p0: det(dr/dp) = -1 for the initial (identity) twincher, all batches rejected
    learner = make_learner(
        tmp_path, {"momentum_reset": {"neg_frac": 0.0}}, sign=-1.0, output_stride=1
    )
    learner.init()
    a0 = learner.SH.a.clone()
    train(learner, 3)
    hs = learner.loss_terms[0]
    assert hs.reset.n == 3 and torch.equal(learner.SH.a, a0)
    assert learner.lr == 1e-3 and "resets:3" in hs.state
