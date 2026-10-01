# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

"""
End-to-end test based on the spiral example of the tutorial.

A twincher of the hs architecture is trained to represent the position along a spiral, and
is then used to solve the inverse problem. The setting is minimal: Learner with the hs loss
only (no noise-tolerance loss, no outputs) and the hs solver.
"""

import math
import numpy as np
import pytest
import twincher

# Define stencil
class Spiral:
    def __init__(self, n_turns = 1.0):
        self.n_p, self.n_y = 1, 2
        self.n_turns = n_turns
        self.name = f"spiral_{n_turns:3.1f}"
    def __call__(self, p, y):
        r = 1/(0.5*p[0] + 1.5)
        alpha = 0.25*math.pi + self.n_turns*math.pi*(p[0] + 1)
        y[0] = r*math.cos(alpha)
        y[1] = r*math.sin(alpha)
    def batch_call(self, p, y):
        r = (0.5*p[:, 0] + 1.5)**(-1)
        pi = 3.14159265358979
        alpha = self.n_turns*pi*(p[:, 0] + 1) + 0.25*pi
        backend = type(p).__module__.split(".")[0]
        if backend == "numpy":
            y[:, 0] = r*np.cos(alpha)
            y[:, 1] = r*np.sin(alpha)
        elif backend == "torch":
            import torch
            y[:, 0] = r*torch.cos(alpha)
            y[:, 1] = r*torch.sin(alpha)
    def compute(self, p, y, dy_dp, dp_eval = 1e-4):
        self.batch_call(p, y)
        for i_p in range(self.n_p):
            p[:, i_p] += dp_eval
            self.batch_call(p, dy_dp[:, :, i_p])
            dy_dp[:, :, i_p] -= y[:]
            dy_dp[:, :, i_p] /= dp_eval
            p[:, i_p] -= dp_eval

@pytest.mark.e2e
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_spiral(device, tmp_path):
    stencil = Spiral(n_turns=1.3)

    # Create learner
    learner = twincher.Learner(
        arch="hs",
        n_s=16,
        n_l=64,
        dir=tmp_path,
        lr=0.003,
        n_c=256,
        device=device,
        plot_progress=False,
    )

    # disable noise-tolerance loss
    learner.modify_params(params={"loss_terms": {"hs_noise": None,}})

    # Generate data and initiate learner
    learner.generate_data(stencil=stencil, n_grid=128)

    # Run optimization
    n_iter = 5001
    for iter in range(n_iter):
        learner.step()
    print(f"{n_iter} iterations: {learner.state}")
    learner.SH.save(str(tmp_path / "final.twc"))

    # Test solutions
    solver = twincher.solver(file_name=str(tmp_path / "final.twc"), n_c=5*1024, device=device)
    rng = np.random.default_rng(0)
    p_true = rng.uniform(-1, 1, size=(solver.n_c, 1))
    y_true = np.empty((solver.n_c, stencil.n_y), dtype=np.float64)
    stencil.batch_call(p_true, y_true)
    solver.start_solution(y_true)
    for _ in range(6):
        stencil.compute(solver.p, solver.y, solver.dy_dp)
        solver.step(y=solver.y, dy_dp=solver.dy_dp)
    p_sol = solver.p.detach().cpu().numpy()
    diffs = np.linalg.norm(p_true - p_sol, axis=1)
    print(f"Max difference in p: {np.max(diffs)}")
    assert np.max(diffs) < 1e-2
