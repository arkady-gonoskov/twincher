# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

from . import registry
from .shuttle import load_metadata

from types import MethodType
from .tw_util import estimate_J, copy_to
def solve(self, stencil, y_true, n_iter, out, p_start = None, y_norm_res = None):
    self.start_solution(y_true=y_true, p_start=p_start)
    if hasattr(stencil, "compute"):
        for iter in range(n_iter):
            stencil.compute(self.p, self.y, self.dy_dp)
            self.step(y=self.y, dy_dp=self.dy_dp)

    elif hasattr(stencil, "__call__"):
        import numpy as np
        n_c = self.n_c
        n_p, n_y = stencil.n_p, stencil.n_y
        p = np.empty((n_c, n_p), dtype=np.float64)
        y = np.empty((n_c, n_y), dtype=np.float64)
        dy_dp = np.empty((n_c, n_y, n_p), dtype=np.float64)
        for iter in range(n_iter):
            copy_to(p, self.p)
            for i_c in range(n_c):
                estimate_J(
                    stencil=stencil,
                    p=p[i_c, :],
                    y=y[i_c, :],
                    dy_dp=dy_dp[i_c, :, :],
                    d_eval=1e-8
                )
            self.step(y=y, dy_dp=dy_dp)
    else:
        raise RuntimeError("Missing callable options in stencil.")
    copy_to(out, self.p)
    if y_norm_res is not None:
        copy_to(y_norm_res, ((self.y - self.y_true)**2).sum(dim=-1).sqrt())

def solver(*args, **kwargs):
    if len(args) > 0:
        file_name = args[0]
    elif "file_name" in kwargs:
        file_name = kwargs["file_name"]
    else:
        raise ValueError("Missing file_name")

    metadata = load_metadata(file_name)
    solver_type = metadata["twincher_type"]

    if solver_type not in registry.components("solvers"):
        raise RuntimeError(f"Unknown solver type {solver_type}")
    else:
        Solver = registry.components("solvers")[solver_type](*args, **kwargs)
        Solver.solve = MethodType(solve, Solver)
        return Solver
    