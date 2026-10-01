# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

from pathlib import Path
import tempfile
import numpy as np
from .shuttle import load_metadata
from .tw_util import estimate_J, copy_to, to_np, output_dir

from .solver import solver

class Verifier:
    def __init__(self, 
        stencil,
        SH = None,
        file_name = None, 
        n_tests = 10*1024,
        n_c = 10*1024,
        n_iter = 32,
        y_L2_tol = 1e-4,
        device = "auto",
        rng_seed = 2,
        dir = None, # directory for outputs (default: directory of file_name, or ./twincher_output)
        time_monitor = None,
        i_detail = None,
    ):
        self.stencil = stencil

        if (SH is None) == (file_name is None):
            raise ValueError("Verifier: provide either SH or file_name")

        if dir is None and file_name is not None:
            dir = Path(file_name).resolve().parent
        self.dir = output_dir(dir)

        self.file_name = file_name

        with tempfile.TemporaryDirectory() as tmp_dir:
            if SH is not None:
                file_name = str(Path(tmp_dir) / "shuttle.twc")
                SH.save(file_name)

            self.tw_meta = load_metadata(file_name)

            twincher_stencil = (self.tw_meta["n_p"], self.tw_meta["n_y"], self.tw_meta["stencil_name"])
            if twincher_stencil != (stencil.n_p, stencil.n_y, stencil.name):
                raise ValueError(
                    "stencil mismatch: the twincher was trained for (n_p, n_y, name) = "
                    f"{twincher_stencil}, but the given stencil has {(stencil.n_p, stencil.n_y, stencil.name)}"
                )

            self.CS = solver(
                file_name = file_name,
                n_c = n_c,
                device = device,
                dp_cap = 1.0,
                epsilon = 1e-8,
                damper = {"down_factor": 0.5, "up_factor": 1.1},
            )
        if time_monitor == "auto":
            from .time_monitor import TimeMonitor
            self.time_monitor = TimeMonitor(str(self.dir) + "/tw_verifier_tm.log")
        else:
            self.time_monitor = time_monitor
        if self.time_monitor is not None:
            self.CS.SH.enable_time_monitor(self.time_monitor)

        self.n_tests = n_tests
        self.n_iter = n_iter
        self.y_L2_tol = y_L2_tol
        self.rng_seed = rng_seed

        n_p = stencil.n_p
        n_y = stencil.n_y
        self.p_true = np.empty((n_c, n_p), dtype=np.float64)
        self.p_start = np.empty((n_c, n_p), dtype=np.float64)
        self.p = np.empty((n_c, n_p), dtype=np.float64)

        self.y_true = np.empty((n_c, n_y), dtype=np.float64)
        self.y = np.empty((n_c, n_y), dtype=np.float64)
        self.dy_dp = np.empty((n_c, n_y, n_p), dtype=np.float64)
        self.y_L2 = np.empty((n_c), dtype=np.float64)

        self.i_detail = i_detail

    def verify_solution(self, SH_a = None, verbose = False, print_fails = False):

        if SH_a is not None:
            copy_to(self.CS.SH.a, SH_a)
            self.CS.SH.tw_update()

        if self.time_monitor is not None: self.time_monitor.start("tw_verifier::verify")
        rng = np.random.default_rng(self.rng_seed)
        n_c = self.CS.SH.n_c
        n_batches = (self.n_tests - 1) // n_c + 1
        fail_count = 0
        for i_e in range(n_batches):

            self.p_true[...] = rng.uniform(-1, 1, self.p_true.shape)
            self.p_start[...] = rng.uniform(-1, 1, self.p_start.shape)

            for i_c in range(n_c):
                self.stencil(p=self.p_true[i_c, :], y=self.y_true[i_c, :])

            self.CS.start_solution(y_true=self.y_true, p_start=self.p_start)

            for iter in range(self.n_iter):
                if verbose: print(f"\rverifier run test: {100*(i_e*self.n_iter + iter)/(n_batches*self.n_iter):5.2f}%", end="")
                copy_to(self.p, self.CS.p)
                if not hasattr(self.stencil, "compute"):
                    if self.time_monitor is not None: self.time_monitor.start("tw_verifier::estimate_J", n_c)
                    for i_c in range(n_c):
                        estimate_J(
                            stencil=self.stencil,
                            p=self.p[i_c, :],
                            y=self.y[i_c, :],
                            dy_dp=self.dy_dp[i_c, :, :],
                            d_eval=1e-8
                        )
                    if self.time_monitor is not None: self.time_monitor.stop("tw_verifier::estimate_J")
                    self.CS.step(y=self.y, dy_dp=self.dy_dp, i_detail=self.i_detail)
                else:
                    if self.time_monitor is not None: self.time_monitor.start("tw_verifier::estimate_J", n_c)
                    self.stencil.compute(self.CS.p, self.CS.y, self.CS.dy_dp)
                    if self.time_monitor is not None: self.time_monitor.stop("tw_verifier::estimate_J")
                    copy_to(self.y, self.CS.y)
                    self.CS.step(y=self.CS.y, dy_dp=self.CS.dy_dp, i_detail=self.i_detail)
                
            self.y_L2[:] = ((self.y_true - self.y)**2).sum(-1)
            for i_c in range(n_c):
                if self.y_L2[i_c] > self.y_L2_tol:
                    fail_count += 1
                    if print_fails: print(
                        "\r",
                        f"failed {i_e}:{i_c}:",
                        "p_true=", self.p_true[i_c, :], 
                        "p_sol=", to_np(self.CS.p[i_c, :]),
                        "diff=", f"{np.linalg.norm(self.p_true[i_c, :] - to_np(self.CS.p[i_c, :])):5.2e}"
                    )
        if verbose: print(f"\rverifier: failed: {fail_count} out of {n_c*n_batches}, {100*fail_count/(n_c*n_batches):8.5f}%")
        if self.time_monitor is not None: self.time_monitor.stop("tw_verifier::verify")

        return fail_count/(n_c*n_batches)