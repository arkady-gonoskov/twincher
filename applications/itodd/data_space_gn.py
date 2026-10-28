"""Gauss-Newton in the space of y against Gauss-Newton in the learned representation r.

Ablation for the paper, on the charts of README Part III (the configurations of Section
22.3): random synthetic views p_true of the trained box |p|_inf <= box (rendered, exact
data), observations y_true = y(p_true), and the inverse problem y(p) = y_true solved with
the same forward model (the stencil of the chart), the same start and the same clamp of p
to the box, in two ways:
- "r": in the representation of the twincher (twincher.solver, as in eval_chart.py);
- "y": damped Gauss-Newton in the space of y, (J^T J + lam I) dp = J^T (y_true - y(p)),
  J = dy/dp, with the step clipped to |dp| <= dp_max: the baseline of the harmonic
  entanglers in the paper (lam 1e-3, dp_max 0.1); dp_max = inf: no clipping.
Starts: p = 0 (as in LidarFit) and random p_in in the box (as for the complexity
C = -ln p_success of the paper).

Per chart, solver and start: the fraction of views with max_k |p_k - p_true,k| below
1e-2 and 1e-3 after each step.

Usage:
    python data_space_gn.py [--n 512] [--n-iter-r 40] [--n-iter-y 100]
                            [--out output_data_space_gn/results.json]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import twincher

from lidar_fit import LidarFit

HERE = Path(__file__).resolve().parent
CONFIGS = {12: "lidarfit_obj12_m3_tent_mask.json", 26: "lidarfit_obj26_m3_mask.json"}
TOLS = (1e-2, 1e-3)
BATCH = 64  # views per call of the stencil (1 + 2 n_p renderings each)


def compute(st, p):
    """y (n, n_y) and dy/dp (n, n_y, n_p) of the stencil, float64 on its device."""
    n = len(p)
    y = torch.empty((n, st.n_y), dtype=torch.float64, device=st.device)
    dy_dp = torch.empty((n, st.n_y, st.n_p), dtype=torch.float64, device=st.device)
    for i in range(0, n, BATCH):
        st.compute(p[i : i + BATCH], y[i : i + BATCH], dy_dp[i : i + BATCH])
    return y, dy_dp


def success(p, p_true):
    err = (p.double().cpu() - p_true).abs().amax(dim=1)
    return [float((err < tol).double().mean()) for tol in TOLS]


def solve_r(twc, st, y_true, p_true, p_start, n_iter, box):
    solver = twincher.solver(str(twc), n_c=len(y_true))
    solver.start_solution(y_true.to(solver.y.dtype), p_start=p_start.to(solver.p))
    history = [success(solver.p, p_true)]
    for _ in range(n_iter):
        y, dy_dp = compute(st, solver.p)
        solver.step(y=y.to(solver.y.dtype), dy_dp=dy_dp.to(solver.dy_dp.dtype))
        solver.p.clamp_(-box, box)
        history.append(success(solver.p, p_true))
    return history


def solve_y(st, y_true, p_true, p_start, n_iter, box, lam=1e-3, dp_max=0.1):
    p = p_start.clone().to(st.device, torch.float64)
    eye = torch.eye(st.n_p, dtype=torch.float64, device=st.device)
    history = [success(p, p_true)]
    for _ in range(n_iter):
        y, J = compute(st, p)
        Jt = J.transpose(1, 2)
        dp = torch.linalg.solve(Jt @ J + lam * eye, Jt @ (y_true - y)[..., None])[..., 0]
        if math.isfinite(dp_max):
            norm = torch.linalg.vector_norm(dp, dim=1, keepdim=True)
            dp = dp * torch.clamp(dp_max / norm.clamp_min(1e-30), max=1.0)
        p = (p + dp).clamp(-box, box)
        history.append(success(p, p_true))
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--n-iter-r", type=int, default=40)
    parser.add_argument("--n-iter-y", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path,
                        default=HERE / "output_data_space_gn/results.json")  # fmt: skip
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for obj, cfg in CONFIGS.items():
        lf = LidarFit(HERE / cfg)
        box = lf.clamp
        for chart in lf.charts:
            key = f"{obj}/{chart.name}"
            if key in results:
                print(f"{key}: done")
                continue
            st = chart.stencil
            g = torch.Generator().manual_seed(args.seed)
            p_true = box * (2 * torch.rand((args.n, st.n_p), generator=g,
                                           dtype=torch.float64) - 1)  # fmt: skip
            p_rand = box * (2 * torch.rand((args.n, st.n_p), generator=g,
                                           dtype=torch.float64) - 1)  # fmt: skip
            y_true, _ = compute(st, p_true)
            out = {"box": box, "n": args.n}
            for start, p_start in (("zero", torch.zeros_like(p_true)),
                                   ("random", p_rand)):  # fmt: skip
                out[f"r_{start}"] = solve_r(chart.twc, st, y_true, p_true, p_start,
                                            args.n_iter_r, box)  # fmt: skip
                out[f"y_{start}"] = solve_y(st, y_true, p_true, p_start,
                                            args.n_iter_y, box)  # fmt: skip
                out[f"y_noclip_{start}"] = solve_y(st, y_true, p_true, p_start,
                                                   args.n_iter_y, box, dp_max=math.inf)  # fmt: skip
                summary = {k: v[-1][0] for k, v in out.items() if k.endswith(start)}
                print(f"{key}, start {start}: success (< 1e-2) {summary}", flush=True)
            results[key] = out
            args.out.write_text(json.dumps(results, indent=1))
            torch.cuda.empty_cache()
        del lf
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
