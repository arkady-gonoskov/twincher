"""Evaluate a twincher of a chart (train_chart.py) with the exact stencil.

1. det J on N_DET random points of [-1, 1]^n_p (y and dy/dp rendered, no interpolation),
   reported for the boxes |p|_inf <= R for each R in --boxes;
2. the solver (start p = 0, --n-iter iterations, the stencil as forward model) on
   N_SOLVE random synthetic views per box: success = max |p - p_true| < SOLVE_TOL.

Usage:
    python eval_chart.py TWC [TWC ...] [--boxes 0.8,0.86,0.9,1] [--n-iter 20]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import twincher

from train_chart import SOLVE_TOL, stencil_from_metadata

N_DET, N_SOLVE = 8192, 512


def dets_exact(twc, st, seed=11):
    g = torch.Generator().manual_seed(seed)
    p = 2 * torch.rand((N_DET, st.n_p), generator=g, dtype=torch.float64) - 1
    y = torch.empty((N_DET, st.n_y), device=st.device)
    dy_dp = torch.empty((N_DET, st.n_y, st.n_p), device=st.device)
    for i in range(0, N_DET, 512):
        st.compute(p[i : i + 512], y[i : i + 512], dy_dp[i : i + 512])
    sh = twincher.Shuttle()
    sh.load(str(twc), n_c=N_DET, n_p=st.n_p, n_m=st.n_p)
    sh.s.fill_(0)
    sh.s[: st.n_y].copy_(y.T)
    sh.Q.fill_(0)
    sh.Q[: st.n_y].copy_(dy_dp.permute(1, 2, 0))
    sh.forward_query()
    dets = torch.linalg.det(sh.Q[: st.n_p].permute(2, 1, 0)).cpu()
    return p, dets


def solve(twc, st, R, n_iter, seed=7, clamp=None, n=N_SOLVE):
    """max |p - p_true| of the solver (start p = 0) for n random views of |p|_inf <= R;
    with clamp, p is kept in |p|_inf <= clamp (the trained box) after each step. R and
    clamp: a number or one per component of p."""
    solver = twincher.solver(str(twc), n_c=n)
    g = torch.Generator().manual_seed(seed)
    R = torch.as_tensor(R, dtype=torch.float64)
    p_true = R * (2 * torch.rand((n, st.n_p), generator=g, dtype=torch.float64) - 1)
    if clamp is not None:
        clamp = torch.as_tensor(clamp, dtype=solver.p.dtype, device=solver.p.device)
    y_true, _ = st.forward(p_true)
    solver.start_solution(y_true)
    for _ in range(n_iter):
        st.compute(solver.p, solver.y, solver.dy_dp)
        solver.step(y=solver.y, dy_dp=solver.dy_dp)
        if clamp is not None:
            solver.p.clamp_(-clamp, clamp)
    return (solver.p.double().cpu() - p_true).abs().max(dim=1).values.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("twc", nargs="+", type=Path)
    parser.add_argument("--boxes", default="0.8,0.86,0.9,1")
    parser.add_argument("--n-iter", type=int, default=20)
    args = parser.parse_args()
    boxes = [float(b) for b in args.boxes.split(",")]
    for twc in args.twc:
        meta = twincher.load_metadata(str(twc))
        st = stencil_from_metadata(meta["stencil"])
        p, dets = dets_exact(twc, st)
        norm = p.abs().amax(dim=1)
        res = {"twc": str(twc), "boxes": {}}
        print(f"{twc}")
        for R in boxes:
            d = dets[norm <= R]
            err = solve(twc, st, R, args.n_iter)
            r = {"n_det": len(d), "det_min": d.min().item(),
                 "det_p01": torch.quantile(d, 0.01).item(),
                 "frac_det_neg": (d <= 0).float().mean().item(),
                 "frac_det_low": (d <= 0.5).float().mean().item(),
                 "solver_success": float((err < SOLVE_TOL).mean()),
                 "solver_err_p50": float(np.median(err)),
                 "solver_err_p99": float(np.quantile(err, 0.99)),
                 "solver_frac_err_above_0.1": float((err > 0.1).mean())}  # fmt: skip
            res["boxes"][str(R)] = r
            print(f"  |p| <= {R:4.2f}: det <= 0 {100 * r['frac_det_neg']:6.3f} %, "
                  f"det <= 0.5 {100 * r['frac_det_low']:6.3f} %, min det {r['det_min']:+8.2f}; "
                  f"solver success {100 * r['solver_success']:5.1f} %, error > 0.1: "
                  f"{100 * r['solver_frac_err_above_0.1']:4.1f} %, median error "
                  f"{r['solver_err_p50']:.1e}")  # fmt: skip
        out = twc.with_name(twc.stem + "_eval.json")
        out.write_text(json.dumps(res, indent=1))
        del st
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
