"""Why does the solver fail on a chart whose det J is positive almost everywhere?

For a twincher of train_chart.py / chart_training.py (TWC) and its pool (POOL):
1. overfitting: det J at the pool samples (training data) and at fresh random points
   (exact stencil), both in the box |p|_inf <= R;
2. the solver (start p = 0, 20 iterations) on random views in the box; for the failures
   (max |p - p_true| > 0.1): |r(p_stall) - r_true| (another preimage, or a stall with a
   residual), det J at p_stall, whether p_stall lies on the boundary of [-1, 1]^n_p
   (the solver clamps p), and the smallest det J on the segment p_true -> p_stall.

Usage:
    python solver_failures.py TWC POOL [--box 0.86]
"""

import argparse

import numpy as np
import torch
import twincher

from eval_chart import dets_exact
from train_chart import stencil_from_metadata


def r_and_det(sh, st, p, chunk):
    """r (n, n_p) and det J (n) of the twincher at p (n, n_p) with the exact stencil."""
    rs, ds = [], []
    for i in range(0, len(p), chunk):
        q = p[i : i + chunk]
        y = torch.empty((len(q), st.n_y), device=st.device)
        dy = torch.empty((len(q), st.n_y, st.n_p), device=st.device)
        st.compute(q, y, dy)
        sh.s.fill_(0)
        sh.s[: st.n_y, : len(q)].copy_(y.T)
        sh.Q.fill_(0)
        sh.Q[: st.n_y, :, : len(q)].copy_(dy.permute(1, 2, 0))
        sh.forward_query()
        rs.append(sh.s[: st.n_p, : len(q)].T.clone())
        ds.append(torch.linalg.det(sh.Q[: st.n_p, :, : len(q)].permute(2, 1, 0)))
    return torch.cat(rs).cpu(), torch.cat(ds).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("twc")
    parser.add_argument("pool")
    parser.add_argument("--box", type=float, default=0.86)
    parser.add_argument("--n-solve", type=int, default=1024)
    args = parser.parse_args()
    meta = twincher.load_metadata(args.twc)
    st = stencil_from_metadata(meta["stencil"])
    n_p, R = st.n_p, args.box
    chunk = 512
    sh = twincher.Shuttle()
    sh.load(args.twc, n_c=chunk, n_p=n_p, n_m=n_p)

    # 1. det J at the training samples vs at fresh random points
    pool = torch.load(args.pool)
    inside = pool["p"].abs().amax(dim=1) <= R
    idx = torch.nonzero(inside).squeeze(1)[:16384]
    dets_pool = []
    for i in range(0, len(idx), chunk):
        j = idx[i : i + chunk]
        sh.s.fill_(0)
        sh.s[: st.n_y, : len(j)].copy_(pool["y"][j].T.cuda())
        sh.Q.fill_(0)
        sh.Q[: st.n_y, :, : len(j)].copy_(pool["dy_dp"][j].permute(1, 2, 0).cuda())
        sh.forward_query()
        dets_pool.append(torch.linalg.det(sh.Q[:n_p, :, : len(j)].permute(2, 1, 0)))
    dets_pool = torch.cat(dets_pool).cpu()
    del pool
    p_rand, dets_rand = dets_exact(args.twc, st)
    dets_rand = dets_rand[p_rand.abs().amax(dim=1) <= R]
    for label, d in (
        ("pool samples (training)", dets_pool),
        ("random points", dets_rand),
    ):
        print(f"{label:24s}: n {len(d):6d}, det <= 0 {100 * (d <= 0).float().mean():.3f} %, "
              f"det <= 0.5 {100 * (d <= 0.5).float().mean():.3f} %, 1 % quantile "
              f"{torch.quantile(d, 0.01):+.3f}")  # fmt: skip

    # 2. failures of the solver
    solver = twincher.solver(args.twc, n_c=args.n_solve)
    g = torch.Generator().manual_seed(7)
    p_true = R * (
        2 * torch.rand((args.n_solve, n_p), generator=g, dtype=torch.float64) - 1
    )
    y_true, _ = st.forward(p_true)
    solver.start_solution(y_true)
    for _ in range(20):
        st.compute(solver.p, solver.y, solver.dy_dp)
        solver.step(y=solver.y, dy_dp=solver.dy_dp)
    p_end = solver.p.double().cpu()
    err = (p_end - p_true).abs().max(dim=1).values
    fail = err > 0.1
    print(f"solver: {int(fail.sum())} of {args.n_solve} failures (max |dp| > 0.1)")
    if not fail.any():
        return
    pf, pt = p_end[fail], p_true[fail]
    r_f, d_f = r_and_det(sh, st, pf, chunk)
    r_t, _ = r_and_det(sh, st, pt, chunk)
    dr = (r_f - r_t).norm(dim=1).numpy()
    on_border = (pf.abs().amax(dim=1) >= 1 - 1e-6).numpy()
    seg_min = []
    for k in range(len(pf)):
        s = torch.linspace(0, 1, 33, dtype=torch.float64)[:, None]
        _, d_seg = r_and_det(sh, st, pt[k] + s * (pf[k] - pt[k]), chunk)
        seg_min.append(d_seg.min().item())
    seg_min = np.array(seg_min)
    q = lambda x: " ".join(f"{v:.3g}" for v in np.quantile(x, [0.1, 0.5, 0.9]))
    print(f"  |r(p_stall) - r_true| (10/50/90 %): {q(dr)}")
    print(f"  det J at p_stall (10/50/90 %): {q(d_f.numpy())}; det <= 0 at "
          f"{100 * (d_f <= 0).float().mean():.1f} %")  # fmt: skip
    print(f"  p_stall on the boundary of [-1, 1]^{n_p}: {100 * on_border.mean():.1f} %")
    print(f"  min det J on the segment p_true -> p_stall: <= 0 for "
          f"{100 * (seg_min <= 0).mean():.1f} %; 10/50/90 %: {q(seg_min)}")  # fmt: skip
    print(f"  p_true of the failures, |p_k| median per component: "
          f"{np.round(pt.abs().median(dim=0).values.numpy(), 2).tolist()} (all: "
          f"{np.round(p_true.abs().median(dim=0).values.numpy(), 2).tolist()})")  # fmt: skip
    print(
        f"  p_stall - p_true, median |.| per component: "
        f"{np.round((pf - pt).abs().median(dim=0).values.numpy(), 3).tolist()}"
    )


if __name__ == "__main__":
    main()
