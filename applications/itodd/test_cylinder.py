"""Stage 2: solve the inverse problem with a trained twincher for random synthetic views.

For n_cases random p_true in [-1, 1]^3, y_true = y(p_true) from DepthStencil; the hs
solver (damped Gauss-Newton in the representation space, start p = 0) uses the stencil
as forward model (y and finite-difference dy/dp at the current p). Prints per iteration
the distribution of the error of p (max over components, and per component in mm and
degrees) and of the residual |y - y_true|.

Usage:
    python test_cylinder.py [TWC_FILE] [--n-cases 1024] [--n-iter 10]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import twincher

from depth_stencil import DepthStencil
from itodd_data import model

MODEL = model(12)


def quantiles(x):
    q = np.quantile(x, [0.5, 0.9, 0.99, 1.0])
    return " ".join(f"{v:9.2e}" for v in q)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "twc",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "output_train_cylinder/final.twc",
    )
    parser.add_argument("--n-cases", type=int, default=1024)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--resolution", type=int, help="of the stencil (default: as in training)"
    )
    args = parser.parse_args()

    meta = twincher.load_metadata(str(args.twc))
    cfg = meta["stencil"]
    st = DepthStencil(
        MODEL,
        n_grid=cfg["n_grid"],
        resolution=args.resolution or cfg["resolution"],
        kernel=cfg.get("kernel", "max"),
        dp=cfg["dp"],
        dist_range=cfg["ranges"][0],
        elevation_range=cfg["ranges"][1],
        roll_range=cfg["ranges"][2],
    )
    print(f"{args.twc}: {json.dumps(cfg)}")
    solver = twincher.solver(str(args.twc), n_c=args.n_cases)

    g = torch.Generator().manual_seed(args.seed)
    p_true = (
        2 * torch.rand((args.n_cases, st.n_p), generator=g, dtype=torch.float64) - 1
    )
    y_true, _ = st.forward(p_true)
    v_true = st.view(p_true).numpy()
    solver.start_solution(y_true)
    print(
        "iter |   max_k |dp_k|: median, 90%, 99%, max   |   |y - y_true|: median, 90%, 99%, max"
    )
    for it in range(args.n_iter + 1):
        st.compute(solver.p, solver.y, solver.dy_dp)
        p = solver.p.double().cpu()
        dp = (p - p_true).abs().max(dim=1).values.numpy()
        res = (solver.y.double() - y_true.double()).norm(dim=1).cpu().numpy()
        print(f"{it:4d} | {quantiles(dp)} | {quantiles(res)}")
        if it < args.n_iter:
            solver.step(y=solver.y, dy_dp=solver.dy_dp)
    err = np.abs(st.view(p).numpy() - v_true)
    print("final errors, median / 99% / max:")
    print(f"  distance  {quantiles(err[:, 0])} mm")
    for k, name in [(1, "elevation"), (2, "roll")]:
        print(f"  {name:9s} {quantiles(np.degrees(err[:, k]))} deg")
    for tol in (1e-2, 1e-3):
        print(f"cases with max |dp_k| < {tol:g}: {np.mean(dp < tol):.3f}")
    worst = np.argsort(dp)[-5:][::-1]
    print("worst cases (p_true -> p):")
    for c in worst:
        print(f"  {np.round(p_true[c].numpy(), 3)} -> {np.round(p[c].numpy(), 3)}")


if __name__ == "__main__":
    main()
