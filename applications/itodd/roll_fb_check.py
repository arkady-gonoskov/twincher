"""Roll-aware Fourier-Bessel data (roll_fb.py) against rendered views and the full grid.

ChartY0, all four components, FB stencil with the near background and the depth
component (as y0_fb.py fbtrim_full). Errors of y (|y - y_ref|) and tangent errors
(|dy/dp_k - ref| / |ref| per component) against the stencil (rendering, central
differences):
A. roll only: distance, elevation, azimuth at nodes of the grid (no interpolation in
   them), roll uniform in [-0.86, 0.86]: the accuracy of the exact phase;
B. random points of |p|_inf <= 0.86: RollFB (grid 8 x 32 x 96 at roll 0) and the full
   grid (8, 32, 96, 24) of y0_fb.py (interpolated multilinearly in all four
   components; read memory-mapped).

Usage:
    python roll_fb_check.py [--n 2048]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from chart_training import data_tag
from multires_interpolator import TorchMultilinearInterpolator, grid_points
from roll_fb import RollFB
from train_chart import chart_stencil
from y0_fb import RUNS

HERE = Path(__file__).resolve().parent
NAMES = ["D", "e", "phi", "psi"]


def render(st, p, batch=64):
    y = torch.empty((len(p), st.n_y), device=st.device)
    d = torch.empty((len(p), st.n_y, st.n_p), device=st.device)
    for i in range(0, len(p), batch):
        st.compute(p[i : i + batch], y[i : i + batch], d[i : i + batch])
    return y.double().cpu(), d.double().cpu()


def report(name, y, d, y_ref, d_ref):
    q = lambda x: "/".join(f"{v:.3f}" for v in np.quantile(x, [0.5, 0.9, 0.99]))
    ey = (y - y_ref).norm(dim=1).numpy()
    tan = ((d - d_ref).norm(dim=1) / d_ref.norm(dim=1).clamp(min=1e-9)).numpy()
    print(f"  {name:22s} |dy| {q(ey)}  tangent "
          + ", ".join(f"{n} {q(tan[:, k])}" for k, n in enumerate(NAMES)), flush=True)  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=2048)
    args = parser.parse_args()
    cfg = RUNS["fbtrim_full"]
    st = chart_stencil("Y0", 26, dp=cfg["dp"], fb=cfg["fb"])
    data_dir = Path(cfg["data_dir"])
    rf = RollFB(st, [8, 32, 96], file=data_dir / f"rollfb_{data_tag(cfg)}_8x32x96.pt")
    g = torch.Generator().manual_seed(4)
    print("quantiles 50/90/99 %")
    # A: roll only
    nodes = torch.as_tensor(grid_points([8, 32, 96]))
    nodes = nodes[nodes.abs().amax(dim=1) <= 0.86]
    n = 512
    p = torch.empty((n, 4), dtype=torch.float64)
    p[:, :3] = nodes[torch.randint(len(nodes), (n,), generator=g)]
    p[:, 3] = 0.86 * (2 * torch.rand(n, generator=g, dtype=torch.float64) - 1)
    y_ref, d_ref = render(st, p)
    y = torch.empty((n, st.n_y), device=st.device)
    d = torch.empty((n, st.n_y, 4), device=st.device)
    rf.compute(p, y, d)
    print("A. roll only (other components at grid nodes):")
    report("RollFB", y.double().cpu(), d.double().cpu(), y_ref, d_ref)
    # B: random points
    p = 0.86 * (2 * torch.rand((args.n, 4), generator=g, dtype=torch.float64) - 1)
    y_ref, d_ref = render(st, p)
    y = torch.empty((args.n, st.n_y), device=st.device)
    d = torch.empty((args.n, st.n_y, 4), device=st.device)
    rf.compute(p, y, d)
    print("B. random points of |p| <= 0.86:")
    report("RollFB 8x32x96", y.double().cpu(), d.double().cpu(), y_ref, d_ref)
    n_g = cfg["grid"]
    full = torch.load(data_dir / f"grid_{data_tag(cfg)}_{'x'.join(map(str, n_g))}.pt",
                      mmap=True)  # fmt: skip
    mli = TorchMultilinearInterpolator(4, n_g, "cpu")
    yf = torch.empty((args.n, st.n_y))
    df = torch.empty((args.n, st.n_y, 4))
    mli.interpolate(p.float(), yf, full["y"])
    mli.interpolate(p.float(), df, full["dy_dp"])
    report(
        "full grid " + "x".join(map(str, n_g)), yf.double(), df.double(), y_ref, d_ref
    )


if __name__ == "__main__":
    main()
