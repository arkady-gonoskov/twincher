"""Interpolation error of training grids of a (reduced) chart, without training.

For each grid: y and dy/dp at the nodes (chart_training.grid_data, cached in --data-dir
with the names used by the training), interpolated multilinearly at --n random points
of the box |p|_inf <= --box and compared with the rendered values:
- tangent error per component k: |dy/dp_k(int) - dy/dp_k| / |dy/dp_k|;
- its maximum over k (the quantity of wrinkle_map.py);
- |y(int) - y| relative to the median |y(p + 0.1 e_k) - y(p)| as a scale.
Printed: median, 90 %, 99 % and max over the points.

Usage:
    python grid_tangent.py --dims 1 2 --grids 16x16 16x48 [--chart Y0] [--scale 1]
    python grid_tangent.py --grids 12x16x48x16 12x24x48x16   (all four components)
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from chart_training import data_tag, grid_data, stencil_of
from multires_interpolator import TorchMultilinearInterpolator
from wrinkle_map import render

HERE = Path(__file__).resolve().parent
NAMES = ["D", "e", "phi", "psi"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chart", default="Y0")
    parser.add_argument("--dims", type=int, nargs="+")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--grids", nargs="+", required=True)
    parser.add_argument("--box", type=float, default=0.86)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument(
        "--data-dir", type=Path, default=HERE / "output_y0_reduced/data"
    )
    args = parser.parse_args()
    cfg = {"obj": 26, "chart": args.chart, "margin": 1.0, "kernel": "flat",
           "probes": args.probes, "spectral": None, "dp": 0.025, "dims": args.dims,
           "scale": args.scale}  # fmt: skip
    st = stencil_of(cfg)
    names = [NAMES[k] for k in (args.dims or range(4))]
    g = torch.Generator().manual_seed(3)
    p = args.box * (2 * torch.rand((args.n, st.n_p), generator=g,
                                   dtype=torch.float64) - 1)  # fmt: skip
    y, d = render(st, p)
    ys = torch.stack([st.forward(p + 0.1 * torch.eye(st.n_p, dtype=p.dtype)[k])[0]
                      for k in range(st.n_p)])  # fmt: skip
    y_scale = (ys - y).norm(dim=2).median().item()
    q = lambda x: "/".join(f"{v:.2f}" for v in np.quantile(x, [0.5, 0.9, 0.99, 1.0]))
    print(f"chart {args.chart}, dims {args.dims}, scale {args.scale}, box {args.box}: "
          "quantiles 50/90/99/100 %")  # fmt: skip
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for grid in args.grids:
        n_g = [int(n) for n in grid.split("x")]
        if len(n_g) != st.n_p:
            raise SystemExit(f"grid {grid} needs {st.n_p} entries")
        yg, dg = grid_data(st, n_g, args.data_dir / f"grid_{data_tag(cfg)}_{grid}.pt")
        mli = TorchMultilinearInterpolator(st.n_p, n_g, st.device)
        yi, di = torch.empty_like(y), torch.empty_like(d)
        mli.interpolate(p.float().to(st.device), yi, yg.to(st.device))
        mli.interpolate(p.float().to(st.device), di, dg.to(st.device))
        tan = ((di - d).norm(dim=1) / d.norm(dim=1).clamp(min=1e-9)).cpu().numpy()
        dy = ((yi - y).norm(dim=1) / y_scale).cpu().numpy()
        per = ", ".join(f"{n} {q(tan[:, k])}" for k, n in enumerate(names))
        print(f"  {grid:14s} tangent max_k {q(tan.max(axis=1))} | {per} | "
              f"y {q(dy)}")  # fmt: skip
        del yg, dg
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
