"""Interpolation error of grid data (chart_training.py) against the exact stencil.

At n random points of [-1, 1]^4: |y_interpolated - y| (norm over the components), the
same for a step of 0.1 in all components of p as a scale, and the relative error of each
dy/dp_k.

Usage:
    python interp_check.py RUN_DIR [--n 2048]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chart_training import data_tag, stencil_of
from multires_interpolator import TorchMultilinearInterpolator


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n", type=int, default=2048)
    args = parser.parse_args()
    cfg = json.loads((args.run_dir / "config.json").read_text())
    st = stencil_of(cfg)
    n_g = "x".join(map(str, cfg["grid"]))
    data = torch.load(Path(cfg["data_dir"]) / f"grid_{data_tag(cfg)}_{n_g}.pt")
    y_g, d_g = data["y"].cuda(), data["dy_dp"].cuda()
    del data
    mli = TorchMultilinearInterpolator(st.n_p, cfg["grid"], "cuda")
    g = torch.Generator().manual_seed(3)
    p = 2 * torch.rand((args.n, st.n_p), generator=g, dtype=torch.float64) - 1
    y = torch.empty((args.n, st.n_y), device="cuda")
    d = torch.empty((args.n, st.n_y, st.n_p), device="cuda")
    for i in range(0, args.n, 256):
        st.compute(p[i : i + 256], y[i : i + 256], d[i : i + 256])
    yi, di = torch.empty_like(y), torch.empty_like(d)
    mli.interpolate(p.float().cuda(), yi, y_g)
    mli.interpolate(p.float().cuda(), di, d_g)
    step = (
        p + 0.1 * torch.sign(torch.randn(p.shape, generator=g, dtype=torch.float64))
    ).clamp(-1, 1)
    y_step, _ = st.forward(step)
    q = lambda x: " ".join(
        f"{v:.3f}" for v in np.quantile(x.cpu().numpy(), [0.5, 0.9, 1.0])
    )
    print(f"{args.run_dir.name}: grid {cfg['grid']}, n_y {st.n_y}")
    print(f"  |y_int - y| (median, 90 %, max): {q((yi - y).norm(dim=1))}")
    print(f"  scale |y(p + 0.1 steps) - y(p)|: {q((y_step - y).norm(dim=1))}")
    rel = (di - d).norm(dim=1) / d.norm(dim=1).clamp(min=1e-9)
    names = ["D", "elevation", "azimuth", "roll"]
    print("  dy/dp relative error (median / 90 %): " + ", ".join(
        f"{names[k]} {rel[:, k].median():.3f}/{rel[:, k].quantile(0.9):.3f}" for k in range(st.n_p)))  # fmt: skip


if __name__ == "__main__":
    main()
