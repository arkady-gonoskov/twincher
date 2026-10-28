"""Properties of y(p) for stencil variants of a chart, before any training.

On the (elevation, azimuth) plane of the chart (distance and roll at the centre of
their ranges), |p| <= 0.86:
- speed: medians of |dy/de|, |dy/dphi|;
- conditioning: quantiles 1 % / 10 % of sigma_min / sigma_max of dy/dp, also with the
  roll (dy/dp for e, phi, psi);
- near-cusps: chord/arc of y along phi over 0.063 (6 steps of a 192-point grid), 0.1 %
  quantile and minimum (1 = straight, small = the curve turns back);
- interpolation: tangent error of a 32 x 96 grid (max over e, phi), 90 % / 99 %;
- finite-difference noise: |dy/dp(dp) - dy/dp(dp / 2)| / |dy/dp|, median / 90 %.

Usage:
    python stencil_metrics.py [--chart Y0]
"""

import argparse

import numpy as np
import torch

from multires_interpolator import TorchMultilinearInterpolator, grid_points
from train_chart import chart_stencil
from wrinkle_map import render

VARIANTS = {  # name: stencil options
    "flat8": {"kernel": "flat", "n_grid": 8},
    "fb64 C2": {"fb": {"n": 64, "scale": 2.0}},
    "fb64 C2 nocut": {"fb": {"n": 64, "scale": 2.0, "gamma": 0.0}},
    "fb64 C12 bg20": {"fb": {"n": 64, "scale": 12.0, "background": 20.0}},
    "fb64 C8 bg40": {"fb": {"n": 64, "scale": 8.0, "background": 40.0}},
    "fb32 C2": {"fb": {"n": 32, "scale": 2.0}},
}
BOX = 0.86


def q(x, qs):
    return "/".join(f"{v:.2f}" for v in np.quantile(x, qs))


def metrics(chart, opts):
    st = chart_stencil(chart, 26, dp=0.025, dims=[1, 2], **opts)
    g = torch.Generator().manual_seed(3)
    p = BOX * (2 * torch.rand((4096, 2), generator=g, dtype=torch.float64) - 1)
    y, d = render(st, p)
    d = d.double()
    speed = d.norm(dim=1).cpu().numpy()
    sv = torch.linalg.svdvals(d).cpu().numpy()
    out = {"speed e/phi": f"{np.median(speed[:, 0]):.2f}/{np.median(speed[:, 1]):.2f}",
           "cond 1/10 %": q(sv[:, 1] / sv[:, 0], [0.01, 0.1])}  # fmt: skip
    st3 = chart_stencil(chart, 26, dp=0.025, dims=[1, 2, 3], **opts)
    p3 = BOX * (2 * torch.rand((2048, 3), generator=g, dtype=torch.float64) - 1)
    _, d3 = render(st3, p3)
    sv3 = torch.linalg.svdvals(d3.double()).cpu().numpy()
    out["cond3 1/10 %"] = q(sv3[:, 2] / sv3[:, 0], [0.01, 0.1])
    # chord / arc along phi
    G, k = 192, 6
    x = np.linspace(-BOX, BOX, G)
    pg = torch.as_tensor(np.stack(np.meshgrid(x, x, indexing="ij"), -1).reshape(-1, 2))
    yy = torch.cat([st.forward(pg[i : i + 2048])[0] for i in range(0, len(pg), 2048)])
    yy = yy.double().view(G, G, -1)
    steps = (yy[:, 1:] - yy[:, :-1]).norm(dim=-1)
    arc = torch.stack([steps[:, i : i + k].sum(1) for i in range(G - k)], 1)
    ratio = ((yy[:, k:] - yy[:, :-k]).norm(dim=-1) / arc.clamp(min=1e-12)).cpu().numpy()
    out["chord/arc 0.1 %/min"] = f"{np.quantile(ratio, 0.001):.2f}/{ratio.min():.2f}"
    # tangent error of a 32 x 96 grid
    n_g = [32, 96]
    yg, dg = render(st, torch.as_tensor(grid_points(n_g)))
    mli = TorchMultilinearInterpolator(2, n_g, st.device)
    yi, di = torch.empty_like(y), torch.empty_like(d.float())
    pf = p.float().to(st.device)
    mli.interpolate(pf, yi, yg)
    mli.interpolate(pf, di, dg)
    tan = ((di.double() - d).norm(dim=1) / d.norm(dim=1).clamp(min=1e-9)).amax(dim=1)
    out["tangent 90/99 %"] = q(tan.cpu().numpy(), [0.9, 0.99])
    # finite-difference noise
    st.st.dp = st.st.dp / 2
    _, d_half = render(st, p)
    fd = ((d_half.double() - d).norm(dim=1) / d.norm(dim=1).clamp(min=1e-9)).cpu()
    out["fd noise 50/90 %"] = q(fd.numpy().ravel(), [0.5, 0.9])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chart", default="Y0")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args()
    for name in args.variants:
        m = metrics(args.chart, VARIANTS[name])
        print(f"{args.chart} {name:14s} " + "  ".join(f"{k} {v}" for k, v in m.items()),
              flush=True)  # fmt: skip
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
