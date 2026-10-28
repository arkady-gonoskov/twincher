"""Finite-difference step for the stencil variants of a chart (n_p = 4).

For n_cases random p of a chart, dy/dp by central differences at resolution 512 for the
steps STEPS, compared with a reference at resolution 2048 (kernel sizes scale with the
resolution) and step 0.02: relative error |dy/dp - ref| / |ref| (norm over y) per case and
parameter; printed per variant: median and 90 % quantile for each step.

Usage:
    python study_fd_step_chart.py [--chart Z0] [--n-cases 32]
"""

import argparse

import numpy as np
import torch

from train_chart import chart_stencil

STEPS = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]
VARIANTS = {  # name: stencil options
    "tent 16": {"kernel": "tent", "n_grid": 16},
    "flat 16": {"kernel": "flat", "n_grid": 16},
    "flat 12": {"kernel": "flat", "n_grid": 12},
    "flat 8": {"kernel": "flat", "n_grid": 8},
    "spectral 64": {"kernel": "flat", "n_grid": 16, "spectral": 64},
    "spectral 16": {"kernel": "flat", "n_grid": 16, "spectral": 16},
}


def fd(st, p, dp):
    st.dp = torch.full((st.n_p,), dp, dtype=torch.float64, device=st.device)
    y = torch.empty((len(p), st.n_y), device=st.device)
    d = torch.empty((len(p), st.n_y, st.n_p), device=st.device)
    for i in range(0, len(p), 8):
        st.compute(p[i : i + 8], y[i : i + 8], d[i : i + 8])
    return d.double().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chart", default="Z0")
    parser.add_argument("--n-cases", type=int, default=32)
    args = parser.parse_args()
    g = torch.Generator().manual_seed(0)
    p = 1.7 * torch.rand((args.n_cases, 4), generator=g, dtype=torch.float64) - 0.85
    print(
        "relative error of dy/dp (median / 90 %) for the steps "
        + ", ".join(map(str, STEPS))
    )
    for name, opts in VARIANTS.items():
        ref = fd(chart_stencil(args.chart, 26, resolution=2048, **opts), p, 0.02)
        st = chart_stencil(args.chart, 26, resolution=512, **opts)
        norm = ref.norm(dim=1)  # (cases, n_p)
        line = []
        for dp in STEPS:
            err = ((fd(st, p, dp) - ref).norm(dim=1) / norm.clamp(min=1e-9)).numpy()
            line.append(f"{np.median(err):.3f}/{np.quantile(err, 0.9):.3f}")
        print(f"{name:12s} " + "  ".join(line), flush=True)
        del ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
