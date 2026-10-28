"""det J of a twincher of a reduced chart with dims (1, 2, 3) on (e, phi) slices at
several rolls, inside the box |p|_inf <= --box (exact y and dy/dp).

Usage:
    python slice_dets.py TWC --box 0.52 --out FIGURE.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import twincher
from matplotlib.colors import TwoSlopeNorm

from train_chart import chart_stencil
from wrinkle_map import render

PSIS = [-0.5, -0.25, 0.0, 0.25, 0.5]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("twc")
    parser.add_argument("--box", type=float, default=0.5)
    parser.add_argument("--G", type=int, default=104)
    parser.add_argument("--chart", default="Y0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--psis", type=float, nargs="+", default=PSIS)
    args = parser.parse_args()
    st = chart_stencil(
        args.chart, 26, kernel="flat", n_grid=8, dp=0.025, dims=[1, 2, 3]
    )
    G, box = args.G, args.box
    x = np.linspace(-box, box, G)
    E, F = np.meshgrid(x, x, indexing="ij")
    fig, axes = plt.subplots(1, len(args.psis), figsize=(4.2 * len(args.psis), 4))
    for ax, psi in zip(axes, args.psis):
        p = torch.as_tensor(np.stack([E.ravel(), F.ravel(), np.full(E.size, psi)], 1))
        y, d = render(st, p)
        sh = twincher.Shuttle()
        sh.load(args.twc, n_c=len(p), n_p=3, n_m=3)
        sh.s.fill_(0)
        sh.s[: st.n_y].copy_(y.T)
        sh.Q.fill_(0)
        sh.Q[: st.n_y].copy_(d.permute(1, 2, 0))
        sh.forward_query()
        det = torch.linalg.det(sh.Q[:3].permute(2, 1, 0)).cpu().numpy().reshape(G, G)
        print(f"psi {psi:+.2f}: det <= 0 {100 * (det <= 0).mean():.2f} %, <= 0.5 "
              f"{100 * (det <= 0.5).mean():.2f} %, min {det.min():.2f}")  # fmt: skip
        norm = TwoSlopeNorm(vcenter=0.5, vmin=min(det.min(), 0) - 1e-6, vmax=1.5)
        im = ax.imshow(det, origin="lower", extent=(-box, box, -box, box), cmap="RdBu",
                       norm=norm)  # fmt: skip
        ax.set_title(f"det J, psi = {psi}")
        ax.set_xlabel("p_phi")
        ax.set_ylabel("p_e")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=75)
    plt.close(fig)


if __name__ == "__main__":
    main()
