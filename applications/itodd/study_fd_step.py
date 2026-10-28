"""Finite-difference dy/dp of DepthStencil as a function of the step dp and the resolution.

Computes central differences (y(p + dp e_k) - y(p - dp e_k)) / (2 dp) for n_cases random
p, all probes, steps dp in [1e-4, 0.3] and resolutions 256..2048 (the kernel size scales
with the resolution, so the stencil has a limit for resolution -> infinity), writes them
to OUT_DIR/fd_step.npz and plots them (plot_fd_step.py).

Usage:
    python study_fd_step.py [OUT_DIR] [--n-cases 32]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from depth_stencil import DepthStencil
from itodd_data import model

MODEL = model(12)
RESOLUTIONS = (256, 512, 1024, 2048)
STEPS = np.logspace(-4, -0.5, 15)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "output_fd_step",
    )
    parser.add_argument("--n-cases", type=int, default=32)
    args = parser.parse_args()

    g = torch.Generator().manual_seed(0)
    p = 1.6 * torch.rand((args.n_cases, 3), generator=g, dtype=torch.float64) - 0.8
    n_k = p.shape[1]
    y = {}  # resolution -> (n_cases, n_y)
    fd = {}  # resolution -> (n_steps, n_k, n_cases, n_y)
    for res in RESOLUTIONS:
        st = DepthStencil(MODEL, resolution=res)
        t0 = time.perf_counter()
        y[res] = st.forward(p)[0].cpu().numpy()
        e = torch.eye(n_k, dtype=torch.float64)
        q = torch.stack(
            [p[None] + sign * h * e[:, None] for h in STEPS for sign in (1, -1)]
        )  # (n_steps*2, n_k, n_cases, n_p)
        s = torch.cat([st.forward(c)[0] for c in q.reshape(-1, 3).split(st.chunk)])
        s = s.view(len(STEPS), 2, n_k, args.n_cases, st.n_y).cpu().numpy()
        fd[res] = (s[:, 0] - s[:, 1]) / (2 * STEPS[:, None, None, None])
        torch.cuda.synchronize()
        print(f"resolution {res}: {time.perf_counter() - t0:.1f} s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "fd_step.npz",
        p=p.numpy(),
        steps=STEPS,
        resolutions=np.array(RESOLUTIONS),
        **{f"y_{r}": y[r] for r in RESOLUTIONS},
        **{f"fd_{r}": fd[r] for r in RESOLUTIONS},
    )
    print(f"written {args.out_dir / 'fd_step.npz'}")


if __name__ == "__main__":
    main()
