"""Stage 2: train an hs twincher on the depth stencil of the cylinder (ITODD obj 12).

Data: y and dy/dp (central differences with the step dp) of DepthStencil on an
n_grid^3 grid of p (Learner.get_p_request), cached in OUT_DIR/data_*.pt; the learner
interpolates them multilinearly (set_data). The noise loss (hs_noise) is disabled. The hs
loss is that of twincher 0.1.0 (HS_0_1_0), with which these twinchers were trained.

Outputs in OUT_DIR: lc.png (learning curve of the package, with min det J and
max |dr/dp| per stride), history.csv and r_history.png (curriculum range R, min det J
and negative determinants per stride), iter_<n>.twc checkpoints, final.twc.

Usage:
    python -u train_cylinder.py [OUT_DIR] [--iters 30000] [--stride 500]
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import twincher

from depth_stencil import DepthStencil
from itodd_data import model

MODEL = model(12)
# The hs loss of twincher 0.1.0, with which the runs of Part I and the first runs of
# Part II were made (README Section 8): the amplitude of the det loss raised towards the
# border of the curriculum box by the sum over the components (ramp_boost 2), no trimmer
# over the whole domain, no momentum reset.
HS_0_1_0 = {"ramp_mode": "sum", "ramp_boost": 2.0, "trimmer_domain_amp": 0.0,
            "momentum_reset": None}  # fmt: skip


def grid_data(st, p_grid, file_name, batch=1024):
    """y (n, n_y) and dy/dp (n, n_y, n_p) on the grid, from the cache if present."""
    if file_name.exists():
        data = torch.load(file_name)
        if np.array_equal(data["p"].numpy(), p_grid):
            print(f"data from {file_name}")
            return data["y"], data["dy_dp"]
    t0 = time.perf_counter()
    p = torch.as_tensor(p_grid, device=st.device)
    y = torch.empty((len(p), st.n_y), device=st.device)
    dy_dp = torch.empty((len(p), st.n_y, st.n_p), device=st.device)
    for i in range(0, len(p), batch):
        st.compute(p[i : i + batch], y[i : i + batch], dy_dp[i : i + batch])
    y, dy_dp = y.cpu(), dy_dp.cpu()
    torch.save({"p": torch.as_tensor(p_grid), "y": y, "dy_dp": dy_dp}, file_name)
    print(f"data computed in {time.perf_counter() - t0:.0f} s, saved to {file_name}")
    return y, dy_dp


def plot_history(rows, file_name, det_margin, det_clearance):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink2, grid, blue, orange = (
        "#fcfcfb",
        "#52514e",
        "#e4e3df",
        "#2a78d6",
        "#eb6834",
    )
    it = np.array([r["iter"] for r in rows])
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, layout="constrained", facecolor=surface
    )
    ax0.plot(it, [r["R"] for r in rows], color=blue, lw=1.5)
    ax0.set_ylabel("curriculum range R")
    ax0.set_ylim(0, 1.05)
    ax1.plot(it, [r["det_min"] for r in rows], color=orange, lw=1.5)
    for level, label in [
        (det_margin, "margin"),
        (det_clearance, "clearance"),
        (0, "0"),
    ]:
        ax1.axhline(level, color=ink2, lw=0.8, ls=":")
        ax1.annotate(label, (1.0, level), xycoords=("axes fraction", "data"),
                     xytext=(4, 0), textcoords="offset points", va="center",
                     color=ink2, fontsize=9)  # fmt: skip
    ax1.set_ylabel("min det J over the stride")
    ax1.set_ylim(-1, 2)
    ax1.set_xlabel("iteration")
    for ax in (ax0, ax1):
        ax.set_facecolor(surface)
        ax.grid(color=grid, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=ink2)
    fig.savefig(file_name, dpi=110, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "output_train_cylinder",
    )
    parser.add_argument("--iters", type=int, default=30000, help="maximal iterations")
    parser.add_argument("--stride", type=int, default=500, help="output stride")
    parser.add_argument("--checkpoint", type=int, default=5000)
    parser.add_argument("--n-s", type=int, default=256)
    parser.add_argument("--n-l", type=int, default=256)
    parser.add_argument("--n-grid", type=int, default=32, help="grid points per p_k")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dp", type=float, default=0.05, help="finite-difference step")
    parser.add_argument("--kernel", choices=["max", "tent"], default="max")
    parser.add_argument(
        "--roll-range", type=float, nargs=2, metavar=("LO", "HI"),
        default=(-0.5 * math.pi + 0.3, 0.5 * math.pi - 0.3), help="radians",
    )  # fmt: skip
    parser.add_argument(
        "--elevation-range", type=float, nargs=2, metavar=("LO", "HI"),
        default=(-0.5 * math.pi + 0.3, 0.5 * math.pi - 0.3), help="radians",
    )  # fmt: skip
    parser.add_argument(
        "--done-strides", type=int, default=4,
        help="stop after this many strides at R = 1 with min det J > clearance",
    )  # fmt: skip
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    st = DepthStencil(
        MODEL,
        resolution=args.resolution,
        kernel=args.kernel,
        dp=args.dp,
        roll_range=tuple(args.roll_range),
        elevation_range=tuple(args.elevation_range),
    )
    learner = twincher.Learner(
        arch="hs",
        n_s=args.n_s,
        n_l=args.n_l,
        dir=args.out_dir,
        output_stride=args.stride,
        plot_progress=True,
    )
    learner.modify_params({"loss_terms": {"hs_noise": None, "hs": HS_0_1_0}})
    p_grid = learner.get_p_request(n_p=st.n_p, n_grid=args.n_grid)
    name = f"data_n{args.n_grid}_res{args.resolution}_dp{args.dp:g}_centre.pt"
    if not np.allclose(
        st.ranges[1:], [(-0.5 * math.pi + 0.3, 0.5 * math.pi - 0.3)] * 2
    ):
        name = name.replace(".pt", "_roll{:.3f}_{:.3f}.pt".format(*args.roll_range))
    if args.kernel != "max":
        name = name.replace(".pt", f"_{args.kernel}.pt")
    y, dy_dp = grid_data(st, p_grid, args.out_dir / name)
    learner.set_data(y, dy_dp, stencil_name="itodd_obj12_depth")
    learner.init()
    learner.SH.metadata["stencil"] = {
        "model": "itodd obj_000012",
        "n_grid": st.n_grid,
        "resolution": st.res,
        "kernel": st.kernel,
        "kernel_size": st.kernel_size,
        "ranges": st.ranges,
        "s_amp": st.s_amp,
        "dp": args.dp,
        "data_grid": args.n_grid,
    }
    hs = next(term for term in learner.loss_terms if term.name == "hs")
    print(
        f"n_s {args.n_s}, n_l {args.n_l}, n_c {learner.n_c}, n_y {st.n_y}, n_p {st.n_p}"
    )

    rows, det_min, n_neg, done, t0 = [], float("inf"), 0, 0, time.perf_counter()
    with open(args.out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, ["iter", "time_s", "R", "det_min", "neg_dets", "loss", "lr"]
        )
        writer.writeheader()
        for it in range(args.iters + 1):
            learner.step()
            det_min = min(det_min, hs.dets.min().item())
            n_neg += (hs.dets < 0).sum().item()
            if it % args.stride == 0:
                row = {
                    "iter": it,
                    "time_s": round(time.perf_counter() - t0, 1),
                    "R": round(hs.p_range, 3),
                    "det_min": round(det_min, 4),
                    "neg_dets": n_neg,
                    "loss": float(learner.loss),
                    "lr": learner.lr_dynamic,
                }
                rows.append(row)
                writer.writerow(row)
                f.flush()
                print(
                    f"{it:6d} {row['time_s']:8.1f} s  R {hs.p_range:.3f}  "
                    f"min det {det_min:+.3f}  neg {n_neg:6d}  loss {row['loss']:.3e}  "
                    f"{learner.state}"
                )
                plot_history(
                    rows,
                    args.out_dir / "r_history.png",
                    hs.det_margin,
                    hs.det_clearance,
                )
                done = done + 1 if hs.p_range >= 1 and det_min > hs.det_clearance else 0
                det_min, n_neg = float("inf"), 0
            if it % args.checkpoint == 0 and it > 0:
                learner.SH.save(str(args.out_dir / f"iter_{it}.twc"))
            if done >= args.done_strides:
                print(f"R = 1 with min det J > clearance for {done} strides: stop")
                break
    learner.SH.save(str(args.out_dir / "final.twc"))
    print(f"saved {args.out_dir / 'final.twc'} after {it} iterations")


if __name__ == "__main__":
    main()
