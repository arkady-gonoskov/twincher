"""Stage 3: continue training a twincher with the single-mode noise loss (hs_sm_noise).

Starts from a trained twincher (default: the result of train_cylinder.py), with the same
grid data, keeps the hs loss (nondegeneracy, curriculum range R = 1 from the start; that of
twincher 0.1.0, HS_0_1_0) and adds the single-mode noise loss hs_sm_noise of the package.

Outputs in OUT_DIR: lc.png, history.csv and history.png (R, min det J, largest
normal-mode sensitivity |dr/dy| of the batch), iter_<n>.twc every --checkpoint
iterations, final.twc.

Usage:
    python -u train_noise.py [OUT_DIR] [--init TWC] [--iters 10000] [--y-range 0.05]
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import twincher

from depth_stencil import DepthStencil
from train_cylinder import HS_0_1_0, MODEL, grid_data

HERE = Path(__file__).resolve().parent


def stencil_from_metadata(cfg):
    return DepthStencil(
        MODEL,
        n_grid=cfg["n_grid"],
        resolution=cfg["resolution"],
        kernel=cfg.get("kernel", "max"),
        dp=cfg["dp"],
        dist_range=cfg["ranges"][0],
        elevation_range=cfg["ranges"][1],
        roll_range=cfg["ranges"][2],
    )


def plot_history(rows, file_name, det_clearance):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink2, grid = "#fcfcfb", "#52514e", "#e4e3df"
    blue, orange, dark = "#2a78d6", "#eb6834", "#1c5cab"
    it = np.array([r["iter"] for r in rows])
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True, layout="constrained", facecolor=surface
    )
    axes[0].plot(it, [r["R"] for r in rows], color=blue, lw=1.5)
    axes[0].set_ylabel("curriculum range R")
    axes[0].set_ylim(0, 1.05)
    axes[1].plot(it, [r["det_min"] for r in rows], color=orange, lw=1.5)
    axes[1].axhline(det_clearance, color=ink2, lw=0.8, ls=":")
    axes[1].axhline(0, color=ink2, lw=0.8, ls=":")
    axes[1].set_ylabel("min det J over the stride")
    axes[1].set_ylim(-1, 2)
    axes[2].plot(it, [r["vel_max"] for r in rows], color=dark, lw=1.5, label="max")
    axes[2].plot(it, [r["vel_p75"] for r in rows], color=blue, lw=1.5, ls="--",
                 label="75th percentile")  # fmt: skip
    axes[2].set_yscale("log")
    axes[2].set_ylabel("worst normal-mode |dr/dy|")
    axes[2].set_xlabel("iteration")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(color=grid, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=ink2)
    fig.savefig(file_name, dpi=110, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "out_dir", nargs="?", type=Path, default=HERE / "output_train_noise"
    )
    parser.add_argument(
        "--init", type=Path, default=HERE / "output_train_cylinder/final.twc"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=HERE / "output_train_cylinder/data_n32_res512_dp0.05_centre.pt",
    )
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument("--checkpoint", type=int, default=2000)
    parser.add_argument("--y-range", type=float, default=0.05)
    parser.add_argument("--amp", type=float, help="of hs_sm_noise (default: its own)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    src = twincher.Shuttle()
    src.load(str(args.init), n_c=1, n_p=3, n_m=1)
    cfg = src.metadata["stencil"]
    st = stencil_from_metadata(cfg)

    learner = twincher.Learner(
        arch="hs",
        n_s=src.n_s,
        n_l=src.n_l,
        dir=args.out_dir,
        output_stride=args.stride,
        plot_progress=True,
    )
    noise_params = {"y_range": args.y_range}
    if args.amp is not None:
        noise_params["amp"] = args.amp
    learner.modify_params(
        {"loss_terms": {"hs_noise": None, "hs": HS_0_1_0, "hs_sm_noise": noise_params}}
    )
    p_grid = learner.get_p_request(n_p=st.n_p, n_grid=cfg["data_grid"])
    y, dy_dp = grid_data(st, p_grid, args.data)
    learner.set_data(y, dy_dp, stencil_name="itodd_obj12_depth")
    learner.init()

    # start from the trained twincher: copy its parameters into the learner's shuttle
    # (whose parameter tensor is the one the optimizer updates) and check the result
    learner.SH.a.copy_(src.a)
    learner.SH.tw_update()
    learner.SH.metadata["stencil"] = cfg
    learner.SH.metadata["init"] = str(args.init)
    check = twincher.Shuttle()
    check.load(str(args.init), n_c=learner.n_c, n_p=3, n_m=1)
    for sh in (learner.SH, check):
        sh.s[: st.n_y].copy_(learner.data_y[: learner.n_c].T)
        sh.s[st.n_y :].fill_(0)
        sh.forward_inference()
    diff = (learner.SH.s - check.s).abs().max().item()
    print(f"twincher from {args.init}: max |s_out difference| = {diff:.1e}")
    if diff > 1e-5:
        raise RuntimeError("the learner does not reproduce the loaded twincher")
    del check

    terms = {term.name: term for term in learner.loss_terms}
    hs, sm = terms["hs"], terms["hs_sm_noise"]
    hs.p_range = learner.p_range = 1.0  # the twincher is valid on the whole domain
    print(
        f"loss terms: {list(terms)}; hs_sm_noise: amp {sm.amp}, y_range {sm.y_range}, "
        f"vel_scale {sm.vel_scale}, vel_u {sm.vel_u}"
    )

    rows, t0 = [], time.perf_counter()
    det_min, n_neg, vel_max = float("inf"), 0, 0.0
    fields = ["iter", "time_s", "R", "det_min", "neg_dets", "loss_hs", "loss_noise"]
    fields += ["vel_p75", "vel_max"]
    with open(args.out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for it in range(args.iters + 1):
            learner.step()
            det_min = min(det_min, hs.dets.min().item())
            n_neg += (hs.dets < 0).sum().item()
            vel_max = max(vel_max, sm.vel_max.max().item())
            if it % args.stride == 0:
                row = {
                    "iter": it,
                    "time_s": round(time.perf_counter() - t0, 1),
                    "R": round(hs.p_range, 3),
                    "det_min": round(det_min, 4),
                    "neg_dets": n_neg,
                    "loss_hs": float(hs.loss),
                    "loss_noise": float(sm.loss),
                    "vel_p75": torch.quantile(sm.vel_max, 0.75).item(),
                    "vel_max": vel_max,
                }
                rows.append(row)
                writer.writerow(row)
                f.flush()
                print(
                    f"{it:6d} {row['time_s']:8.1f} s  R {hs.p_range:.3f}  "
                    f"min det {det_min:+.3f}  neg {n_neg:5d}  "
                    f"|dr/dy|_normal p75 {row['vel_p75']:.3e} max {vel_max:.3e}  "
                    f"loss hs {row['loss_hs']:.2e} noise {row['loss_noise']:.2e}"
                )
                plot_history(rows, args.out_dir / "history.png", hs.det_clearance)
                det_min, n_neg, vel_max = float("inf"), 0, 0.0
            if it % args.checkpoint == 0 and it > 0:
                learner.SH.save(str(args.out_dir / f"iter_{it}.twc"))
    learner.SH.save(str(args.out_dir / "final.twc"))
    print(f"saved {args.out_dir / 'final.twc'} after {it} iterations")


if __name__ == "__main__":
    main()
