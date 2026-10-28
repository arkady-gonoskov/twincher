"""Plots of the results of study_fd_step.py.

fd_step_probes.png: dy/dp_k against the step dp for three probes of one view
(rows: a probe deep on the object, two probes at its edge; columns: k), one line per
resolution. fd_step_error.png: relative error against dp per k: median over the
probes that see the object (y < background) of |FD(res, dp) - FD(2048, dp)| for
resolutions 256..1024 (pixel noise), and of |FD(2048, dp) - FD(2048, 0.01)| (smoothing
by large steps); both divided by the RMS of dy/dp_k.

Usage:
    python plot_fd_step.py [OUT_DIR]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAMP = {256: "#86b6ef", 512: "#3987e5", 1024: "#1c5cab", 2048: "#0d366b"}
SMOOTH = "#8a8984"
NAMES = ["p0 (distance)", "p1 (elevation)", "p2 (roll)"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "lines.linewidth": 1.5,
    "legend.frameon": False,
})  # fmt: skip


def main():
    out_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "output_fd_step"
    )
    d = np.load(out_dir / "fd_step.npz")
    steps, res = d["steps"], [int(r) for r in d["resolutions"]]
    fd = {r: d[f"fd_{r}"] for r in res}  # (n_steps, n_k, n_cases, n_y)
    y = d["y_2048"]  # (n_cases, n_y)
    bg = np.abs(y - 0.8) < 1e-6

    # example probes of view 0: deepest, and two at the edge (y about 0.3 and 0.7)
    c = 0
    i_ref = np.argmin(np.abs(steps - 0.01))
    probes = [
        ("deep on the object", np.argmin(y[c])),
        ("edge, y ≈ 0.3", np.argmin(np.abs(y[c] - 0.3))),
        ("edge, y ≈ 0.7", np.argmin(np.where(bg[c], 9, np.abs(y[c] - 0.7)))),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, layout="constrained")
    for row, (label, j) in enumerate(probes):
        for k in range(3):
            ax = axes[row, k]
            for r in res:
                ax.plot(steps, fd[r][:, k, c, j], color=RAMP[r], marker="o",
                        markersize=3.5, label=f"{r}²")  # fmt: skip
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(f"dy/d{NAMES[k]}", loc="left")
            if k == 0:
                ax.set_ylabel(f"probe {j}: {label}\ny = {y[c, j]:+.2f}")
            if row == 2:
                ax.set_xlabel("step Δp")
            # the scale of the stable part, not of the noise at small steps
            stable = fd[2048][i_ref:, k, c, j]
            lo, hi = stable.min(), stable.max()
            pad = max(0.5 * (hi - lo), 0.25 * max(abs(lo), abs(hi)), 0.05)
            ax.set_ylim(lo - pad, hi + pad)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), title="resolution",
               loc="outside upper right", ncol=4)  # fmt: skip
    fig.suptitle(f"Central differences of three probes of view 0, p = {np.round(d['p'][c], 2)}",
                 x=0.01, ha="left")  # fmt: skip
    fig.savefig(out_dir / "fd_step_probes.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, layout="constrained")
    seen = ~bg
    for k, ax in enumerate(axes):
        scale = np.sqrt((fd[2048][i_ref, k] ** 2).mean())
        for r in res[:-1]:
            err = np.abs(fd[r][:, k] - fd[2048][:, k])[:, seen]
            ax.plot(steps, np.median(err, axis=1) / scale, color=RAMP[r], marker="o",
                    markersize=3.5, label=f"pixel noise, {r}² vs 2048²")  # fmt: skip
        sm = np.abs(fd[2048][i_ref:, k] - fd[2048][i_ref, k])[:, seen]
        ax.plot(steps[i_ref + 1 :], np.median(sm, axis=1)[1:] / scale, color=SMOOTH,
                linestyle="--", marker="s", markersize=3.5,
                label="smoothing, 2048²: Δp vs 0.01")  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"dy/d{NAMES[k]}", loc="left")
        ax.set_xlabel("step Δp")
    axes[0].set_ylabel("median relative error (probes seeing the object)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Error of finite-difference dy/dp: pixel noise and smoothing", x=0.01,
                 ha="left")  # fmt: skip
    fig.savefig(out_dir / "fd_step_error.png", dpi=120)
    plt.close(fig)
    print(f"written {out_dir / 'fd_step_probes.png'}, {out_dir / 'fd_step_error.png'}")


if __name__ == "__main__":
    main()
