"""Noise tolerance of trained twinchers, as twincher.noise_check, with the batched stencil.

twincher.noise_check calls the stencil case by case and estimates dy/dp with the step
1e-8, which does not suit DepthStencil (pixel-quantized forward model: finite differences
need dp ~ 0.05, see study_fd_step.py). This script follows the same procedure with
DepthStencil.compute: for n_c random p_true in [-1, 1]^3, the observation
y_provided = clip(y(p_true) + noise, -1, 1), with noise along a random direction normal to
the manifold (orthogonal to the columns of dy/dp) and of norm uniform in [0, noise_max];
the hs solver starts at p = 0 and runs n_iter steps. The same observations are used for
all twinchers.

Outputs in OUT_DIR: noise_tol_<label>.png per twincher (left, as noise_check: residual
|y(p_solution) - y_provided| against the noise norm; right: |p_solution - p_true|),
noise_tol_compare.png (binned medians of both for all twinchers), noise_check.npz.

Usage:
    python noise_check_depth.py [TWC ...] [--out-dir DIR] [--noise-max 1.0]
                                [--n-c 1024] [--n-iter 32]
(default: output_train_cylinder/final.twc and output_train_noise/iter_*.twc)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import twincher

from train_noise import stencil_from_metadata

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal, light -> dark
BASELINE = "#8a8984"


def label_of(path):
    if path.parent.name == "output_train_cylinder":
        return "stage 2"
    return path.stem.replace("iter_", "iter ")


def observations(st, n_c, noise_max, seed):
    """p_true, y(p_true), y_provided and the noise norms (same random sequence as
    twincher.noise_check)."""
    rng = np.random.default_rng(seed)
    amps = rng.uniform(0, noise_max, size=n_c)
    p_true = rng.uniform(-1, 1, size=(n_c, st.n_p))
    noise = rng.uniform(-1, 1, size=(n_c, st.n_y))
    y_true = torch.empty((n_c, st.n_y), dtype=torch.float64, device=st.device)
    dy_dp = torch.empty((n_c, st.n_y, st.n_p), dtype=torch.float64, device=st.device)
    st.compute(torch.as_tensor(p_true), y_true, dy_dp)
    A = dy_dp.cpu().numpy()
    noise -= (A @ (np.linalg.pinv(A) @ noise[..., None]))[..., 0]  # normal component
    noise *= (amps / np.linalg.norm(noise, axis=1).clip(1e-12))[:, None]
    y_true = y_true.cpu().numpy()
    y_provided = np.clip(y_true + noise, -1, 1)
    return p_true, y_provided, np.linalg.norm(y_provided - y_true, axis=1)


def solve(twc, st, y_provided, n_iter):
    solver = twincher.solver(str(twc), n_c=len(y_provided))
    solver.start_solution(y_provided)
    for _ in range(n_iter):
        st.compute(solver.p, solver.y, solver.dy_dp)
        solver.step(y=solver.y, dy_dp=solver.dy_dp)
    p = solver.p.double().cpu()
    y, _ = st.forward(p)
    residual = np.linalg.norm(y.double().cpu().numpy() - y_provided, axis=1)
    return p.numpy(), residual


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def plot_one(file_name, label, noise, residual, dp, noise_max):
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11, 4.5), layout="constrained", facecolor=SURFACE
    )
    ax0.scatter(noise, residual, s=8, color=RAMP[2], alpha=0.5, edgecolor="none")
    ax0.plot([0, noise_max], [0, noise_max], color=INK2, lw=0.8, ls=":")
    ax0.annotate("residual = noise", (0.98 * noise_max, 0.95 * noise_max),
                 ha="right", va="top", color=INK2, fontsize=9)  # fmt: skip
    ax0.set_xlabel(r"input noise $\|y(p_{true}) - y_{provided}\|$")
    ax0.set_ylabel(r"residual $\|y(p_{solution}) - y_{provided}\|$")
    ax0.set_xlim(0, noise_max)
    ax0.set_ylim(0, max(2 * noise_max, residual.max() * 1.05))
    ax1.scatter(noise, dp, s=8, color=RAMP[2], alpha=0.5, edgecolor="none")
    ax1.set_xlabel(r"input noise $\|y(p_{true}) - y_{provided}\|$")
    ax1.set_ylabel(r"error $\|p_{solution} - p_{true}\|$")
    ax1.set_xlim(0, noise_max)
    ax1.set_ylim(bottom=0)
    for ax in (ax0, ax1):
        style(ax)
    fig.suptitle(f"Noise tolerance: {label}", x=0.01, ha="left", color=INK)
    fig.savefig(file_name, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def plot_compare(file_name, labels, noise, residuals, dps, noise_max, n_bins=10):
    import matplotlib.pyplot as plt

    edges = np.linspace(0, noise_max, n_bins + 1)
    centre = 0.5 * (edges[1:] + edges[:-1])
    bins = np.clip(np.digitize(noise, edges) - 1, 0, n_bins - 1)

    def median(v):
        return np.array([np.median(v[bins == b]) for b in range(n_bins)])

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), layout="constrained", facecolor=SURFACE
    )
    colours = [BASELINE if lab == "stage 2" else None for lab in labels]
    ramp = iter(RAMP[-sum(c is None for c in colours) :])
    colours = [c or next(ramp) for c in colours]
    for lab, col, res, dp in zip(labels, colours, residuals, dps):
        ls = "--" if lab == "stage 2" else "-"
        axes[0].plot(centre, median(res), color=col, ls=ls, marker="o", ms=4, label=lab)
        axes[1].plot(centre, median(dp), color=col, ls=ls, marker="o", ms=4, label=lab)
    axes[0].plot([0, noise_max], [0, noise_max], color=INK2, lw=0.8, ls=":")
    axes[0].set_ylabel(r"median residual $\|y(p_{solution}) - y_{provided}\|$")
    axes[1].set_ylabel(r"median error $\|p_{solution} - p_{true}\|$")
    for ax in axes:
        ax.set_xlabel(r"input noise $\|y(p_{true}) - y_{provided}\|$")
        ax.set_xlim(0, noise_max)
        ax.set_ylim(bottom=0)
        style(ax)
    axes[1].legend(frameon=False, title="twincher", loc="upper left")
    fig.suptitle("Noise tolerance during training with hs_sm_noise (binned medians)",
                 x=0.01, ha="left", color=INK)  # fmt: skip
    fig.savefig(file_name, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("twc", nargs="*", type=Path)
    parser.add_argument(
        "--out-dir", type=Path, default=HERE / "output_train_noise/noise_check"
    )
    parser.add_argument("--noise-max", type=float, default=1.0)
    parser.add_argument("--n-c", type=int, default=1024)
    parser.add_argument("--n-iter", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    files = args.twc or [HERE / "output_train_cylinder/final.twc"] + sorted(
        (HERE / "output_train_noise").glob("iter_*.twc"),
        key=lambda f: int(f.stem.split("_")[1]),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    st = stencil_from_metadata(twincher.load_metadata(str(files[0]))["stencil"])
    p_true, y_provided, noise = observations(st, args.n_c, args.noise_max, args.seed)
    labels, residuals, dps = [], [], []
    print(f"{'twincher':>10s} | error |dp|: median, 90%, max | eta = median |dp|/noise "
          f"(noise > 0.1) | residual/noise: median")  # fmt: skip
    for f in files:
        label = label_of(f)
        p, residual = solve(f, st, y_provided, args.n_iter)
        dp = np.linalg.norm(p - p_true, axis=1)
        labels.append(label)
        residuals.append(residual)
        dps.append(dp)
        big = noise > 0.1
        print(
            f"{label:>10s} | {np.median(dp):.2e} {np.quantile(dp, 0.9):.2e} "
            f"{dp.max():.2e} | {np.median(dp[big] / noise[big]):.3f} | "
            f"{np.median(residual[big] / noise[big]):.3f}"
        )
        name = label.replace(" ", "_")
        plot_one(args.out_dir / f"noise_tol_{name}.png", label, noise, residual, dp,
                 args.noise_max)  # fmt: skip
    plot_compare(args.out_dir / "noise_tol_compare.png", labels, noise, residuals, dps,
                 args.noise_max)  # fmt: skip
    np.savez(
        args.out_dir / "noise_check.npz",
        files=[str(f) for f in files],
        labels=labels,
        p_true=p_true,
        noise=noise,
        residuals=np.array(residuals),
        dps=np.array(dps),
    )
    print(f"plots in {args.out_dir}")


if __name__ == "__main__":
    main()
