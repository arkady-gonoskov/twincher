"""Noise tolerance of chart twinchers (e.g. the snapshots of noise_charts.py).

As noise_check_depth.py (milestone 1) for the charts of train_chart.py: n_c random views
p_true in |p|_inf <= --box, rendered by the stencil of the first twincher (metadata);
observations y_provided = clip(y(p_true) + noise, -1, 1) with noise along a random
direction normal to the manifold (orthogonal to the columns of dy/dp) and of norm
uniform in [0, --noise-max]; the hs solver starts at p = 0, runs --n-iter steps with p
clamped to the box (the views are rendered, so each twincher takes a few minutes). The same observations for all twinchers.

Per twincher: eta = median |p_solution - p_true| / |noise| (the sensitivity of the
solution to off-manifold perturbations; 0 is ideal), the median ratio residual / noise
(residual |y(p_solution) - y_provided|; 1: the solution is the projection of the
observation onto the manifold), the fraction of views with max_k |dp_k| < 0.01 for noise
norms below 0.1, and the noise-free accuracy (a separate set of views without noise).

Outputs in --out-dir: noise_check.csv (one row per twincher), noise_check.png.

Usage:
    python noise_check_chart.py TWC [TWC ...] [--out-dir DIR] [--noise-max 0.5]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import twincher

from train_chart import stencil_from_metadata

HERE = Path(__file__).resolve().parent


def observations(st, n_c, box, noise_max, seed):
    """p_true, y_provided and the noise norms."""
    rng = np.random.default_rng(seed)
    amps = rng.uniform(0, noise_max, size=n_c)
    p_true = rng.uniform(-box, box, size=(n_c, st.n_p))
    noise = rng.uniform(-1, 1, size=(n_c, st.n_y))
    y = torch.empty((n_c, st.n_y), dtype=torch.float64, device=st.device)
    dy_dp = torch.empty((n_c, st.n_y, st.n_p), dtype=torch.float64, device=st.device)
    for i in range(0, n_c, 512):
        st.compute(torch.as_tensor(p_true[i : i + 512]), y[i : i + 512],
                   dy_dp[i : i + 512])  # fmt: skip
    A = dy_dp.cpu().numpy()
    noise -= (A @ (np.linalg.pinv(A) @ noise[..., None]))[..., 0]  # normal component
    noise *= (amps / np.linalg.norm(noise, axis=1).clip(1e-12))[:, None]
    y = y.cpu().numpy()
    y_provided = np.clip(y + noise, -1, 1)
    return p_true, y_provided, np.linalg.norm(y_provided - y, axis=1)


def solve(twc, st, y_provided, n_iter, box):
    n = len(y_provided)
    solver = twincher.solver(str(twc), n_c=n)
    solver.start_solution(torch.as_tensor(y_provided, dtype=solver.y.dtype,
                                          device=solver.y.device))  # fmt: skip
    for _ in range(n_iter):
        for i in range(0, n, 512):
            st.compute(solver.p[i : i + 512], solver.y[i : i + 512],
                       solver.dy_dp[i : i + 512])  # fmt: skip
        solver.step(y=solver.y, dy_dp=solver.dy_dp)
        solver.p.clamp_(-box, box)
    p = solver.p.double().cpu()
    y = torch.cat([st.forward(p[i : i + 512])[0] for i in range(0, n, 512)])
    residual = np.linalg.norm(y.double().cpu().numpy() - y_provided, axis=1)
    return p.numpy(), residual


def plot(file_name, labels, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink2, muted, grid = "#fcfcfb", "#52514e", "#898781", "#e1e0d9"
    blue, orange = "#2a78d6", "#eb6834"
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained",
                             facecolor=surface)  # fmt: skip
    panels = [("eta", "eta = median |dp| / |noise|", blue),
              ("residual_ratio", "median residual / noise", orange),
              ("solved_low_noise", "views solved (< 0.01) at noise < 0.1", blue)]  # fmt: skip
    for ax, (key, title, color) in zip(axes, panels):
        ax.plot(x, [r[key] for r in rows], color=color, marker="o", ms=4, lw=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x[:: max(1, len(x) // 10)])
        ax.set_xticklabels(labels[:: max(1, len(x) // 10)], rotation=45, ha="right")
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=ink2)
    axes[1].axhline(1.0, color=muted, lw=0.8, ls=":")
    fig.savefig(file_name, dpi=110, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("twc", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--noise-max", type=float, default=0.5)
    parser.add_argument("--box", type=float, default=0.86)
    parser.add_argument("--n-c", type=int, default=512)
    parser.add_argument("--n-free", type=int, default=256)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    out = args.out_dir or args.twc[0].parent / "noise_check"
    out.mkdir(parents=True, exist_ok=True)
    st = stencil_from_metadata(twincher.load_metadata(str(args.twc[0]))["stencil"])
    p_true, y_obs, noise = observations(st, args.n_c, args.box, args.noise_max,
                                        args.seed)  # fmt: skip
    p_free, y_free, _ = observations(st, args.n_free, args.box, 0.0, args.seed + 1)
    rows, labels = [], []
    for twc in args.twc:
        p, residual = solve(twc, st, y_obs, args.n_iter, args.box)
        dp = np.linalg.norm(p - p_true, axis=1)
        dp_max = np.abs(p - p_true).max(axis=1)
        p0, _ = solve(twc, st, y_free, args.n_iter, args.box)
        dp0 = np.abs(p0 - p_free).max(axis=1)
        sel = noise > 0.05 * args.noise_max
        low = noise < 0.1
        row = {
            "twc": str(twc.relative_to(HERE)) if twc.is_relative_to(HERE) else str(twc),
            "eta": float(np.median(dp[sel] / noise[sel])),
            "residual_ratio": float(np.median(residual[sel] / noise[sel])),
            "solved_low_noise": float((dp_max[low] < 0.01).mean()),
            "failures": float((dp_max > 0.1).mean()),
            "free_median": float(np.median(dp0)),
            "free_below_1e-3": float((dp0 < 1e-3).mean()),
            "free_below_0.01": float((dp0 < 0.01).mean()),
        }
        rows.append(row)
        labels.append(twc.stem)
        print(f"{row['twc']}: eta {row['eta']:.4f}, residual/noise "
              f"{row['residual_ratio']:.3f}, solved at noise < 0.1 "
              f"{100 * row['solved_low_noise']:.1f} %, failures (> 0.1) "
              f"{100 * row['failures']:.1f} %; noise-free: median {row['free_median']:.1e},"
              f" < 1e-3 {100 * row['free_below_1e-3']:.1f} %, < 0.01 "
              f"{100 * row['free_below_0.01']:.1f} %", flush=True)  # fmt: skip
    with open(out / "noise_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(out / "noise_check.png", labels, rows)
    print(f"written: {out / 'noise_check.csv'}, {out / 'noise_check.png'}")


if __name__ == "__main__":
    main()
