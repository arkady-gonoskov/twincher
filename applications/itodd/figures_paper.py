"""Figures of the paper (section on 6D localization in ITODD and its appendix), as PDF.

From the outputs of run_m3.py (output_m3/), data_space_gn.py (output_data_space_gn/),
gn_trajectories.py (output_gn_trajectories/), adam_momentum.py (output_momentum/),
y0_reduced.py (output_y0_reduced/) and the histories of y0_fb.py, y0_full.py and
train_charts_fb.py; the figures "scenes" and "vortices" render views and need a GPU. Plain white background and sizes for a text width of 6.5 inches; the colours
of README Parts II and III. With --format png, the same figures for the README. Output
in output_figures_paper/ unless --out-dir is given.

- itodd_scenes.pdf: whole scenes (as scene_demo.py), cropped to the objects;
- itodd_agreement.pdf: depth mismatch against the unavailable fraction, and the fraction
  of solutions that agree with the observation below a threshold of it;
- itodd_convergence.pdf: synthetic views solved against the Gauss-Newton step in r and
  in y; MSSD to the ground truth against the step on the validation instances; the
  residual of the solver in r;
- itodd_scaling.pdf: growth of the curriculum range R for ChartY0 with n_p = 2, 3, 4 and
  for the six charts, and the iterations per step of R against the volume of the shell
  that the step adds (README Section 16);
- itodd_vortices.pdf: r meshes and winding numbers of the reduced ChartY0 (64 and 256
  layers) and a zoom on a branch point;
- itodd_momentum.pdf: the momentum study of README Section 18.

Usage:
    python figures_paper.py [--out-dir DIR] [--format {pdf,png}]
                            [scenes agreement convergence scaling vortices momentum]
"""

import argparse
import csv
import gc
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from itodd_data import MODELS_EVAL

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_figures_paper"
INFO = MODELS_EVAL / "models_info.json"
NAMES = {12: "cylinder", 26: "tee connector"}
COLORS = {12: "#2a78d6", 26: "#eb6834"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = {12: "o", 26: "^"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#3f3e3b", "#77756f", "#e3e2dc"
GOOD_IOU, GOOD_DZ = 0.95, 2.0
RUNS = {(12, "mask"): "output_m3/obj12_tent_mask",
        (12, "visib"): "output_m3/obj12_tent_visib",
        (26, "mask"): "output_m3/obj26_mask", (26, "visib"): "output_m3/obj26_visib"}  # fmt: skip
PAPER_RC = {  # applied again after importing figures_m2, which sets the README style
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#a9a8a0", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.2, "pdf.fonttype": 42,
}  # fmt: skip
plt.rcParams.update(PAPER_RC)


def diameters():
    info = json.loads(INFO.read_text())
    return {obj: info[str(obj)]["diameter"] for obj in NAMES}


def save(fig, out_dir, name):
    path = out_dir / f"{name}.{plt.rcParams['savefig.format']}"  # set by --format
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    gc.collect()
    print(path)


def panel_label(ax, text, x=-0.02, y=1.02):
    ax.text(x, y, text, transform=ax.transAxes, ha="right", va="bottom",
            fontweight="bold", fontsize=9)  # fmt: skip


# --- agreement with the observation ---------------------------------------------------
def load_results():
    return {k: json.loads((HERE / v / "results.json").read_text())
            for k, v in RUNS.items()}  # fmt: skip


def column(results, key):
    return np.array([r["metrics"].get(key, np.nan) for r in results], float)


def agreeing(results):
    return (column(results, "silhouette_iou") >= GOOD_IOU) & (
        column(results, "depth_mismatch_mm") <= GOOD_DZ
    )


def fig_agreement(out_dir):
    data = {k: [r for r in v if not r["touches_border"]] for k, v in load_results().items()}
    D = diameters()
    labels = {"mask": "amodal mask", "visib": "visible mask"}
    fig = plt.figure(figsize=(6.5, 2.55), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[2.2, 1])
    ax = fig.add_subplot(gs[:, 0])
    for obj in NAMES:  # 5 % of the diameter: the strictest MSSD threshold of BOP
        ax.axhline(0.05 * D[obj], color=COLORS[obj], lw=0.8, ls=(0, (4, 3)), zorder=1)
    for (obj, mode), rs in data.items():
        filled = mode == "mask"
        ax.scatter(column(rs, "unavailable"), column(rs, "depth_mismatch_mm"), s=9,
                   marker=MARKERS[obj], linewidths=0.6, alpha=0.8,
                   facecolors=COLORS[obj] if filled else "none", edgecolors=COLORS[obj],
                   label=f"{NAMES[obj]}, {labels[mode]}", zorder=2)  # fmt: skip
    ax.axhline(GOOD_DZ, color=MUTED, lw=0.6, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 300)
    ax.set_xlim(0, 1)
    ax.set_xlabel("unavailable fraction of the object")
    ax.set_ylabel(r"mean $|Z_\mathrm{solution} - Z_\mathrm{sensor}|$ (mm)")
    ax.grid(color=GRID, lw=0.4)
    ax.legend(loc="upper left", ncols=2, handletextpad=0.2, columnspacing=0.8,
              borderaxespad=0.2)  # fmt: skip
    panel_label(ax, "(a)", x=-0.1)
    # agreement below a threshold of the unavailable fraction, and the number of instances
    thresholds = np.linspace(0.05, 1.0, 96)
    ax1, ax2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    for (obj, mode), rs in data.items():
        u, ok = column(rs, "unavailable"), agreeing(rs)
        frac = [ok[u <= t].mean() if (u <= t).any() else np.nan for t in thresholds]
        ax1.plot(thresholds, frac, color=COLORS[obj], ls="-" if mode == "mask" else "--",
                 label=f"{NAMES[obj]}, {labels[mode]}")  # fmt: skip
        if mode == "mask":
            ax2.plot(thresholds, [(u <= t).sum() for t in thresholds], color=COLORS[obj])
    ax1.set_ylim(0, 1.03)
    ax1.set_ylabel("fraction agreeing")
    ax1.tick_params(labelbottom=False)
    ax1.legend(loc="lower left", handlelength=2.2, borderaxespad=0.2)
    ax2.set_ylabel("instances")
    ax2.set_xlabel("threshold of the unavailable fraction")
    for a in (ax1, ax2):
        a.set_xlim(0, 1)
        a.grid(color=GRID, lw=0.4)
    panel_label(ax1, "(b)", x=-0.12)
    save(fig, out_dir, "itodd_agreement")


# --- convergence ------------------------------------------------------------------------
def fig_convergence(out_dir):
    D = diameters()
    synth = json.loads((HERE / "output_data_space_gn/results.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.1), layout="constrained")
    ax = axes[0]
    for obj in NAMES:
        runs = [v for k, v in synth.items() if k.startswith(f"{obj}/")]
        for key, ls in (("r_zero", "-"), ("y_zero", "--")):
            h = np.mean([[s[0] for s in v[key]] for v in runs], axis=0)
            ax.plot(np.arange(len(h)), 100 * h, color=COLORS[obj], ls=ls)
    ax.set_xlim(0, 30)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Gauss–Newton step")
    ax.set_ylabel("synthetic views solved (%)")
    # colours: objects; line styles: coordinates of the iteration (between the curves)
    handles = [plt.Line2D([], [], color=COLORS[o]) for o in NAMES] + [
        plt.Line2D([], [], color=MUTED, ls=ls) for ls in ("-", "--")]  # fmt: skip
    labels = [*NAMES.values(), "in $r$", "in $y$"]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.17, 0.95), ncols=2,
              handlelength=1.8, columnspacing=0.6, fontsize=6.5)  # fmt: skip
    ax.grid(color=GRID, lw=0.4)
    panel_label(ax, "(a)", x=-0.12)
    trajectories = {obj: json.loads((HERE / f"output_gn_trajectories/obj{obj}_mask_val.json")
                                    .read_text()) for obj in NAMES}  # fmt: skip
    for obj, ts in trajectories.items():
        for i, t in enumerate(ts):
            m = 100 * np.array(t["mssd_gt_mm"]) / D[obj]
            axes[1].plot(np.arange(len(m)), m, color=COLORS[obj], lw=0.9, alpha=0.85,
                         marker=MARKERS[obj], ms=2.2, label=NAMES[obj] if i == 0 else None)  # fmt: skip
            res = np.array(t["residual"])
            axes[2].plot(np.arange(1, len(res) + 1), res, color=COLORS[obj], lw=0.9,
                         alpha=0.85, label=NAMES[obj] if i == 0 else None)  # fmt: skip
    axes[1].axhline(5, color=MUTED, lw=0.7, ls=":")
    axes[1].text(20, 5 * 1.15, "strictest BOP threshold", ha="right", va="bottom",
                 fontsize=6.5, color=INK2)  # fmt: skip
    axes[1].set_yscale("log")
    axes[1].set_ylabel("MSSD / diameter (%)")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("residual in $r$")
    for ax, label in zip(axes[1:], ("(b)", "(c)")):
        ax.set_xlim(0, 20)
        ax.set_xlabel("Gauss–Newton step")
        ax.grid(color=GRID, lw=0.4)
        ax.legend(loc="upper right")
        panel_label(ax, label, x=-0.12)
    save(fig, out_dir, "itodd_convergence")


# --- growth of the curriculum domain against n_p -----------------------------------------
def read_csv(file):
    with open(file, newline="") as f:
        return list(csv.DictReader(f))


def r_curve(file, max_iter=None, offset=0):
    rows = read_csv(file)
    it = np.array([int(r["iter"]) for r in rows]) + offset
    R = np.array([float(r["R"]) for r in rows])
    keep = slice(None) if max_iter is None else it - offset <= max_iter
    return it[keep], R[keep]


def growth_runs():
    """(label, n_p, colour, line style, iterations, R) of the runs of README Sections
    12-14: ChartY0 with the Fourier-Bessel encoder and the whole-domain trimmer for
    n_p = 2 (64 layers), 3 (256) and 4 (256 layers, extended to 384 at iteration 25000),
    and the other five charts of the connector (384 layers)."""
    y0_4d = [r_curve(HERE / "output_y0_fb/fbtrim_full/history.csv", 25000),
             r_curve(HERE / "output_y0_full/l384/history.csv", 7500, offset=25000)]  # fmt: skip
    y0_4d = (np.concatenate([c[0] for c in y0_4d]), np.concatenate([c[1] for c in y0_4d]))
    runs = [("Y0, $n_p = 2$", 2, SERIES[2], ":",
             *r_curve(HERE / "output_y0_fb/fbtrim_bg20_l64/history.csv")),
            ("Y0, $n_p = 3$", 3, SERIES[1], "--",
             *r_curve(HERE / "output_y0_fb/fbtrim_bg20_d123/history.csv")),
            ("Y0, $n_p = 4$", 4, SERIES[0], "-", *y0_4d)]  # fmt: skip
    for chart, ls in zip(("Y1", "Y2", "Z0", "Z1", "Z2"), ("--", ":", "-", "--", ":")):
        colour = SERIES[0] if chart[0] == "Y" else MUTED
        runs.append((chart, 4, colour, ls,
                     *r_curve(HERE / f"output_charts_fb/{chart}/history.csv")))  # fmt: skip
    return runs


def fig_scaling(out_dir):
    runs = growth_runs()
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2), layout="constrained")
    for ax, sel, title in ((axes[0], lambda lab: lab.startswith("Y0"), "chart Y0"),
                           (axes[1], lambda lab: "n_p" not in lab or "= 4" in lab,
                            "all charts, $n_p = 4$")):  # fmt: skip
        for label, n_p, colour, ls, it, R in runs:
            if sel(label):
                ax.plot(it / 1000, R, color=colour, ls=ls,
                        label=label if ax is axes[0] else label.split(",")[0])  # fmt: skip
        ax.axhline(0.86, color=MUTED, lw=0.6, ls=":")
        ax.set_xlim(0, 33)
        ax.set_ylim(0, 0.92)
        ax.set_xlabel("iteration ($10^3$)")
        ax.set_title(title)
        ax.grid(color=GRID, lw=0.4)
    axes[0].set_ylabel("curriculum range $R$")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower right", ncols=2, columnspacing=0.8)
    # iterations per growth step of R by 0.2 against the volume of the shell it adds,
    # both relative to the first step (0.2 to 0.4)
    ax = axes[2]
    edges = (0.2, 0.4, 0.6, 0.8)
    for label, n_p, colour, ls, it, R in runs:
        reach = [it[np.nonzero(R >= e - 1e-9)[0][0]] for e in edges]
        iters = np.diff(reach).astype(float)
        vol = np.array([(2 * b) ** n_p - (2 * a) ** n_p for a, b in itertools.pairwise(edges)])
        ax.plot(vol / vol[0], iters / iters[0], color=colour, ls=ls, marker="o", ms=2.5)
    ax.plot([1, 12], [1, 12], color=INK2, lw=0.7, ls=(0, (4, 3)))
    ax.text(5.2, 7.2, "cost $\\propto$ volume", rotation=38, fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.9, 13)
    ax.set_ylim(0.5, 13)
    ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    ax.set_yticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_xlabel("volume of the added shell (rel.)")
    ax.set_ylabel("iterations per step (rel.)")
    ax.set_title("steps of $R$ by 0.2")
    ax.grid(color=GRID, lw=0.4)
    for ax, label in zip(axes, ("(a)", "(b)", "(c)")):
        panel_label(ax, label, x=-0.12)
    save(fig, out_dir, "itodd_scaling")


# --- momentum study ---------------------------------------------------------------------


def fig_momentum(out_dir):
    base = HERE / "output_momentum"
    cases = [("l384", "ChartY0, $R$ held at 0.86", 9500),
             ("y2", "ChartY2, $R$ growing", 24500)]  # fmt: skip
    variants = [("ctrl", r"control ($\beta_1 = 0.99$)", "-"),
                ("b09", r"$\beta_1 = 0.9$", "--"), ("reset", "momentum reset", "-.")]  # fmt: skip
    keys = [("neg_dets", "samples with\n" + r"$\det J < 0$"), ("g_max", "largest\ngradient norm")]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.2), sharex="col", layout="constrained")
    for col, (case, title, start) in enumerate(cases):
        series = [("before the start", MUTED, "-",
                   [r for r in read_csv(base / f"replay_{case}" / "history.csv")
                    if start - 3000 <= int(r["iter"]) <= start])]  # fmt: skip
        for k, (v, label, ls) in enumerate(variants):
            series.append((label, SERIES[k], ls,
                           read_csv(base / f"{case}_{v}" / "history.csv")))  # fmt: skip
        for label, color, ls, rows in series:
            it = np.array([int(r["iter"]) for r in rows]) / 1000
            for row, (key, _) in enumerate(keys):
                val = np.array([float(r[key]) for r in rows])
                if key == "neg_dets":
                    val = np.maximum(val, 0.5)  # zero drawn at the bottom of the axis
                axes[row, col].plot(it, val, color=color, ls=ls, marker=".", ms=3,
                                    label=label)  # fmt: skip
        axes[0, col].set_title(title)
        for row, (key, ylabel) in enumerate(keys):
            ax = axes[row, col]
            ax.set_yscale("log")
            ax.axvline(start / 1000, color=MUTED, lw=0.6, ls=":")
            ax.grid(axis="y", color=GRID, lw=0.4)
            if col == 0:
                ax.set_ylabel(ylabel)
        axes[0, col].set_ylim(0.4, 3e6)
        axes[1, col].set_xlabel(r"iteration ($10^3$; end of each stride of 500)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, handlelength=2.6)
    save(fig, out_dir, "itodd_momentum")


# --- figures that render ----------------------------------------------------------------
def crop_box(masks, shape, margin=40, aspect=4 / 3):
    """(x0, x1, y0, y1) around the union of the masks, widened to the aspect w / h."""
    union = np.any(masks, axis=0)
    rows, cols = np.nonzero(union)
    y0, y1 = rows.min() - margin, rows.max() + margin
    x0, x1 = cols.min() - margin, cols.max() + margin
    h, w = y1 - y0, x1 - x0
    if w / h < aspect:
        grow = aspect * h - w
        x0, x1 = x0 - grow / 2, x1 + grow / 2
    else:
        grow = w / aspect - h
        y0, y1 = y0 - grow / 2, y1 + grow / 2
    H, W = shape
    dx = max(0, -x0) - max(0, x1 - W)
    dy = max(0, -y0) - max(0, y1 - H)
    x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
    return (int(max(0, x0)), int(min(W, x1)), int(max(0, y0)), int(min(H, y1)))


def fig_scenes(out_dir):
    import torch
    from matplotlib import patheffects

    from lidar_fit import LidarFit
    from run_m3 import CASES
    from scene_demo import DEPTH_RANGE, GREEN, MAGENTA, MISMATCH_RANGE, RESULTS, scene

    rows = [(26, "mask", 643, "(a) validation image 643: 5 tee connectors, amodal masks"),
            (26, "visib", 655, "(b) test image 655: 10 tee connectors, visible masks only"),
            (12, "mask", 244, "(c) test image 244: 11 cylinders, amodal masks"),
            (26, "mask", 667, ("(d) test image 667: 5 tee connectors among objects without "
                               "models, amodal masks"))]  # fmt: skip
    titles = ["sensor depth", "solution (rendered)", "difference",
              "reconstruction", "gray image (not used)"]  # fmt: skip
    viridis = plt.get_cmap("viridis").copy()
    viridis.set_bad("black")
    rdbu = plt.get_cmap("RdBu").copy()
    rdbu.set_bad("#d9d9d6")
    fig, axes = plt.subplots(len(rows), 5, figsize=(6.5, 1.25 * len(rows)),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.22})  # fmt: skip

    def contour(ax, m, color, lw=0.5, ls="solid", zorder=2):
        if m.any():
            cs = ax.contour(m, levels=[0.5], colors=[color], linewidths=lw,
                            linestyles=[ls], zorder=zorder)  # fmt: skip
            cs.set_path_effects([patheffects.withStroke(linewidth=lw + 0.6,
                                                        foreground="white", alpha=0.7)])  # fmt: skip

    for row, (obj, mode, im, label) in zip(axes, rows):
        cfg, run = RESULTS[obj][mode]
        results = json.loads((HERE / run / "results.json").read_text())
        lf = LidarFit(HERE / cfg)
        fits = [(lf, r) for r in results if r["im_id"] == im]
        split = fits[0][1]["split"]
        path = next(p for s, p, i, _ in CASES[obj] if i == im)
        d = scene(fits, split, path, im, mode, np.random.default_rng(7))
        x0, x1, y0, y1 = crop_box(d["masks"] + d["silhouettes"], d["depth"].shape)
        s = np.s_[y0:y1, x0:x1]
        depth = np.where(d["depth"] > 0, d["depth"], np.nan)
        diff = np.where(d["depth"] > 0, d["solution"] - d["depth"], np.nan)
        kw = {"interpolation": "antialiased", "interpolation_stage": "rgba"}
        row[0].imshow(depth[s], cmap=viridis, vmin=DEPTH_RANGE[0], vmax=DEPTH_RANGE[1], **kw)
        for m in d["masks"] if mode == "mask" else d["visibs"]:
            contour(row[0], m[s], GREEN if mode == "mask" else MAGENTA)
        im1 = row[1].imshow(d["solution"][s], cmap=viridis, vmin=DEPTH_RANGE[0],
                            vmax=DEPTH_RANGE[1], **kw)  # fmt: skip
        im2 = row[2].imshow(diff[s], cmap=rdbu, vmin=MISMATCH_RANGE[0],
                            vmax=MISMATCH_RANGE[1], **kw)  # fmt: skip
        for m in d["masks"]:  # the whole amodal mask, below the solutions (all versions)
            contour(row[2], m[s], GREEN)
        for m in d["silhouettes"]:
            for ax in row[1:3]:
                contour(ax, m[s], "black", lw=0.45, ls=(0, (3, 2)), zorder=10)
        row[3].imshow(d["shaded"][s], cmap="gray", vmin=0, vmax=1, **kw)
        row[4].imshow(d["gray"][s], cmap="gray", vmin=0, vmax=255, **kw)
        for ax in row:
            ax.set_axis_off()
        row[0].text(0, 1.04, label, transform=row[0].transAxes, ha="left", va="bottom",
                    fontsize=7.5)  # fmt: skip
        del d, lf
        gc.collect()
        torch.cuda.empty_cache()
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=7.5, pad=16)
    bottom = axes[-1, 0].get_position()  # colour bars below the last row
    cax1 = fig.add_axes([0.13, bottom.y0 - 0.042, 0.28, 0.011])
    cax2 = fig.add_axes([0.44, bottom.y0 - 0.042, 0.14, 0.011])
    fig.colorbar(im1, cax=cax1, orientation="horizontal", label="depth (mm)")
    fig.colorbar(im2, cax=cax2, orientation="horizontal", label="difference (mm)")
    for cax in (cax1, cax2):
        cax.tick_params(labelsize=6.5, length=2)
        cax.xaxis.label.set_size(7)
    save(fig, out_dir, "itodd_scenes")


def fig_vortices(out_dir):
    import torch
    from matplotlib.collections import LineCollection

    from figures_m2 import plane, windings
    from train_chart import chart_stencil
    from wrinkle_map import r_and_det as r_det
    from wrinkle_map import render

    plt.rcParams.update(PAPER_RC)
    det_cmap = LinearSegmentedColormap.from_list("det", ["#d03b3b", "#f0efec", "#2a78d6"])
    wind_cmap = LinearSegmentedColormap.from_list(
        "w", ["#d03b3b", "#eda100", "#f0efec", "#2a78d6", "#1c5cab"], N=5
    )

    def mesh(ax, r, G, step=2, alpha=0.55, lw=0.35):
        rr = r.reshape(G, G, 2)[::step, ::step]
        segs = [rr[i] for i in range(len(rr))] + [rr[:, j] for j in range(rr.shape[1])]
        ax.add_collection(LineCollection(segs, linewidths=lw, colors=INK, alpha=alpha))
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["left", "bottom"]].set_visible(False)

    G = 64
    st = chart_stencil("Y0", 26, kernel="flat", n_grid=8, dp=0.025, dims=[1, 2])
    runs = [("c40", "final.twc", 0.72, "64 layers"), ("c40_l256", "best.twc", 0.86, "256 layers")]
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0), layout="constrained")
    wmaps = []
    for col, (run, file, box, label), (top, bottom) in zip((0, 2), runs, ("ad", "cf")):
        twc = HERE / "output_y0_reduced" / run / file
        _, p = plane(box, G)
        y, d = render(st, p, batch=128)
        r, _ = r_det(twc, y, d)
        mesh(axes[0, col], r, G)
        axes[0, col].set_title(f"({top}) image of a grid, {label}")
        w = windings(twc, st, box)
        wmaps.append(w)
        print(f"{run}: winding number != 1 for {100 * (w != 1).mean():.2f} % of the circles")
        imw = axes[1, col].imshow(w, origin="lower", extent=(-box, box, -box, box),
                                  cmap=wind_cmap, vmin=-1.5, vmax=3.5)  # fmt: skip
        axes[1, col].set_title(f"({bottom}) winding numbers, {label}")
        axes[1, col].set_xlabel("azimuth $p_2$")
        axes[1, col].set_ylabel("elevation $p_1$")
    fig.colorbar(imw, ax=axes[1, 2], shrink=0.7, aspect=15, ticks=[-1, 0, 1, 2, 3],
                 label="winding number")  # fmt: skip
    # zoom on a branch point (winding number 2) of the 64-layer twincher
    twc, box, w = HERE / "output_y0_reduced/c40/final.twc", runs[0][2], wmaps[0]
    Gc = len(w)
    i, j = np.argwhere(w == 2)[0]
    c = ((np.array([i, j]) + 0.5) / Gc * 2 - 1) * box
    rho, h, Gz = 0.75 * 2 * box / Gc, 0.045, 31
    xs = np.linspace(-h, h, Gz)
    E, F = np.meshgrid(c[0] + xs, c[1] + xs, indexing="ij")
    y, d = render(st, torch.as_tensor(np.stack([E.ravel(), F.ravel()], 1)), batch=128)
    r, det = r_det(twc, y, d)
    t = np.linspace(0, 2 * np.pi, 241)
    pc = torch.as_tensor(np.stack([c[0] + rho * np.sin(t), c[1] + rho * np.cos(t)], 1))
    y, d = render(st, pc, batch=128)
    rc, _ = r_det(twc, y, d)
    ax = axes[0, 1]
    mesh(ax, r, Gz, step=1, alpha=0.3, lw=0.3)
    ax.plot(rc[:, 0], rc[:, 1], color=SERIES[1], lw=1.1)
    for k in (0, 60, 120, 180):
        ax.annotate("", rc[k + 5], rc[k], arrowprops={"arrowstyle": "-|>", "lw": 1.1,
                    "color": SERIES[1], "mutation_scale": 9})  # fmt: skip
    ax.set_title("(b) zoom near a branch point")
    ax = axes[1, 1]
    imd = ax.imshow(det.reshape(Gz, Gz), origin="lower", cmap=det_cmap,
                    norm=TwoSlopeNorm(vcenter=0.0, vmin=-0.5, vmax=1.5),
                    extent=(c[1] - h, c[1] + h, c[0] - h, c[0] + h))  # fmt: skip
    ax.plot(c[1] + rho * np.cos(t), c[0] + rho * np.sin(t), color=SERIES[1], lw=1.1)
    ax.set_title(r"(e) $\det J$ near the branch point")
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel("azimuth $p_2$")
    ax.set_ylabel("elevation $p_1$")
    fig.colorbar(imd, ax=ax, shrink=0.7, aspect=15, ticks=[-0.5, 0, 0.5, 1, 1.5])
    save(fig, out_dir, "itodd_vortices")
    del st
    torch.cuda.empty_cache()


FIGURES = {"scenes": fig_scenes, "agreement": fig_agreement, "convergence": fig_convergence,
           "scaling": fig_scaling, "vortices": fig_vortices, "momentum": fig_momentum}  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("figures", nargs="*", choices=[[], *FIGURES])
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--format", choices=["pdf", "png"], default="pdf",
                        help="png: for the README (e.g. --out-dir figures)")  # fmt: skip
    args = parser.parse_args()
    plt.rcParams["savefig.format"] = args.format
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.figures or FIGURES:
        FIGURES[name](args.out_dir)


if __name__ == "__main__":
    main()
