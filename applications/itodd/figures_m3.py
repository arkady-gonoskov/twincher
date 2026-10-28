"""Figures and tables of README Part III (metrics and results), from the results of
run_m3.py (output_m3/obj<id>_<mode>/results.json; pose_metrics.py for the definitions).

Instances cut by the image border (mask touching it) are left out of everything that
uses the unavailable fraction (the parts outside the image are not in the masks of
BOP, so their fraction is underestimated); they are tabulated separately.

- figures/m3_depth_mismatch.png: mean |Z_solution - Z_sensor| against the unavailable
  fraction of the object, one marker per instance (shape: object; filled: with the
  amodal mask, open: without it); horizontal lines: 5 % of the object diameter, the
  strictest MSSD threshold of BOP, as a scale;
- figures/m3_silhouette.png: 1 - IoU of the silhouette of the solution and the amodal
  mask, the same way;
- figures/m3_cumulative.png: over a threshold of the unavailable fraction (instances
  above it excluded): the fraction of instances whose solution agrees with the
  observation (silhouette IoU >= 0.95 and mean |dZ| <= 2 mm), and their number;
- tables (printed, Markdown): the validation instances with the pose errors of BOP, the
  agreement by bins of the unavailable fraction, and the instances cut by the border.

Usage:
    python figures_m3.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from itodd_data import MODELS_EVAL

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
RUNS = {(12, "mask"): "output_m3/obj12_tent_mask",
        (12, "visib"): "output_m3/obj12_tent_visib",
        (26, "mask"): "output_m3/obj26_mask", (26, "visib"): "output_m3/obj26_visib"}  # fmt: skip
NAMES = {12: "cylinder", 26: "tee connector"}
COLORS = {12: "#2a78d6", 26: "#eb6834"}
MARKERS = {12: "o", 26: "^"}
MODES = {"mask": "with the amodal mask", "visib": "without it"}
INFO = MODELS_EVAL / "models_info.json"
DIAMETERS = {obj: json.loads(INFO.read_text())[str(obj)]["diameter"] for obj in NAMES}
INK2, MUTED, GRID, SURFACE = "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
GOOD_IOU, GOOD_DZ = 0.95, 2.0
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})  # fmt: skip


def load():
    out = {}
    for key, run in RUNS.items():
        f = HERE / run / "results.json"
        if f.exists():
            out[key] = json.loads(f.read_text())
    return out


def inside(data):
    """The instances not cut by the image border."""
    return {k: [r for r in v if not r["touches_border"]] for k, v in data.items()}


def column(results, key, default=np.nan):
    return np.array([r["metrics"].get(key, default) for r in results], float)


def good(results):
    iou, dz = column(results, "silhouette_iou"), column(results, "depth_mismatch_mm")
    return (iou >= GOOD_IOU) & (dz <= GOOD_DZ)


def scatter(data, key, ylabel, file_name, transform=lambda v: v, ylim=None,
            scale_lines=False):  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    if scale_lines:  # 5 % of the diameter (strictest MSSD threshold of BOP)
        for obj, d in DIAMETERS.items():
            ax.axhline(0.05 * d, color=COLORS[obj], lw=1.0, ls="--", zorder=1)
            ax.text(0.005, 0.05 * d * 1.06, f"5 % of the diameter of the {NAMES[obj]}: "
                    f"{0.05 * d:.1f} mm", color=INK2, fontsize=8, ha="left",
                    va="bottom", zorder=4, bbox={"facecolor": SURFACE, "edgecolor": "none",
                                                 "alpha": 0.85, "pad": 1.0})  # fmt: skip
    for (obj, mode), results in data.items():
        x, y = column(results, "unavailable"), transform(column(results, key))
        filled = mode == "mask"
        ax.scatter(x, y, s=22, marker=MARKERS[obj], linewidths=1.0,
                   facecolors=COLORS[obj] if filled else "none", edgecolors=COLORS[obj],
                   alpha=0.85, label=f"{NAMES[obj]}, {MODES[mode]} ({len(x)})")  # fmt: skip
    ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(0, 1)
    ax.set_xlabel("unavailable fraction of the object (occluded or without depth)")
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, lw=0.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(FIG / file_name, dpi=130)
    plt.close(fig)
    print(f"figures/{file_name}")


def cumulative(data, file_name):
    thresholds = np.linspace(0.05, 1.0, 96)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True, layout="constrained",
                             height_ratios=[2, 1])  # fmt: skip
    for (obj, mode), results in data.items():
        u, ok = column(results, "unavailable"), good(results)
        frac = [ok[u <= th].mean() if (u <= th).any() else np.nan for th in thresholds]
        count = [(u <= th).sum() for th in thresholds]
        style = {"color": COLORS[obj], "ls": "-" if mode == "mask" else "--", "lw": 1.6,
                 "label": f"{NAMES[obj]}, {MODES[mode]}"}  # fmt: skip
        axes[0].plot(thresholds, frac, **style)
        if mode == "mask":
            axes[1].plot(thresholds, count, color=COLORS[obj], lw=1.6,
                         label=NAMES[obj])  # fmt: skip
    axes[0].set_ylabel(
        f"fraction with IoU >= {GOOD_IOU}\nand mean |dZ| <= {GOOD_DZ} mm"
    )
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].set_ylabel("instances included")
    axes[1].set_xlabel(
        "threshold of the unavailable fraction (instances above it excluded)"
    )
    axes[1].legend(fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(color=GRID, lw=0.5)
    fig.savefig(FIG / file_name, dpi=130)
    plt.close(fig)
    print(f"figures/{file_name}")


def tables(data, data_in):
    print("\n| Object | Mode | Instance | Unavailable | MSSD (mm) | MSPD (px) | e_VSD "
          "(tau 0.05..0.5) | AR |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|")
    for (obj, mode), results in data.items():
        for r in results:
            m = r["metrics"]
            if "score_ar" in m:
                vsd = f"{min(m['vsd']):.3f}..{max(m['vsd']):.3f}"
                print(f"| {NAMES[obj]} | {mode} | {r['im_id']}/{r['inst_id']} | "
                      f"{m['unavailable']:.0%} | {m['mssd_mm']:.2f} | {m['mspd_px']:.2f} | "
                      f"{vsd} | {m['score_ar']:.3f} |")  # fmt: skip
    bins = ((0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.75), (0.75, 1.01))
    data = data_in  # the bins: instances inside the image only
    print("\n| Object | Mode | " + " | ".join(f"{lo:.0%}-{min(hi, 1):.0%}" for lo, hi in bins)
          + " | all |")  # fmt: skip
    print("|---|---|" + "---|" * (len(bins) + 1))
    for (obj, mode), results in data.items():
        u, ok = column(results, "unavailable"), good(results)
        cells = [f"{ok[(u >= lo) & (u < hi)].sum()} of {((u >= lo) & (u < hi)).sum()}"
                 for lo, hi in bins]  # fmt: skip
        print(f"| {NAMES[obj]} | {mode} | " + " | ".join(cells)
              + f" | {ok.sum()} of {len(ok)} |")  # fmt: skip


def border_table(data):
    print("\n| Object | Mode | Cut by the border | Agreeing |")
    print("|---|---|---|---|")
    for (obj, mode), results in data.items():
        rs = [r for r in results if r["touches_border"]]
        ok = good(rs) if rs else np.zeros(0, bool)
        print(f"| {NAMES[obj]} | {mode} | {len(rs)} | {ok.sum()} of {len(rs)} |")


def main():
    data = load()
    data_in = inside(data)
    scatter(data_in, "depth_mismatch_mm", "mean |Z solution - Z sensor| (mm)",
            "m3_depth_mismatch.png", ylim=(0.1, 100), scale_lines=True)  # fmt: skip
    scatter(data_in, "silhouette_mismatch", "1 - IoU (silhouette of the solution, mask)",
            "m3_silhouette.png", transform=lambda v: np.maximum(v, 1e-3),
            ylim=(1e-3, 1.5))  # fmt: skip
    cumulative(data_in, "m3_cumulative.png")
    tables(data, data_in)
    border_table(data)


if __name__ == "__main__":
    main()
