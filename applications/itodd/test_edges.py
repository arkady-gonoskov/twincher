"""Edge cases with ground truth: val instances cut by a virtual image border.

For each val cylinder instance, the visible frame is reduced from one side (left, right,
top, bottom) so that a fraction f of the extent of its mask lies outside; the instance is
then solved with edge_cases on (pixels outside the frame unknown) in each mode (MODES:
centre mode and observation, e.g. com_track_mask: com_track with the mask as silhouette;
*_visib: observation without the mask); the uncut instances are solved too (frame =
whole image). Pose errors against scene_gt.json; success: MSSD < 5 % (BOP's smallest
threshold) and 10 % of the diameter.

Outputs in OUT_DIR: edges.json, edges.png (MSSD against the cut fraction of the mask),
per case <im>_<inst>_<mode>_<side><f>.png.

Usage:
    python test_edges.py [--config lidarfit_obj12_m1.json]
                         [--modes com_track_visib_fill,com_track_mask,dist_y_visib]
                         [--out-dir DIR] [--plot-only]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from itodd_data import VAL
from lidar_fit import LidarFit

HERE = Path(__file__).resolve().parent
INSTANCES = [(230, 0), (233, 0), (233, 1), (233, 2)]
SIDES = ["left", "right", "top", "bottom"]
FRACTIONS = [0.1, 0.25, 0.4]
MODES = {  # name: settings of LidarFit (others as in DEFAULTS)
    "com": {"centre": "com"},
    "dist_y": {"centre": "dist_y"},
    "com_track": {"centre": "com_track"},
    "com_track_mask": {"centre": "com_track", "centre_silhouette": "mask"},
    "dist_y_visib": {"centre": "dist_y", "observation": "visib"},
    "dist_y_curr_visib": {
        "centre": "dist_y",
        "dist_grid": "curr",
        "observation": "visib",
    },
    "com_track_visib": {"centre": "com_track", "observation": "visib"},
    "dist_y_visib_fill": {
        "centre": "dist_y",
        "observation": "visib",
        "fill_visib": True,
    },
    "com_track_visib_fill": {
        "centre": "com_track",
        "observation": "visib",
        "fill_visib": True,
    },
}
DEFAULTS = {
    "centre_silhouette": "fused",
    "observation": "mask",
    "dist_grid": "own",
    "fill_visib": False,
    "fill_target": False,
}
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated slots 1-5
MARKERS = ["o", "D", "^", "v", "s"]


def cut_frame(mask, side, f, width, height):
    """Frame (x0, x1, y0, y1) that leaves the fraction f of the mask extent outside."""
    rows, cols = np.nonzero(mask)
    x_lo, x_hi, y_lo, y_hi = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
    frame = [0, width, 0, height]
    if side == "left":
        frame[0] = round(float(x_lo + f * (x_hi - x_lo)))
    elif side == "right":
        frame[1] = round(float(x_hi - f * (x_hi - x_lo)))
    elif side == "top":
        frame[2] = round(float(y_lo + f * (y_hi - y_lo)))
    else:
        frame[3] = round(float(y_hi - f * (y_hi - y_lo)))
    return tuple(frame)


def plot(file_name, rows, diameter):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(dict.fromkeys(r["mode"] for r in rows))  # in the order of the run
    if len(modes) > len(COLOURS):
        raise ValueError(f"at most {len(COLOURS)} modes per plot")
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained", facecolor="#fcfcfb")
    for mode, col, marker in zip(modes, COLOURS, MARKERS):
        rs = [r for r in rows if r["mode"] == mode and r["status"] == "solved"]
        x = [1 - r["mask_in_frame"] for r in rs]
        y = [r["errors"]["mssd_mm"] for r in rs]
        ax.scatter(x, y, s=28, color=col, marker=marker, alpha=0.8, label=mode)
    for level, label in [(0.05, "5 % of the diameter"), (0.1, "10 %")]:
        ax.axhline(level * diameter, color="#52514e", lw=0.8, ls=":")
        ax.annotate(label, (1.0, level * diameter), xycoords=("axes fraction", "data"),
                    xytext=(-4, 3), textcoords="offset points", ha="right",
                    color="#52514e", fontsize=8)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel("fraction of the mask outside the frame")
    ax.set_ylabel("MSSD (mm)")
    ax.set_title("val cylinders cut by a virtual image border", loc="left")
    ax.legend(
        frameon=False, title="centre mode", loc="upper left", bbox_to_anchor=(1.01, 1)
    )
    ax.set_facecolor("#fcfcfb")
    ax.grid(color="#e4e3df", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(file_name, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=HERE / "lidarfit_obj12_m1.json")
    parser.add_argument("--out-dir", type=Path, default=HERE / "output_edges")
    parser.add_argument(
        "--modes", default="com_track_visib_fill,com_track_mask,dist_y_visib"
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="redraw from edges.json"
    )
    args = parser.parse_args()
    if args.plot_only:
        rows = json.loads((args.out_dir / "edges.json").read_text())
        plot(args.out_dir / "edges.png", rows, 107.703)
        return

    fit = LidarFit(args.config)
    fit.edge_cases = True
    rows = []
    print(
        " im inst side   f   in-frame | mode               MSSD   t err  axis  | misfit"
    )
    for im, inst in INSTANCES:
        mask = np.asarray(Image.open(VAL / f"mask/{im:06d}_{inst:06d}.png")) > 0
        height, width = mask.shape
        cases = [("none", 0.0, None)] + [
            (s, f, cut_frame(mask, s, f, width, height))
            for s in SIDES
            for f in FRACTIONS
        ]
        for side, f, frame in cases:
            for mode in args.modes.split(","):
                for key, value in {**DEFAULTS, **MODES[mode]}.items():
                    setattr(fit, key, value)
                tag = f"_{mode}_{side}{f:g}"
                r = fit.process_instance(VAL, im, inst, out_dir=args.out_dir, frame=frame,
                                         tag=tag)  # fmt: skip
                r.update({"mode": mode, "side": side, "cut": f})
                rows.append(r)
                e = r["errors"]
                print(f"{im:4d} {inst:2d} {side:6s} {f:4.2f} {r['mask_in_frame']:6.3f}  | "
                      f"{mode:17s} {e['mssd_mm']:6.2f} {e['t_error_mm']:6.2f} "
                      f"{e['axis_error_deg']:5.2f} | {r['misfit']:.4f}", flush=True)  # fmt: skip
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "edges.json").write_text(json.dumps(rows, indent=1))
    plot(args.out_dir / "edges.png", rows, fit.diameter)

    print("\nsuccess rates (MSSD < 5 % / 10 % of the diameter) and median MSSD:")
    for mode in args.modes.split(","):
        for label, sel in [("uncut", lambda r: r["side"] == "none"),
                           ("cut", lambda r: r["side"] != "none")]:  # fmt: skip
            m = np.array(
                [r["errors"]["mssd_mm"] for r in rows if r["mode"] == mode and sel(r)]
            )
            print(f"  {mode:17s} {label:5s}: n {len(m):3d}, < 5 %: {np.mean(m < 0.05 * fit.diameter):.2f}, "
                  f"< 10 %: {np.mean(m < 0.1 * fit.diameter):.2f}, median {np.median(m):.2f} mm")  # fmt: skip
    print(f"outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
