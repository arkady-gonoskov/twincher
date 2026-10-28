"""Signed depth difference dZ = Z_solution - Z_sensor of a LidarFit run: systematic or noise?

Recomputes, from the saved depth maps of the solutions (<split>/<im>_<inst>_depth.tif)
and the BOP data, the robust statistics of LidarFit.depth_comparison on the interior
available pixels (plane offset and tilt, scatter around the plane), also for the
ground-truth poses (val), and plots for the good solutions (silhouette IoU >= 0.95):
the pooled distribution of dZ and of its residual around each instance's plane, and the
plane offsets and scatters per instance. Reference: sensor noise on flat table patches,
70-90 um (measured 2026-10-09).

Usage:
    python analyze_depth_diff.py RUN_DIR [--config lidarfit_obj12_m1.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_fit import LidarFit, read_image

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "lidarfit_obj12_m1.json")
    args = parser.parse_args()

    fit = LidarFit(args.config)
    st = fit.charts[0].stencil
    results = json.loads((args.run_dir / "results.json").read_text())
    rows, pooled, pooled_res = [], [], []
    for r in results:
        if r["status"] != "solved":
            continue
        path, name = Path(r["path"]), f"{r['im_id']:06d}_{r['inst_id']:06d}"
        cam = json.loads((path / "scene_camera.json").read_text())[str(r["im_id"])]
        depth = read_image(path / f"depth/{r['im_id']:06d}.tif") * cam.get(
            "depth_scale", 1.0
        )
        mask = read_image(path / f"mask/{name}.png", np.uint8) > 0
        visib = read_image(path / f"mask_visib/{name}.png", np.uint8) > 0
        z_sol = read_image(args.run_dir / r["split"] / f"{name}_depth.tif")
        available = visib & (depth > 0)
        d = fit.depth_comparison(z_sol, depth, mask, available)
        row = {"split": r["split"], "im": r["im_id"], "inst": r["inst_id"], **d}
        gt_file = path / "scene_gt.json"
        if gt_file.exists():
            gt = json.loads(gt_file.read_text())[str(r["im_id"])][r["inst_id"]]
            K = np.asarray(cam["cam_K"]).reshape(3, 3)
            z_gt = st.render_pinhole(
                gt["cam_R_m2c"], gt["cam_t_m2c"], K, *depth.shape[::-1]
            )
            row["gt"] = fit.depth_comparison(z_gt.cpu().numpy(), depth, mask, available)
        rows.append(row)
        if d["silhouette_iou"] >= 0.95 and "plane_offset_mm" in d:
            interior = available & (z_sol > 0)
            dz = (z_sol - depth)[interior]
            pooled.append(np.clip(dz, -5, 5))
            pooled_res.append(np.clip(dz - d["plane_offset_mm"], -5, 5))

    good = [x for x in rows if x["silhouette_iou"] >= 0.95 and "plane_offset_mm" in x]
    print(
        f"{len(good)} good solutions of {len(rows)} (interior pixels, robust plane fit):"
    )
    for key, label in [("plane_offset_mm", "plane offset (mm)"),
                       ("plane_tilt_mm_per_100px", "tilt (mm / 100 px)"),
                       ("scatter_robust_mm", "scatter (mm)"),
                       ("outlier_fraction", "outliers")]:  # fmt: skip
        v = np.array([x[key] for x in good])
        print(f"  {label:20s}: median {np.median(v):+.3f}, 10 % {np.quantile(v, 0.1):+.3f}, "
              f"90 % {np.quantile(v, 0.9):+.3f}")  # fmt: skip
    gts = [x["gt"] for x in rows if "gt" in x]
    for x in rows:
        if "gt" in x:
            g = x["gt"]
            print(f"  val {x['im']}/{x['inst']}: solution offset {x['plane_offset_mm']:+.2f} mm, "
                  f"tilt {x['plane_tilt_mm_per_100px']:.2f}, scatter {x['scatter_robust_mm']:.2f} | "
                  f"ground truth offset {g['plane_offset_mm']:+.2f} mm, tilt "
                  f"{g['plane_tilt_mm_per_100px']:.2f}, scatter {g['scatter_robust_mm']:.2f}")  # fmt: skip
    (args.run_dir / "depth_diff.json").write_text(json.dumps(rows, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink2, grid = "#fcfcfb", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.6), layout="constrained", facecolor=surface
    )
    bins = np.linspace(-3, 3, 121)
    axes[0].hist(np.concatenate(pooled), bins=bins, density=True, histtype="step", lw=1.5,
                 color="#2a78d6", label="dZ")  # fmt: skip
    axes[0].hist(np.concatenate(pooled_res), bins=bins, density=True, histtype="step",
                 lw=1.5, color="#eb6834", label="dZ - plane offset of the instance")  # fmt: skip
    axes[0].set_xlabel(
        "Z solution - Z sensor (mm), interior pixels of the good solutions"
    )
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, loc="upper left")
    off = np.array([x["plane_offset_mm"] for x in good])
    sc = np.array([x["scatter_robust_mm"] for x in good])
    axes[1].scatter(off, sc, s=24, color="#2a78d6", label="solutions")
    if gts:
        axes[1].scatter([g["plane_offset_mm"] for g in gts], [g["scatter_robust_mm"] for g in gts],
                        s=40, marker="x", color="#eb6834", label="ground-truth poses (val)")  # fmt: skip
    axes[1].axhspan(0.07, 0.09, color=grid, label="sensor noise (flat table)")
    axes[1].axvline(0, color=ink2, lw=0.8, ls=":")
    axes[1].set_xlabel("plane offset per instance (mm)")
    axes[1].set_ylabel("scatter around the plane (mm)")
    axes[1].legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(color=grid, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Signed depth difference of the solutions", x=0.01, ha="left")
    fig.savefig(args.run_dir / "depth_diff.png", dpi=110, facecolor=surface)
    plt.close(fig)
    print(f"written {args.run_dir / 'depth_diff.png'}")


if __name__ == "__main__":
    main()
