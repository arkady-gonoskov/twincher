"""Stage 5: LidarFit on the cylinder instances of ITODD (val with ground truth, test_all).

Writes OUT_DIR/results.json and, per instance, OUT_DIR/<split>/<im>_<inst>.png and the
depth map of the solution (<im>_<inst>_depth.tif), and prints a table.

Usage:
    python run_lidarfit.py [--config lidarfit_obj12_m1.json] [--out-dir DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from itodd_data import TEST, VAL
from lidar_fit import LidarFit

HERE = Path(__file__).resolve().parent
CASES = [  # split, scene directory, images with cylinders (obj 12)
    ("val", VAL, [230, 233]),
    ("test", TEST, [231, 232] + list(range(234, 245))),
]


def summary_line(r):
    v, d = r["view"], r["depth"]
    line = (f"{r['unknown_fraction']:5.2f} {r['chart'][:7]:7s} {r['attempt']} "
            f"{v['D_mm']:6.1f} {v['elevation_deg']:+5.1f} {v['roll_deg']:+6.1f} "
            f"{r['misfit']:6.3f} {d.get('depth_diff_mean_mm', float('nan')):6.2f} "
            f"{d.get('depth_bias_mm', float('nan')):+6.2f} {d.get('scatter_robust_mm', float('nan')):5.2f} "
            f"{d['silhouette_iou']:5.3f}")  # fmt: skip
    if "errors" in r:
        e = r["errors"]
        line += (
            f" | {e['t_error_mm']:5.2f} {e['axis_error_deg']:5.2f} {e['mssd_mm']:5.2f}"
        )
    return line


HEADER = ("split   im inst unknown chart  att  D(mm)  elev   roll  misfit |dZ|  medZ  scat "
          "  IoU  | t err  axis  MSSD")  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=HERE / "lidarfit_obj12_m1.json")
    parser.add_argument("--out-dir", type=Path, default=HERE / "output_lidarfit")
    parser.add_argument(
        "--attempts",
        help="override the solver attempts, e.g. newton,plain (all from p = 0)",
    )
    parser.add_argument("--centre", choices=["com", "dist_y", "com_track"])
    parser.add_argument(
        "--edge-cases", choices=["on", "off"], help="override the configuration"
    )
    parser.add_argument("--centre-silhouette", choices=["fused", "mask"])
    parser.add_argument("--observation", choices=["mask", "visib"])
    parser.add_argument("--carve-mm", type=float)
    parser.add_argument("--dist-grid", choices=["own", "curr"])
    parser.add_argument("--fill-visib", choices=["on", "off"])
    parser.add_argument("--fill-target", choices=["on", "off"])
    args = parser.parse_args()

    fit = LidarFit(args.config)
    if args.attempts:
        fit.attempts = [{"newton": a == "newton"} for a in args.attempts.split(",")]
    if args.centre:
        fit.centre = args.centre
    if args.edge_cases:
        fit.edge_cases = args.edge_cases == "on"
    if args.centre_silhouette:
        fit.centre_silhouette = args.centre_silhouette
    for key in ("observation", "carve_mm", "dist_grid"):
        if getattr(args, key) is not None:
            setattr(fit, key, getattr(args, key))
    if args.fill_visib:
        fit.fill_visib = args.fill_visib == "on"
    if args.fill_target:
        fit.fill_target = args.fill_target == "on"
    print(f"charts: {[c.name for c in fit.charts]}; attempts: {fit.attempts}; "
          f"centre: {fit.centre} ({fit.centre_silhouette}, dist grid {fit.dist_grid}); "
          f"edge cases: {fit.edge_cases}; observation: {fit.observation} "
          f"(carve {fit.carve_mm} mm, fill {fit.fill_visib}, in the target "
          f"{fit.fill_target})")  # fmt: skip
    results = []
    print(HEADER)
    for split, path, images in CASES:
        for im in images:
            for mask_file in sorted((path / "mask").glob(f"{im:06d}_*.png")):
                inst = int(mask_file.stem.split("_")[1])
                r = fit.process_instance(path, im, inst, out_dir=args.out_dir / split)
                r["split"] = split
                results.append(r)
                head = f"{split:5s} {im:4d} {inst:3d}  "
                if r["status"] != "solved":
                    print(head + r["status"])
                else:
                    print(head + summary_line(r), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=1))

    solved = [r for r in results if r["status"] == "solved"]
    for split in ("val", "test"):
        rs = [r for r in solved if r["split"] == split]
        dz = np.array([r["depth"].get("depth_diff_mean_mm", np.nan) for r in rs])
        iou = np.array([r["depth"]["silhouette_iou"] for r in rs])
        n_all = sum(r["split"] == split for r in results)
        print(f"{split}: {len(rs)} of {n_all} solved; mean |dZ| per instance: median "
              f"{np.nanmedian(dz):.2f} mm, max {np.nanmax(dz):.2f} mm; silhouette IoU: median "
              f"{np.median(iou):.3f}, min {iou.min():.3f}; time per instance "
              f"{np.mean([r['time_s'] for r in rs]):.2f} s")  # fmt: skip
    print(f"results in {args.out_dir}")


if __name__ == "__main__":
    main()
