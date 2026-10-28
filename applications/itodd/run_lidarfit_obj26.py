"""LidarFit on the connector instances (obj 26) of ITODD: val with ground truth, test_all.

Images: val 643 (5 instances, one cut by the image border), 658 (1); test_all 644-657
and 673 (connectors only), 659-672 (connectors among other objects; the masks of
test_all exist only for the connectors, the targets of BOP). All masks of these images.

Writes OUT_DIR/results.json and, per instance, OUT_DIR/<split>/<im>_<inst>.png and the
depth map of the solution (<im>_<inst>_depth.tif); prints a table and a summary (val:
pose errors; test: silhouette IoU and depth differences, by group of images).

Usage:
    python run_lidarfit_obj26.py [--config lidarfit_obj26.json] [--out-dir DIR]
                                 [--images 643 644 ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from itodd_data import TEST, VAL
from lidar_fit import LidarFit

HERE = Path(__file__).resolve().parent
CASES = [  # split, scene directory, images with connectors (obj 26)
    ("val", VAL, [643, 658]),
    ("test", TEST,
     list(range(644, 658)) + list(range(659, 674))),
]  # fmt: skip
GROUPS = {"connectors only": set(range(644, 658)) | {673},
          "with other objects": set(range(659, 673))}  # fmt: skip
HEADER = ("split   im inst unknown chart  D(mm)  elev   azim   roll  misfit  IoU  |dZ|  "
          "plane  scat | t(mm) rot(deg) MSSD(mm)  time")  # fmt: skip


def row_text(split, im, inst, r):
    head = f"{split:5s} {im:4d} {inst:3d}  "
    if r["status"] != "solved":
        return head + r["status"]
    v, d = r["view"], r["depth"]
    text = (f"{head}{r['unknown_fraction']:6.1%}  {r['chart']:4s} {v['D_mm']:6.1f} "
            f"{v['elevation_deg']:+6.1f} {v['azimuth_deg']:+6.1f} {v['roll_deg']:+6.1f} "
            f"{r['misfit']:.4f} {d['silhouette_iou']:.3f} "
            f"{d.get('depth_diff_mean_mm', np.nan):5.2f} "
            f"{d.get('plane_offset_mm', np.nan):+5.2f} {d.get('scatter_robust_mm', np.nan):5.2f}")  # fmt: skip
    if "errors" in r:
        e = r["errors"]
        text += f" | {e['t_error_mm']:5.2f} {e['rot_error_deg']:8.2f} {e['mssd_mm']:8.2f}"
    else:
        text += " |" + " " * 24
    return text + f"  {r['time_s']:4.1f} s"


def summary(results, diameter):
    lines = []
    val = [r for r in results if r["split"] == "val" and r["status"] == "solved"]
    if val:
        mssd = np.array([r["errors"]["mssd_mm"] for r in val])
        lines.append(f"val: {len(val)} solved; MSSD median {np.median(mssd):.2f} mm, max "
                     f"{mssd.max():.2f} mm; < 5 % of the diameter "
                     f"{np.sum(mssd < 0.05 * diameter)} of {len(val)}")  # fmt: skip
    for group, images in GROUPS.items():
        rs = [r for r in results if r["split"] == "test" and r["im_id"] in images]
        ok = [r for r in rs if r["status"] == "solved"]
        if not rs:
            continue
        iou = np.array([r["depth"]["silhouette_iou"] for r in ok])
        good = [r for r in ok if r["depth"]["silhouette_iou"] >= 0.95]
        dz = [r["depth"]["depth_diff_mean_mm"] for r in good
              if "depth_diff_mean_mm" in r["depth"]]  # fmt: skip
        lines.append(f"test, {group}: {len(rs)} instances, {len(good)} good (silhouette "
                     f"IoU >= 0.95); IoU median {np.median(iou):.3f}; mean |dZ| of the "
                     f"good ones: median {np.median(dz):.2f} mm")  # fmt: skip
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=HERE / "lidarfit_obj26.json")
    parser.add_argument("--out-dir", type=Path, default=HERE / "output_lidarfit_obj26")
    parser.add_argument("--images", type=int, nargs="+")
    args = parser.parse_args()
    fit = LidarFit(args.config)
    results = []
    print(HEADER, flush=True)
    for split, path, images in CASES:
        for im in images:
            if args.images and im not in args.images:
                continue
            for mask_file in sorted((path / "mask").glob(f"{im:06d}_*.png")):
                inst = int(mask_file.stem.split("_")[1])
                r = fit.process_instance(path, im, inst, out_dir=args.out_dir / split)
                r["split"] = split
                results.append(r)
                print(row_text(split, im, inst, r), flush=True)
                (args.out_dir / "results.json").write_text(json.dumps(results, indent=1))
    print("\n".join(summary(results, fit.diameter)))


if __name__ == "__main__":
    main()
