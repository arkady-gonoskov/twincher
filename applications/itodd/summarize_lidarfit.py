"""Comparison of LidarFit runs (results.json of run_lidarfit.py).

Per run: instances solved, good (silhouette IoU >= 0.95 with the mask, in the frame),
separately for those inside the frame and those touching the border; charts chosen;
for the good solutions, the signed depth difference dZ = Z_solution - Z_sensor:
median, plane offset, plane tilt and the RMS scatter around the plane, compared with
the same quantities for the ground-truth poses (val).

With --chart-a REF_DIR, only the instances whose solution in REF_DIR (e.g. the
two-chart run output_lidarfit_m1) has a roll within +-MAX_ROLL deg after the flip symmetry,
i.e. lies in chart A with a margin (for runs with chart A only).

Usage:
    python summarize_lidarfit.py LABEL=DIR [LABEL=DIR ...] [--chart-a REF_DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def stats(values):
    v = np.asarray(values, float)
    return f"median {np.median(v):+.2f}, 10-90 % {np.quantile(v, 0.1):+.2f} .. {np.quantile(v, 0.9):+.2f}"


MAX_ROLL = 70.0  # deg; chart A: +-73 deg


def in_chart_a(r):
    roll = r["view"]["roll_deg"]
    if abs(roll) > 90:  # flip symmetry: (e, psi) ~ (-e, psi - 180 deg)
        roll -= 180 if roll > 0 else -180
    return abs(roll) <= MAX_ROLL


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", metavar="LABEL=DIR")
    parser.add_argument("--chart-a", type=Path, metavar="REF_DIR")
    args = parser.parse_args()
    keep = None
    if args.chart_a:
        ref = json.loads((args.chart_a / "results.json").read_text())
        keep = {(r["split"], r["im_id"], r["inst_id"]) for r in ref
                if r["status"] == "solved" and in_chart_a(r)}  # fmt: skip
        print(f"instances within chart A (|roll| <= {MAX_ROLL} deg in {args.chart_a}): "
              f"{len(keep)} of {len(ref)}")  # fmt: skip
    runs = {}
    for arg in args.runs:
        label, d = arg.split("=")
        res = json.loads((Path(d) / "results.json").read_text())
        if keep is not None:
            res = [r for r in res if (r["split"], r["im_id"], r["inst_id"]) in keep]
        runs[label] = res
    for label, res in runs.items():
        solved = [r for r in res if r["status"] == "solved"]
        print(f"\n=== {label}: {len(solved)} of {len(res)} solved")
        for split in ("val", "test"):
            for border in (False, True):
                rs = [
                    r
                    for r in solved
                    if r["split"] == split and r["touches_border"] == border
                ]
                if not rs:
                    continue
                good = [r for r in rs if r["depth"]["silhouette_iou"] >= 0.95]
                where = "touching the border" if border else "inside the frame"
                charts = {}
                for r in rs:
                    charts[r["chart"]] = charts.get(r["chart"], 0) + 1
                line = (
                    f"  {split} {where}: good {len(good)} of {len(rs)}; charts {charts}"
                )
                if split == "val":
                    m = [r["errors"]["mssd_mm"] for r in rs]
                    line += f"; MSSD median {np.median(m):.2f} mm, max {max(m):.2f} mm"
                print(line)
        good = [
            r
            for r in solved
            if r["depth"]["silhouette_iou"] >= 0.95
            and "scatter_robust_mm" in r["depth"]
        ]
        if good:
            d = [r["depth"] for r in good]
            print(
                f"  signed dZ = Z_solution - Z_sensor of the {len(good)} good solutions (mm):"
            )
            print(
                f"    median dZ per instance:  {stats([x['depth_bias_mm'] for x in d])}"
            )
            print(
                f"    plane offset:            {stats([x['plane_offset_mm'] for x in d])}"
            )
            print(
                f"    plane tilt (mm / 100 px): {stats([x['plane_tilt_mm_per_100px'] for x in d])}"
            )
            print(
                f"    scatter around plane:    {stats([x['scatter_robust_mm'] for x in d])}"
            )
            print(
                f"    outliers (> 3 sigma):    {stats([x['outlier_fraction'] for x in d])}"
            )
            gt = [
                r["depth_gt"]
                for r in good
                if "depth_gt" in r and "scatter_robust_mm" in r["depth_gt"]
            ]
            if gt:
                print(f"    ground-truth poses (val, {len(gt)}): median dZ "
                      f"{stats([x['depth_bias_mm'] for x in gt])}; scatter "
                      f"{stats([x['scatter_robust_mm'] for x in gt])}")  # fmt: skip


if __name__ == "__main__":
    main()
