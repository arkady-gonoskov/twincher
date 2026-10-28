"""LidarFit on all ITODD instances of an object, with the metrics of README Part III.

Instances (BOP scene 000001 of val and test_all; the test masks exist only for the
target objects of BOP):
- obj 12 (cylinder): val 230, 233; test 231, 232, 234-244 (all masks of these images);
- obj 26 (tee connector): val 643, 658; test 644-657 and 673 (connectors only), 659-672
  (connectors among other objects), all masks of these images; and 735/3, 736/3,
  743/0-2, 744/0-2, 745/0-2 (among other objects; the masks of the other objects of 735
  and 736 belong to other targets).

Writes OUT_DIR/results.json (LidarFit.process plus "split" and "metrics", see
pose_metrics.py) and, with --plots, OUT_DIR/<split>/<im>_<inst>.png; prints one line
per instance and a summary.

Usage:
    python run_m3.py --object 26 --config lidarfit_obj26_m3_mask.json --out-dir DIR
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from itodd_data import TEST, VAL
from lidar_fit import LidarFit
from pose_metrics import instance_metrics

HERE = Path(__file__).resolve().parent
CASES = {  # object: [(split, scene directory, image, instances or None for all masks)]
    12: [("val", VAL, im, None) for im in (230, 233)]
    + [("test", TEST, im, None) for im in [231, 232, *range(234, 245)]],
    26: [("val", VAL, im, None) for im in (643, 658)]
    + [("test", TEST, im, None) for im in [*range(644, 658), *range(659, 674)]]
    + [("test", TEST, im, insts) for im, insts in ((735, [3]), (736, [3]),
                                                    (743, [0, 1, 2]), (744, [0, 1, 2]),
                                                    (745, [0, 1, 2]))],
}  # fmt: skip
ALONE = {26: {*range(644, 658), 673}}  # images with no other objects


def instances(obj):
    for split, path, im, insts in CASES[obj]:
        if insts is None:
            insts = [int(f.stem.split("_")[1])
                     for f in sorted((path / "mask").glob(f"{im:06d}_*.png"))]  # fmt: skip
        for inst in insts:
            yield split, path, im, inst


def line(r):
    m = r["metrics"]
    text = (f"{r['split']:5s} {r['im_id']:4d} {r['inst_id']:2d}  unavailable "
            f"{m['unavailable']:5.1%} (holes {m['holes']:5.1%}, occluded {m['occluded']:5.1%})"
            f"  {r['chart']:28.28s} IoU {m['silhouette_iou']:.3f}  |dZ| "
            f"{m.get('depth_mismatch_mm', np.nan):5.2f} mm")  # fmt: skip
    if "score_ar" in m:
        text += (f"  MSSD {m['mssd_mm']:5.2f} mm, MSPD {m['mspd_px']:5.2f} px, "
                 f"AR {m['score_ar']:.3f}")  # fmt: skip
    return text + f"  ({r['time_s']:.1f} s)"


def summary(obj, results):
    out = []
    for name, sel in (("val", lambda r: r["split"] == "val"),
                      ("test", lambda r: r["split"] == "test"),
                      ("test, alone", lambda r: r["split"] == "test"
                       and r["im_id"] in ALONE.get(obj, set())),
                      ("test, with other objects", lambda r: r["split"] == "test"
                       and obj in ALONE and r["im_id"] not in ALONE[obj])):  # fmt: skip
        rs = [r for r in results if sel(r)]
        if not rs:
            continue
        m = [r["metrics"] for r in rs]
        iou = np.array([x["silhouette_iou"] for x in m])
        dz = np.array([x.get("depth_mismatch_mm", np.nan) for x in m])
        text = (f"{name}: {len(rs)} instances; silhouette IoU >= 0.95: "
                f"{np.sum(iou >= 0.95)}; medians: IoU {np.median(iou):.3f}, |dZ| "
                f"{np.nanmedian(dz):.2f} mm")  # fmt: skip
        if all("score_ar" in x for x in m):
            text += (f"; AR {np.mean([x['score_ar'] for x in m]):.3f}, MSSD median "
                     f"{np.median([x['mssd_mm'] for x in m]):.2f} mm")  # fmt: skip
        out.append(text)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--object", type=int, required=True, choices=sorted(CASES))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--images", type=int, nargs="+")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lf = LidarFit(args.config)
    st = lf.charts[0].stencil
    results = []
    for split, path, im, inst in instances(args.object):
        if args.images and im not in args.images:
            continue
        t0 = time.perf_counter()
        data = lf.load_instance(path, im, inst)
        r = lf.process(data, out_dir=args.out_dir / split if args.plots else None)
        if r["status"] != "solved":
            print(f"{split} {im} {inst}: {r['status']}", flush=True)
            continue
        R = np.array(r["R_m2c"]).reshape(3, 3)
        r.update({"split": split,
                  "metrics": instance_metrics(lf, st, data, R, np.array(r["t_m2c_mm"]))})  # fmt: skip
        r["time_s"] = time.perf_counter() - t0
        results.append(r)
        print(line(r), flush=True)
        (args.out_dir / "results.json").write_text(json.dumps(results, indent=1))
    print("\n".join(summary(args.object, results)))


if __name__ == "__main__":
    main()
