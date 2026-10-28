"""Does the depth prefer a laterally shifted pose? (check of the cross-axis tilt of dZ)

For the good solutions of a LidarFit run (silhouette IoU >= 0.95), the pose is shifted
in the image plane perpendicular to the projected cylinder axis by s in [-2, 2] mm; for
each s the solution is rendered and dZ compared with the sensor on the interior
available pixels (robust plane fit of LidarFit.depth_comparison). Reported: the shift
with the smallest robust scatter, and the tilt and scatter there; for val also for the
ground-truth poses.

Usage:
    python lateral_check.py RUN_DIR [--config lidarfit_obj12_m1.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_fit import LidarFit, read_image

HERE = Path(__file__).resolve().parent
SHIFTS = np.arange(-2.0, 2.01, 0.1)


def scan(fit, st, R, t, K, depth, mask, available):
    """Robust statistics of dZ for the lateral shifts perpendicular to the axis."""
    axis = R[:, 2]  # symmetry axis in the camera frame
    perp = np.cross(axis, [0.0, 0.0, 1.0])  # in the image plane, perpendicular to it
    perp /= np.linalg.norm(perp)
    out = []
    for s in SHIFTS:
        z = st.render_pinhole(R, t + s * perp, K, depth.shape[1], depth.shape[0])
        out.append(fit.depth_comparison(z.cpu().numpy(), depth, mask, available))
    scatter = np.array([o.get("scatter_robust_mm", np.inf) for o in out])
    best = int(np.argmin(scatter))
    i0 = int(np.argmin(np.abs(SHIFTS)))
    return {
        "shift_mm": float(SHIFTS[best]),
        "scatter_0": out[i0]["scatter_robust_mm"],
        "tilt_0": out[i0]["plane_tilt_mm_per_100px"],
        "scatter_best": out[best]["scatter_robust_mm"],
        "tilt_best": out[best]["plane_tilt_mm_per_100px"],
        "iou_best": out[best]["silhouette_iou"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "lidarfit_obj12_m1.json")
    args = parser.parse_args()

    fit = LidarFit(args.config)
    st = fit.charts[0].stencil
    rows = []
    for r in json.loads((args.run_dir / "results.json").read_text()):
        if r["status"] != "solved" or r["depth"]["silhouette_iou"] < 0.95:
            continue
        path, name = Path(r["path"]), f"{r['im_id']:06d}_{r['inst_id']:06d}"
        cam = json.loads((path / "scene_camera.json").read_text())[str(r["im_id"])]
        K = np.asarray(cam["cam_K"]).reshape(3, 3)
        depth = read_image(path / f"depth/{r['im_id']:06d}.tif") * cam.get(
            "depth_scale", 1.0
        )
        mask = read_image(path / f"mask/{name}.png", np.uint8) > 0
        available = (read_image(path / f"mask_visib/{name}.png", np.uint8) > 0) & (
            depth > 0
        )
        R = np.asarray(r["R_m2c"]).reshape(3, 3)
        row = {"split": r["split"], "im": r["im_id"], "inst": r["inst_id"]}
        row["solution"] = scan(
            fit, st, R, np.asarray(r["t_m2c_mm"]), K, depth, mask, available
        )
        gt_file = path / "scene_gt.json"
        if gt_file.exists():
            gt = json.loads(gt_file.read_text())[str(r["im_id"])][r["inst_id"]]
            R_gt = np.asarray(gt["cam_R_m2c"]).reshape(3, 3)
            row["gt"] = scan(fit, st, R_gt, np.asarray(gt["cam_t_m2c"]), K, depth, mask,
                             available)  # fmt: skip
        rows.append(row)
        s = row["solution"]
        line = (f"{r['split']:4s} {r['im_id']} {r['inst_id']:2d}: best shift {s['shift_mm']:+.1f} mm; "
                f"tilt {s['tilt_0']:.2f} -> {s['tilt_best']:.2f} mm/100 px, scatter "
                f"{s['scatter_0']:.2f} -> {s['scatter_best']:.2f} mm, IoU {s['iou_best']:.3f}")  # fmt: skip
        if "gt" in row:
            g = row["gt"]
            line += f" | GT: shift {g['shift_mm']:+.1f}, tilt {g['tilt_0']:.2f} -> {g['tilt_best']:.2f}"
        print(line, flush=True)
    (args.run_dir / "lateral_check.json").write_text(json.dumps(rows, indent=1))
    for split in ("val", "test"):
        s = [x["solution"] for x in rows if x["split"] == split]
        if s:
            sh = np.abs([x["shift_mm"] for x in s])
            print(f"{split}: |best shift| median {np.median(sh):.2f} mm (90 % {np.quantile(sh, 0.9):.2f}); "
                  f"tilt median {np.median([x['tilt_0'] for x in s]):.2f} -> "
                  f"{np.median([x['tilt_best'] for x in s]):.2f} mm/100 px; scatter median "
                  f"{np.median([x['scatter_0'] for x in s]):.2f} -> "
                  f"{np.median([x['scatter_best'] for x in s]):.2f} mm")  # fmt: skip


if __name__ == "__main__":
    main()
