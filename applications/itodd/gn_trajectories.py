"""Gauss-Newton trajectories of LidarFit on ITODD instances (for the paper).

Runs LidarFit with a configuration of README Part III (Section 23.3) on the instances of
run_m3.py and records, for the solution that LidarFit selects (chart and attempt with
the smallest misfit), the pose after each Gauss-Newton step: p and the stencil centre c
when the step starts (c as aligned by the centre tracking for that p), and the final p.
Per step:
- "mssd_final_mm": MSSD (with the symmetries of the model) between the pose of the step
  and the final pose (no ground truth needed);
- "mssd_gt_mm": MSSD to the ground truth (validation images only);
- "residual": the norm of the residual of the solver in the representation space after
  the step (R_norm of the solver).
The final pose is checked against results.json of run_m3.py (same configuration). An
existing output file is resumed (instances already in it are skipped).
Instances already in --out are skipped, so that an interrupted run can be resumed.

Usage:
    python gn_trajectories.py --object 26 --config lidarfit_obj26_m3_mask.json \\
        --results output_m3/obj26_mask/results.json --out output_gn_trajectories/obj26_mask.json \\
        [--split val]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_fit import LidarFit
from run_m3 import instances


class RecordingFit(LidarFit):
    """LidarFit that keeps, for each solve() call, the iterates (p, c) and the residuals."""

    def solve(self, chart, obs, Q, attempt):
        steps, residuals = [], []
        refine, step = self.refine_com, chart.solver.step

        def refine_com(obs_, st, Q_, c, map_curr, ob):
            c_new, ob_new = refine(obs_, st, Q_, c, map_curr, ob)
            steps.append((chart.solver.p.double().clone(), c_new.copy()))
            return c_new, ob_new

        def step_rec(*args, **kwargs):
            out = step(*args, **kwargs)
            residuals.append(float(chart.solver.R_norm[0]))
            return out

        self.refine_com, chart.solver.step = refine_com, step_rec
        try:
            p, c, ob = super().solve(chart, obs, Q, attempt)
        finally:  # back to the methods of the classes
            del self.refine_com
            del chart.solver.step
        self.records.append({"Q": Q, "steps": steps + [(p.clone(), c.copy())],
                             "residuals": residuals})  # fmt: skip
        return p, c, ob


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--object", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"])
    args = parser.parse_args()
    if args.config.stem.endswith("visib"):
        raise ValueError("written for the observation 'mask' (centre tracking com_track)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stored = {(r["im_id"], r["inst_id"]): r
              for r in json.loads(args.results.read_text())}  # fmt: skip
    lf = RecordingFit(args.config)
    out = json.loads(args.out.read_text()) if args.out.exists() else []
    done = {(t["im_id"], t["inst_id"]) for t in out}
    for split, path, im, inst in instances(args.object):
        if (args.split and split != args.split) or (im, inst) in done:
            continue
        lf.records = []
        data = lf.load_instance(path, im, inst)
        r = lf.process(data)
        if r["status"] != "solved":
            continue
        k = [i for i, t in enumerate(r["trials"])
             if t["chart"] == r["chart"] and t["attempt"] == r["attempt"]][0]  # fmt: skip
        rec = lf.records[k]
        chart = next(ch for ch in lf.charts if ch.name == r["chart"])
        R_f, t_f = np.array(r["R_m2c"]).reshape(3, 3), np.array(r["t_m2c_mm"])
        traj = {"split": split, "im_id": im, "inst_id": inst, "chart": r["chart"],
                "attempt": r["attempt"], "residual": rec["residuals"],
                "mssd_final_mm": [], "mssd_gt_mm": []}  # fmt: skip
        for p, c in rec["steps"]:
            R, t = lf.pose(chart.stencil, p, c, None, rec["Q"])
            traj["mssd_final_mm"].append(lf.pose_errors(R, t, R_f, t_f)["mssd_mm"])
            if data["gt"] is not None:
                traj["mssd_gt_mm"].append(lf.pose_errors(R, t, *data["gt"])["mssd_mm"])
        ref = stored.get((im, inst))
        traj["matches_results"] = bool(
            ref is not None and ref["chart"] == r["chart"]
            and np.allclose(ref["p"], r["p"], atol=1e-5)
        )  # fmt: skip
        out.append(traj)
        print(f"{split} {im} {inst}: {r['chart']}, MSSD to the final pose "
              f"{traj['mssd_final_mm'][0]:.1f} -> {traj['mssd_final_mm'][-2]:.3f} mm"
              + (f", to the ground truth {traj['mssd_gt_mm'][-1]:.2f} mm"
                 if traj["mssd_gt_mm"] else "")
              + ("" if traj["matches_results"] else "  (differs from results.json)"),
              flush=True)  # fmt: skip
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
