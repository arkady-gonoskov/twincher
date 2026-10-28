"""LidarFit with the six charts of the connector (obj 26) on synthetic instances.

Each case: a ground-truth view p_true drawn uniformly in |p|_inf <= --box (0.86: the
box that the charts cover) of a random chart; the pose of the model in the stencil camera
of this view (looking at the model origin from the distance D). Variants:
- stencil: the depth map of the stencil square itself (as in training: the object
  centred by its centre of mass); every chart solves y_j = y_from_depth(z) from p = 0
  (plain hs steps), the solution with the smallest misfit (RMS of the y-tone difference
  of the depth maps) is kept: the twinchers and the choice of the chart, without camera
  geometry;
- centred: the ITODD camera (K of the validation images, 1280 x 960) with the model
  origin on its optical axis; depth = rendering, mask = mask_visib = depth > 0;
  LidarFit.process (virtual camera, resampling, com_track);
- anywhere: as centred, the camera turned so that the model origin appears at a random
  pixel at least --margin pixels from the image border;
- occluded: as anywhere, 2-3 ellipses (aspect 1:5, semi-major axis 10-150 pixels,
  log-uniform) centred on object pixels are occluded: removed from mask_visib (the
  observation "mask" treats them as unknown);
- cut: as anywhere, the image frame cuts off the fraction 0.1, 0.25 or 0.4 of the extent
  of the mask from a random side (pixels outside the frame unknown).
Errors against the ground truth: translation, rotation, MSSD (LidarFit.pose_errors);
whether the chosen chart contains p_true (|p|_inf <= 1 in that chart).

Outputs in --out-dir: <variant>.json (results per case), summary.txt, plots of the
--plots worst cases per variant.

Usage:
    python test_lidarfit_synth.py [--variants stencil centred anywhere occluded cut]
                                  [--n 60] [--config lidarfit_obj26.json]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from itodd_data import VAL
from lidar_fit import LidarFit, smallest_rotation
from test_edges import cut_frame

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1280, 960


def stencil_pose(st, p):
    """R, t (model -> stencil camera) of the view p (1, n_p) of a stencil."""
    _, E = st._frames(p.to(st.device, torch.float64))
    D = st.view(p.to(st.device, torch.float64))[0, 0].item()
    return E[0].double().cpu().numpy().T, np.array([0.0, 0.0, D])


def tone(st, z):
    y = st.s_amp * (2 * (z - st.z_lo) / (st.z_bg - st.z_lo) - 1)
    return torch.where(z > 0, y, st.s_amp)


def solve_stencil(lf, z, n_iter):
    """The variant "stencil": each chart solves the depth map z (1, res, res)."""
    trials = []
    for ci, chart in enumerate(lf.charts):
        st, solver = chart.stencil, chart.solver
        y_obs, _ = st.y_from_depth(z)
        solver.start_solution(y_obs)
        for _ in range(n_iter):
            y = torch.empty((1, st.n_y), device=st.device)
            dy = torch.empty((1, st.n_y, st.n_p), device=st.device)
            st.compute(solver.p.double(), y, dy)
            solver.step(y=y, dy_dp=dy)
            solver.p.clamp_(-1, 1)
        p = solver.p.double().clone()
        d = tone(st, st.depth_image(p)[0]) - tone(st, z[0])
        trials.append({"chart": ci, "p": p, "misfit": float(d.square().mean().sqrt())})
    return min(trials, key=lambda tr: tr["misfit"]), trials


def occlude(mask, rng, n_max=3, a_min=10.0, a_max=150.0, aspect=5.0):
    """Mask of 2-3 ellipses centred on pixels of mask."""
    rows, cols = np.nonzero(mask)
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    out = np.zeros_like(mask)
    for _ in range(rng.integers(2, n_max + 1)):
        k = rng.integers(len(rows))
        a = math.exp(rng.uniform(math.log(a_min), math.log(a_max)))
        b, th = a / aspect, rng.uniform(0, math.pi)
        dx, dy = xx - cols[k], yy - rows[k]
        u = (dx * math.cos(th) + dy * math.sin(th)) / a
        v = (-dx * math.sin(th) + dy * math.cos(th)) / b
        out |= u * u + v * v <= 1
    return out


def make_case(lf, rng, box, variant, K, margin):
    """The description of one case, its data (variant "stencil": the depth map of the
    stencil square; otherwise the input of LidarFit.process), the frame (or None) and
    the ground-truth pose (R, t)."""
    ci = int(rng.integers(len(lf.charts)))
    st = lf.charts[ci].stencil
    p_true = torch.as_tensor(rng.uniform(-box, box, size=(1, st.n_p)))
    R_s, t_s = stencil_pose(st, p_true)
    case = {"chart_true": lf.charts[ci].name, "p_true": p_true[0].tolist()}
    if variant == "stencil":
        return case, st.depth_image(p_true.to(st.device)), None, (R_s, t_s)
    if variant == "centred":
        u = np.array([K[0, 2], K[1, 2]])
    else:
        u = rng.uniform([margin, margin], [WIDTH - margin, HEIGHT - margin])
    d = [(u[0] - K[0, 2]) / K[0, 0], (u[1] - K[1, 2]) / K[1, 1], 1.0]
    Q = smallest_rotation([0.0, 0.0, 1.0], d)
    R, t = Q @ R_s, Q @ t_s
    depth = st.render_pinhole(R, t, K, WIDTH, HEIGHT).cpu().numpy().astype(np.float64)
    mask = depth > 0
    visib = mask.copy()
    frame = None
    if variant == "occluded":
        visib &= ~occlude(mask, rng)
        case["occluded_fraction"] = float(1 - visib.sum() / mask.sum())
    elif variant == "cut":
        side = ["left", "right", "top", "bottom"][rng.integers(4)]
        f = float(rng.choice([0.1, 0.25, 0.4]))
        frame = cut_frame(mask, side, f, WIDTH, HEIGHT)
        case.update({"cut_side": side, "cut_fraction": f})
    data = {"depth": depth, "K": K, "mask": mask, "mask_visib": visib, "gt": (R, t)}
    return case, data, frame, (R, t)


def summarize(variant, rows, diameter):
    mssd = np.array([r["errors"]["mssd_mm"] for r in rows])
    rot = np.array([r["errors"]["rot_error_deg"] for r in rows])
    t = np.array([r["errors"]["t_error_mm"] for r in rows])
    in_chart = np.mean([r["chart_contains_truth"] for r in rows])
    lines = [(f"{variant}: {len(rows)} cases; MSSD < 5 % of the diameter "
             f"{100 * np.mean(mssd < 0.05 * diameter):.1f} %, < 2 % "
             f"{100 * np.mean(mssd < 0.02 * diameter):.1f} %, < 1 % "
             f"{100 * np.mean(mssd < 0.01 * diameter):.1f} %; medians: MSSD "
             f"{np.median(mssd):.3f} mm, rotation {np.median(rot):.3f} deg, translation "
             f"{np.median(t):.3f} mm; chosen chart contains the truth "
             f"{100 * in_chart:.1f} %; time {np.mean([r['time_s'] for r in rows]):.1f} s")]  # fmt: skip
    if variant == "occluded":
        occ = np.array([r["occluded_fraction"] for r in rows])
        for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)):
            sel = (occ >= lo) & (occ < hi)
            if sel.any():
                lines.append(f"  occluded {lo:.0%}-{hi:.0%}: {sel.sum()} cases, MSSD < 5 % "
                             f"{100 * np.mean(mssd[sel] < 0.05 * diameter):.1f} %")  # fmt: skip
    if variant == "cut":
        cut = np.array([r["cut_fraction"] for r in rows])
        for f in (0.1, 0.25, 0.4):
            sel = cut == f
            if sel.any():
                lines.append(f"  cut {f:.0%}: {sel.sum()} cases, MSSD < 5 % "
                             f"{100 * np.mean(mssd[sel] < 0.05 * diameter):.1f} %")  # fmt: skip
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=HERE / "lidarfit_obj26.json")
    parser.add_argument("--variants", nargs="+",
                        default=["stencil", "centred", "anywhere", "occluded", "cut"])  # fmt: skip
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--box", type=float, default=0.86)
    parser.add_argument("--margin", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--plots", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=HERE / "output_lidarfit_synth")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lf = LidarFit(args.config)
    cam = json.loads((VAL / "scene_camera.json").read_text())["643"]
    K = np.asarray(cam["cam_K"], float).reshape(3, 3)
    summary = []
    for variant in args.variants:
        rng = np.random.default_rng(args.seed)
        rows, keep = [], []
        for i in range(args.n):
            case, data, frame, (R_gt, t_gt) = make_case(lf, rng, args.box, variant, K,
                                                         args.margin)  # fmt: skip
            t0 = time.perf_counter()
            if variant == "stencil":
                best, trials = solve_stencil(lf, data, lf.n_iter)
                chart = lf.charts[best["chart"]]
                R, t = stencil_pose(chart.stencil, best["p"])
                r = {"chart": chart.name, "p": best["p"][0].tolist(),
                     "misfit": best["misfit"],
                     "trials": [{"chart": lf.charts[tr["chart"]].name,
                                 "misfit": tr["misfit"]} for tr in trials],
                     "errors": lf.pose_errors(R, t, R_gt, t_gt)}  # fmt: skip
            else:
                data["name"] = f"{variant}_{i:03d}"
                data["info"] = {"label": f"{variant} case {i}"}
                r = lf.process(data, frame=frame)
                chart = next(c for c in lf.charts if c.name == r["chart"])
                keep.append((r, data, frame))
            r["time_s"] = time.perf_counter() - t0
            p_in = chart.p_of_pose(R_gt, t_gt)
            r.update({"case": i, **case, "p_true_in_chosen": p_in.tolist(),
                      "chart_contains_truth": bool(np.abs(p_in).max() <= 1.0)})  # fmt: skip
            rows.append(r)
            e = r["errors"]
            print(f"{variant} {i:3d} {case['chart_true']} -> {r['chart']}: MSSD "
                  f"{e['mssd_mm']:7.3f} mm, rotation {e['rot_error_deg']:7.3f} deg, t "
                  f"{e['t_error_mm']:7.3f} mm  ({r['time_s']:.1f} s)", flush=True)  # fmt: skip
        (args.out_dir / f"{variant}.json").write_text(json.dumps(rows, indent=1))
        lines = summarize(variant, rows, lf.diameter)
        print("\n".join(lines), flush=True)
        summary += lines
        worst = sorted(range(len(keep)), key=lambda k: -keep[k][0]["errors"]["mssd_mm"])
        for k in worst[: args.plots]:  # rerun with plots (deterministic)
            r, data, frame = keep[k]
            lf.process(data, out_dir=args.out_dir / "plots", frame=frame)
    (args.out_dir / "summary.txt").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
