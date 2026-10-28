"""Train an hs twincher for one chart of an object without symmetries (n_p = 4).

Charts (directions from the model origin to the camera, image up axis, roll):
- group Z: s = (-cos(e) cos(phi), cos(e) sin(phi), sin(e)), image up: model +Z;
- group Y: s = (cos(e) cos(phi), sin(e), cos(e) sin(phi)), image up: model +Y;
with p0 -> D (dist_range), p1 -> e in +-ELEVATION, p2 -> phi in +-AZIMUTH, p3 -> roll in
ROLL_CENTRE[k] +- ROLL_HALF for chart <group><k>, e.g. Z0 (roll 0 +- 75 deg).

Data (--data): "grid": y and dy/dp of DepthStencil (kernel "tent") on an n_grid^4 grid of
p, multilinear interpolation by the learner (16^4 is too coarse for this object: views
20 deg apart are blended); "pool": exact y and dy/dp at --pool-size random points
(SamplePool), from which the batches are drawn. Both cached in OUT_DIR. Only the hs loss
(embedding phase), that of twincher 0.1.0 (train_cylinder.HS_0_1_0).
Stops when R = 1 and det J > clearance on all training batches for --done-strides
strides, or after --iters iterations.

Checks with the exact stencil (rendering and finite differences, no interpolation):
every --exact-stride iterations, det J on N_EXACT fixed random points of [-1, 1]^4
(history_exact.csv); at the end, the solver (start p = 0, N_SOLVE_ITER iterations) on
N_SOLVE random synthetic views (solver.json).

Outputs in OUT_DIR: history.csv and r_history.png (per stride: R, min det J and negative
determinants of the training batches), history_exact.csv, checkpoints, final.twc,
solver.json, learner_tm.log.

Usage:
    python -u train_chart.py Z0 [OUT_DIR] [--obj 26] [--n-grid 16] [--n-c 2048] [--lr 1e-3]
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import twincher

from depth_stencil import DepthStencil
from itodd_data import MODELS
from train_cylinder import HS_0_1_0, grid_data, plot_history

HERE = Path(__file__).resolve().parent
ELEVATION, AZIMUTH = math.pi / 3, 5 * math.pi / 6
ROLL_CENTRE, ROLL_HALF = [0.0, 2 * math.pi / 3, -2 * math.pi / 3], 5 * math.pi / 12
N_EXACT, N_SOLVE, N_SOLVE_ITER, SOLVE_TOL = 4096, 1024, 20, 1e-3


class SamplePool:
    """Exact samples (p, y, dy/dp) of a stencil at random points of [-1, 1]^n_p: 3/4
    uniform, 1/4 uniform scaled by s ~ U(0.2, 1) (denser near the centre, for the early
    curriculum). Used as the stencil of a Learner (set_stencil): compute(p, y, dy_dp)
    replaces the requested points (uniform in [-R, R]^n_p) by random pool samples in the
    same box, writing their p into the learner's batch (read by the hs loss only for
    its ramp at the border of the curriculum range)."""

    def __init__(self, st, n, file, seed=1, batch=512):
        self.n_p, self.n_y, self.name = st.n_p, st.n_y, st.name
        if file.exists():
            data = torch.load(file)
            p, y, dy_dp = data["p"], data["y"], data["dy_dp"]
            print(f"pool from {file}")
        else:
            t0 = time.perf_counter()
            g = torch.Generator().manual_seed(seed)
            p = 2 * torch.rand((n, st.n_p), generator=g, dtype=torch.float64) - 1
            scale = 0.2 + 0.8 * torch.rand(
                (n // 4, 1), generator=g, dtype=torch.float64
            )
            p[n - n // 4 :] *= scale
            y = torch.empty((n, st.n_y), device=st.device)
            dy_dp = torch.empty((n, st.n_y, st.n_p), device=st.device)
            for i in range(0, n, batch):
                st.compute(p[i : i + batch], y[i : i + batch], dy_dp[i : i + batch])
            p = p.float()
            torch.save({"p": p, "y": y.cpu(), "dy_dp": dy_dp.cpu()}, file)
            print(f"pool of {n} samples computed in {time.perf_counter() - t0:.0f} s, "
                  f"saved to {file}")  # fmt: skip
        order = p.abs().amax(dim=1).argsort()
        self.p = p[order].to(st.device)
        self.y = y.to(st.device)[order.to(st.device)]
        self.dy_dp = dy_dp.to(st.device)[order.to(st.device)]
        self.norm = self.p.abs().amax(dim=1).contiguous()
        self.rng = torch.Generator(device=st.device).manual_seed(seed + 1)

    def compute(self, p, y, dy_dp):
        k = int(torch.searchsorted(self.norm, p.abs().max().float(), right=True))
        idx = torch.randint(
            0, max(k, 1), (len(p),), device=p.device, generator=self.rng
        )
        p.copy_(self.p[idx])
        y.copy_(self.y[idx])
        dy_dp.copy_(self.dy_dp[idx])


class ReducedStencil:
    """A chart stencil restricted to the components dims of p and scaled: the full p
    has p[dims] = scale * p and 0 (the centre of the range) in the other components.
    dy/dp by central differences with the steps of the full stencil (the same physical
    steps for every scale)."""

    def __init__(self, st, dims, scale=1.0):
        if st.spectral is not None:
            raise ValueError("ReducedStencil does not support spectral stencils")
        self.st, self.dims, self.scale = st, [int(k) for k in dims], float(scale)
        self.n_p, self.n_y, self.device = len(self.dims), st.n_y, st.device
        self.name = f"{st.name}_d{''.join(map(str, self.dims))}_s{self.scale:g}"

    def __getattr__(self, name):  # settings of the full stencil (metadata)
        return getattr(self.__dict__["st"], name)

    def full(self, p):
        p = torch.as_tensor(p).to(self.device, torch.float64).reshape(-1, self.n_p)
        q = torch.zeros((len(p), self.st.n_p), dtype=torch.float64, device=self.device)
        q[:, self.dims] = self.scale * p
        return q

    def forward(self, p):
        return self.st.forward(self.full(p))

    def compute(self, p, y, dy_dp):
        q = self.full(p)
        dp = self.st.dp[self.dims]
        steps = torch.zeros((self.n_p, self.st.n_p), dtype=q.dtype, device=self.device)
        steps[range(self.n_p), self.dims] = dp
        q = torch.cat([q[None], q[None] + steps[:, None], q[None] - steps[:, None]])
        s, _ = self.st.forward(q.reshape(-1, self.st.n_p))
        s = s.view(1 + 2 * self.n_p, -1, self.n_y)
        y.copy_(s[0])
        diff = (s[1 : 1 + self.n_p] - s[1 + self.n_p :]) * (
            self.scale / (2 * dp[:, None, None])
        )
        dy_dp.copy_(diff.permute(1, 2, 0))

    def __call__(self, p, y):
        """y (n_y) for one p (n_p), numpy or torch (HSMonitorP2)."""
        y_t = self.forward(p)[0][0]
        if isinstance(y, np.ndarray):
            y[:] = y_t.cpu().numpy()
        else:
            y.copy_(y_t)


def chart_stencil(chart, obj, kernel="tent", margin=1.0, resolution=512, n_grid=16,
                  dp=0.05, dist_range=(638.38, 775.97), spectral=None,
                  spectral_scale=1.0, dims=None, scale=1.0, fb=None):  # fmt: skip
    """The stencil of a chart; with dims (components of p) a ReducedStencil; fb: options
    of the Fourier-Bessel stencil (depth_stencil.FB_DEFAULTS) or None."""
    st = _chart_stencil(chart, obj, kernel, margin, resolution, n_grid, dp,
                        dist_range, spectral, spectral_scale, fb)  # fmt: skip
    if dims is not None:
        st = ReducedStencil(st, dims, scale)
    return st


def _chart_stencil(chart, obj, kernel, margin, resolution, n_grid, dp, dist_range,
                   spectral, spectral_scale, fb=None):  # fmt: skip
    group, k = chart[0], int(chart[1:])
    c = ROLL_CENTRE[k]
    return DepthStencil(
        MODELS / f"obj_{obj:06d}.ply",
        n_grid=n_grid,
        resolution=resolution,
        kernel=kernel,
        spectral=spectral,
        spectral_scale=spectral_scale,
        fb=fb,
        dp=dp,
        dist_range=dist_range,
        elevation_range=(-ELEVATION, ELEVATION),
        azimuth_range=(-AZIMUTH, AZIMUTH),
        roll_range=(c - ROLL_HALF, c + ROLL_HALF),
        directions=group,
        margin=margin,
    )


def stencil_metadata(st, chart, obj, margin, data_grid):
    return {"model": f"itodd obj_{obj:06d}", "obj_id": obj, "chart": chart,
            "directions": st.directions, "n_grid": st.n_grid, "resolution": st.res,
            "kernel": st.kernel, "kernel_size": st.kernel_size, "ranges": st.ranges,
            "margin": margin, "s_amp": st.s_amp, "dp": st.dp[0].item(),
            "spectral": st.spectral, "spectral_scale": st.spectral_scale,
            "dims": getattr(st, "dims", None), "scale": getattr(st, "scale", 1.0),
            "fb": st.fb,
            "data_grid": data_grid}  # fmt: skip


def stencil_from_metadata(cfg):
    st = chart_stencil(cfg["chart"], cfg["obj_id"], kernel=cfg["kernel"],
                       margin=cfg["margin"], resolution=cfg["resolution"],
                       n_grid=cfg["n_grid"], dp=cfg["dp"],
                       dist_range=tuple(cfg["ranges"][0]),
                       spectral=cfg.get("spectral"),
                       spectral_scale=cfg.get("spectral_scale", 1.0),
                       dims=cfg.get("dims"), scale=cfg.get("scale", 1.0),
                       fb=cfg.get("fb"))  # fmt: skip
    if not np.allclose(st.ranges, cfg["ranges"]):
        raise ValueError("ranges of the stencil and of the metadata differ")
    return st


class ExactCheck:
    """det J of a twincher on fixed random points, with y and dy/dp of the stencil."""

    def __init__(self, st, n_s, n_l, seed=2026):
        g = torch.Generator().manual_seed(seed)
        p = 2 * torch.rand((N_EXACT, st.n_p), generator=g, dtype=torch.float64) - 1
        self.y = torch.empty((N_EXACT, st.n_y), device=st.device)
        self.dy_dp = torch.empty((N_EXACT, st.n_y, st.n_p), device=st.device)
        for i in range(0, N_EXACT, 512):
            st.compute(p[i : i + 512], self.y[i : i + 512], self.dy_dp[i : i + 512])
        self.n_p, self.n_y = st.n_p, st.n_y
        self.SH = twincher.Shuttle(data_type="torch", n_s=n_s, n_l=n_l, n_c=N_EXACT,
                                   n_p=st.n_p, n_m=st.n_p)  # fmt: skip

    def __call__(self, a, clearance):
        SH, n_y = self.SH, self.n_y
        SH.a.copy_(a)
        SH.tw_update()
        SH.s[:n_y].copy_(self.y.T)
        SH.s[n_y:].fill_(0)
        SH.Q.fill_(0)
        SH.Q[:n_y].copy_(self.dy_dp.permute(1, 2, 0))
        SH.forward_query()
        dets = torch.linalg.det(SH.Q[: self.n_p].permute(2, 1, 0))
        return {
            "exact_det_min": dets.min().item(),
            "exact_det_p01": torch.quantile(dets, 0.01).item(),
            "exact_frac_neg": (dets <= 0).float().mean().item(),
            "exact_frac_low": (dets <= clearance).float().mean().item(),
        }


def solver_test(twc, st, seed=7):
    """Solve for random synthetic views from p = 0 with the stencil as forward model."""
    solver = twincher.solver(str(twc), n_c=N_SOLVE)
    g = torch.Generator().manual_seed(seed)
    p_true = 2 * torch.rand((N_SOLVE, st.n_p), generator=g, dtype=torch.float64) - 1
    y_true, _ = st.forward(p_true)
    solver.start_solution(y_true)
    for _ in range(N_SOLVE_ITER):
        st.compute(solver.p, solver.y, solver.dy_dp)
        solver.step(y=solver.y, dy_dp=solver.dy_dp)
    err = (solver.p.double().cpu() - p_true).abs().max(dim=1).values.numpy()
    q = np.quantile(err, [0.5, 0.9, 0.99, 1.0])
    return {"success": float((err < SOLVE_TOL).mean()), "tolerance": SOLVE_TOL,
            "err_p50": q[0], "err_p90": q[1], "err_p99": q[2], "err_max": q[3],
            "fraction_err_above_0.1": float((err > 0.1).mean())}  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("chart", help="Z0, Z1, Z2, Y0, Y1 or Y2")
    parser.add_argument("out_dir", nargs="?", type=Path)
    parser.add_argument("--obj", type=int, default=26)
    parser.add_argument("--data", choices=["grid", "pool"], default="grid")
    parser.add_argument("--n-grid", type=int, default=16, help="grid points per p_k")
    parser.add_argument("--pool-size", type=int, default=2**18)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--n-s", type=int, default=256)
    parser.add_argument("--n-l", type=int, default=256)
    parser.add_argument("--n-c", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iters", type=int, default=30000, help="maximal iterations")
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument("--exact-stride", type=int, default=2500)
    parser.add_argument("--done-strides", type=int, default=4)
    parser.add_argument(
        "--init",
        type=Path,
        help="start from this twincher (.twc; may have fewer layers)",
    )
    args = parser.parse_args()
    out = args.out_dir or HERE / f"output_chart_{args.obj}_{args.chart}"
    out.mkdir(parents=True, exist_ok=True)

    st = chart_stencil(args.chart, args.obj, margin=args.margin)
    learner = twincher.Learner(arch="hs", n_s=args.n_s, n_l=args.n_l, lr=args.lr,
                               n_c=args.n_c, dir=out, output_stride=args.stride,
                               plot_progress=False)  # fmt: skip
    learner.modify_params({"loss_terms": {"hs_noise": None, "hs": HS_0_1_0}})
    tag = f"obj{args.obj}_{args.chart}"
    if args.data == "grid":
        p_grid = learner.get_p_request(n_p=st.n_p, n_grid=args.n_grid)
        name = f"data_{tag}_n{args.n_grid}_m{args.margin:g}_{st.kernel}.pt"
        y, dy_dp = grid_data(st, p_grid, out / name)
        learner.set_data(
            y, dy_dp, stencil_name=f"itodd_obj{args.obj}_depth_{args.chart}"
        )
        del y, dy_dp
    else:
        name = f"pool_{tag}_{args.pool_size}_m{args.margin:g}_{st.kernel}.pt"
        pool = SamplePool(st, args.pool_size, out / name)
        pool.name = f"itodd_obj{args.obj}_depth_{args.chart}"
        learner.set_stencil(pool)
    learner.init()
    learner.SH.metadata["stencil"] = stencil_metadata(
        st,
        args.chart,
        args.obj,
        args.margin,
        args.n_grid if args.data == "grid" else f"pool {args.pool_size}",
    )
    if args.init:  # copy the parameters into the learner's shuttle (curriculum anew)
        src = twincher.Shuttle()
        src.load(str(args.init), n_c=64, n_p=st.n_p, n_m=1)
        ranges = src.metadata.get("stencil", {}).get("ranges")
        if ranges is None or not np.allclose(ranges, st.ranges):
            raise ValueError(f"{args.init}: another chart")
        # a twincher with fewer layers fills the first layers (parameters are stored
        # layer by layer, the wiring of a layer depends only on its index); the
        # further layers keep zero twists (identity), so the map is unchanged
        if src.n_s != learner.n_s or src.n_l > learner.n_l:
            raise ValueError(f"{args.init}: n_s {src.n_s}, n_l {src.n_l}")
        learner.SH.a[: len(src.a)].copy_(src.a)
        learner.SH.tw_update()
        learner.SH.metadata["init"] = str(args.init)
        # check: the same output for the first samples of the data
        y0 = (pool.y if args.data == "pool" else learner.data_y)[:64]
        s_out = []
        for sh in (learner.SH, src):
            sh.s.fill_(0)
            sh.s[: st.n_y, : len(y0)].copy_(y0.T)
            sh.forward_inference()
            s_out.append(sh.s[:, : len(y0)].clone())
        diff = (s_out[0] - s_out[1]).abs().max().item()
        print(f"twincher from {args.init}: max |s_out difference| = {diff:.1e}")
        if diff > 1e-4:
            raise RuntimeError("the learner does not reproduce the loaded twincher")
        del src
    hs = next(term for term in learner.loss_terms if term.name == "hs")
    exact = ExactCheck(st, args.n_s, args.n_l)
    data = (
        f"grid {args.n_grid}^{st.n_p}"
        if args.data == "grid"
        else f"pool {args.pool_size}"
    )
    print(f"chart {args.chart}: ranges {np.round(st.ranges, 4).tolist()}, half width "
          f"{st.half_width:.4f}; n_s {args.n_s}, n_l {args.n_l}, n_c {args.n_c}, lr "
          f"{args.lr}, n_y {st.n_y}, n_p {st.n_p}, data {data}")  # fmt: skip

    rows, det_min, n_neg, done, t_train = [], float("inf"), 0, 0, 0.0
    with (
        open(out / "history.csv", "w", newline="") as f,
        open(out / "history_exact.csv", "w", newline="") as fe,
    ):
        writer = csv.DictWriter(
            f, ["iter", "time_s", "R", "det_min", "neg_dets", "loss"]
        )
        writer.writeheader()
        writer_e = None
        for it in range(args.iters + 1):
            t0 = time.perf_counter()
            learner.step()
            t_train += time.perf_counter() - t0
            det_min = min(det_min, hs.dets.min().item())
            n_neg += (hs.dets < 0).sum().item()
            if it % args.exact_stride == 0 and it > 0:
                e = {"iter": it, "R": round(hs.p_range, 3)}
                e.update(exact(learner.SH.a, hs.det_clearance))
                if writer_e is None:
                    writer_e = csv.DictWriter(fe, list(e))
                    writer_e.writeheader()
                writer_e.writerow(e)
                fe.flush()
                print(f"  exact data: min det {e['exact_det_min']:+.3f}, 1 % quantile "
                      f"{e['exact_det_p01']:+.3f}, det <= 0: {100 * e['exact_frac_neg']:.3f} %, "
                      f"det <= clearance: {100 * e['exact_frac_low']:.3f} %")  # fmt: skip
                learner.SH.save(str(out / f"iter_{it}.twc"))
            if it % args.stride != 0:
                continue
            row = {"iter": it, "time_s": round(t_train, 1), "R": round(hs.p_range, 3),
                   "det_min": round(det_min, 4), "neg_dets": n_neg,
                   "loss": float(learner.loss)}  # fmt: skip
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(f"{it:6d} {t_train:8.1f} s  R {hs.p_range:.3f}  min det {det_min:+.3f}  "
                  f"neg {n_neg:6d}  loss {row['loss']:.3e}  {learner.state}", flush=True)  # fmt: skip
            plot_history(rows, out / "r_history.png", hs.det_margin, hs.det_clearance)
            done = done + 1 if hs.p_range >= 1 and det_min > hs.det_clearance else 0
            det_min, n_neg = float("inf"), 0
            if done >= args.done_strides:
                print(f"R = 1 with min det J > clearance for {done} strides: stop")
                break
    learner.time_monitor.save_report()
    learner.SH.save(str(out / "final.twc"))
    e = exact(learner.SH.a, hs.det_clearance)
    print(f"saved {out / 'final.twc'} after {it} iterations ({t_train:.0f} s); exact data: "
          f"min det {e['exact_det_min']:+.3f}, det <= 0: {100 * e['exact_frac_neg']:.3f} %, "
          f"det <= clearance: {100 * e['exact_frac_low']:.3f} %")  # fmt: skip
    del exact, learner
    torch.cuda.empty_cache()
    res = {
        "iters": it,
        "train_time_s": t_train,
        **e,
        **solver_test(out / "final.twc", st),
    }
    (out / "solver.json").write_text(json.dumps(res, indent=1))
    print(f"solver test ({N_SOLVE} views, start p = 0, {N_SOLVE_ITER} iterations): "
          f"success (max |dp| < {SOLVE_TOL:g}) {100 * res['success']:.1f} %, error 99 % "
          f"{res['err_p99']:.1e}, |dp| > 0.1: {100 * res['fraction_err_above_0.1']:.1f} %")  # fmt: skip


if __name__ == "__main__":
    main()
