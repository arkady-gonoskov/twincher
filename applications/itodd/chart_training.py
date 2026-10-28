"""Resumable training of one chart on an exact-sample pool or on grid data.

train(cfg, run_dir, log) trains an hs twincher for cfg["chart"] (train_chart.py: charts,
stencil, SamplePool) and can be interrupted at any time: a checkpoint with the complete
training state (twincher, Adam, random generators of the learner and the pool, damper
and curriculum) is written every stride, and a new call continues from it (bit-identical
to an uninterrupted run).

- Data (cfg["data"]): "pool" (SamplePool) or "grid": y and dy/dp at the nodes of a grid
  with cfg["grid"] points per component of p (cached in cfg["data_dir"]), interpolated
  multilinearly (multires_interpolator.py); "rollfb" (Fourier-Bessel stencil
  only): cfg["grid"] for the components other than the roll, the roll applied exactly
  (roll_fb.RollFB).
- Stencil: cfg["kernel"] ("tent", "flat"), cfg["probes"] per side, cfg["spectral"]
  (number of Fourier components or None), finite-difference step cfg["dp"]; with
  cfg["dims"] a reduced chart: only these components of p, their ranges scaled by
  cfg["scale"] (train_chart.ReducedStencil).
- Loss: the package's hs loss (HSLoss) with cfg["loss_params"]: each run states the
  options in which it differs from the defaults of the package (HS_0_1_0: the hs loss
  of twincher 0.1.0, with which the earlier runs were made; trimmer_domain_amp: the
  trimmer over the whole domain; momentum_reset). cfg["extra_terms"] adds loss terms
  (name -> parameters) of the package (e.g. "hs_sm_noise") or "hs_sep" (hs_sep_loss.py).
- Initial twincher: cfg["init"] (a .twc with the same n_s, n_l, loaded with its wiring;
  e.g. of a reduced chart with fewer components, or extended by twc_extend.py; the
  curriculum and the optimizer start anew). A fresh Adam moves every parameter by ~lr
  in its first steps (first and second moments equal), which wrecks a trained
  twincher even at lr 1e-5: cfg["adam_prime"] iterations update only the moments of
  Adam (parameters restored after each step, R held), then cfg["warmup"] iterations
  raise the learning rate linearly from lr / 100.
  cfg["sgt"]: threshold of the s-gate regularization of the core (default 1.3).
- Curriculum over some components only: cfg["fixed"] = {"components": [...],
  "range": R_F} draws these components uniformly in [-R_F, R_F] (GridSampler, grid data
  only), the curriculum box applies to the others (hyperrectangle).
- Exact checks every cfg["exact_stride"] iterations: det J with y and dy/dp of the
  stencil at N_CHECK fixed random points of the box |p|_inf <= cfg["check_box"] (a
  number or one per component)
  (history_exact.csv); with cfg["solver_check"] > 0 also the solver (start p = 0, p
  clamped to |p|_inf <= cfg["solver_clamp"], cfg["solver_iters"] iterations) on that
  many views of the box: fractions of failures (max |dp| > 0.1) and of views not
  solved (max |dp| > 0.01), errors in units of the full p (scaled by cfg["scale"]). best.twc keeps the checkpoint with the fewest views not
  solved, then the smallest fraction of det <= 0, then of det <= clearance. Winding
  numbers (winding.py) of small circles tiling the plane of the first two components of
  the trained box (at 3 slices of the others): fraction != 1 (wind_bad), and of one
  circle of radius 0.8 R around the centre (wind_c).
- Explosions (a stride with more than cfg["explosion_frac"] (default 1e-3) of its
  cases det < 0, after the first quiet stride): the last 1000 iterations (R, batch min
  det, negatives, |g|, largest |dr/dp_k| inside R and over the whole domain, lr, loss)
  are written to trace_<iter>.csv; with cfg["rollback"] = {"lr_factor", "max"} the run
  goes back to the checkpoint of the previous stride, the learning rate is multiplied by
  lr_factor and the batch generator reseeded (rollbacks.csv).
- Adam: cfg["adam_betas"] (also applied to the state loaded from a checkpoint). The
  momentum reset (loss_params "momentum_reset", README Section 18; with
  cfg["first_reset"] the first version, FirstReset) is paused while Adam is primed, its
  state is kept in the checkpoints, and the number of resets so far is in the history.
  At the onset of an explosion the checkpoint of the previous stride is kept as
  pre_explosion_<iter>.pt.
- Gradient clipping: cfg["clip"] = factor: the summed gradient is scaled down to factor
  times the running mean of its norm when larger (add_clipping).
- Loss terms of the package in cfg["extra_terms"] too (e.g. "hs_sm_noise": the noise
  post-training); with hs_sm_noise the history also records the largest normal-mode
  sensitivity |dr/dy| of the cases (75 % quantile of the last batch of a stride and
  maximum over the stride: noise_p75, noise_max). cfg["snapshot"] = N: the twincher is
  also saved every N iterations as iter_<n>.twc (iter_0: the initial state).
- Stop: when R reaches cfg["r_stop"], it is held there (curriculum step 0); training
  stops when all det J of the training batches inside R are above the clearance for
  cfg["done_strides"] strides, or (cfg["done_exact"] = k) after k consecutive exact
  checks at r_stop without det <= 0 and without views not solved, or after
  cfg["max_iters"] iterations.
- Afterwards: det J and the solver (start p = 0) for best.twc in the boxes
  cfg["eval_boxes"] (eval_chart.py), iterations until R = 0.8, 0.9 and the number of
  folds (strides with det < 0 for more than 1 % of the samples) -> result.json.

Files in run_dir: config.json, history.csv, history_exact.csv, checkpoint.pt (while
training), best.twc, best.json, final.twc, train.json, result.json, learner_tm.log.
"""

import collections
import csv
import json
import math
import shutil
import time
from pathlib import Path

import torch
import twincher
from twincher.hs_loss import MomentumReset

from eval_chart import dets_exact, solve
from multires_interpolator import TorchMultilinearInterpolator, grid_points
from roll_fb import RollFB
from train_chart import SOLVE_TOL, SamplePool, chart_stencil, stencil_metadata
from train_cylinder import HS_0_1_0  # noqa: F401 (for the scripts of the runs)
from winding import LoopWindings, shuttle_for

HERE = Path(__file__).resolve().parent
N_CHECK = 8192
FIELDS = ["iter", "time_s", "R", "det_min", "neg_dets", "loss", "lr", "lr_min", "g_max",
          "trim_max", "noise_p75", "noise_max", "resets"]  # fmt: skip
FIELDS_EXACT = ["iter", "R", "det_min", "det_p01", "frac_neg", "frac_low",
                "frac_neg_0.86", "frac_low_0.86", "solver_fail", "best",
                "solver_nc", "wind_bad", "wind_c"]  # fmt: skip
N_WIND = 24  # circles per side of the tiling for the winding check
# state of the learner and of the hs loss term that a checkpoint restores (with the
# twincher parameters, the optimizer and the random generator)
LEARNER_STATE = ("iter", "lr_dynamic", "p_range", "prev_g", "prev_g_norm", "c_prev",
                 "damper_count", "damper_lr_cum", "damper_n_down")  # fmt: skip
HS_STATE = ("p_range", "valid_det_count", "iter", "state", "neg_dets_count",
            "stats_det_min")  # fmt: skip


def write_json(file_name, data):
    """Write atomically: an interruption leaves either the old or the new file."""
    tmp = file_name.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(file_name)


def stencil_of(cfg):
    """The stencil of a run (settings missing in older configurations: as then)."""
    return chart_stencil(cfg["chart"], cfg["obj"], kernel=cfg.get("kernel", "tent"),
                         margin=cfg["margin"], n_grid=cfg.get("probes", 16),
                         dp=cfg.get("dp", 0.05), spectral=cfg.get("spectral"),
                         spectral_scale=cfg.get("spectral_scale", 1.0),
                         dims=cfg.get("dims"), scale=cfg.get("scale", 1.0),
                         fb=cfg.get("fb"))  # fmt: skip


def data_tag(cfg):
    spec = ""
    if cfg.get("spectral"):
        spec = f"_spec{cfg['spectral']}"
        if cfg.get("spectral_scale", 1.0) != 1.0:
            spec += f"x{cfg['spectral_scale']:g}"
    reduced = ""
    if cfg.get("fb"):
        spec = "_fb" + "_".join(f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}{v}"
                                for k, v in sorted(cfg["fb"].items()))  # fmt: skip
    if cfg.get("dims") is not None:
        reduced = f"_d{''.join(map(str, cfg['dims']))}_s{cfg.get('scale', 1.0):g}"
    return (f"obj{cfg['obj']}_{cfg['chart']}_{cfg.get('kernel', 'tent')}"
            f"{cfg.get('probes', 16)}{spec}_dp{cfg.get('dp', 0.05):g}_m{cfg['margin']:g}"
            f"{reduced}")  # fmt: skip


def grid_data(st, n_g, file, batch=512):
    """y and dy/dp (CPU) at the nodes of the grid n_g (grid_points order), cached."""
    p = grid_points(n_g)
    if file.exists():
        data = torch.load(file)
        if data["n_g"] == list(n_g):
            print(f"grid data from {file}")
            return data["y"], data["dy_dp"]
    t0 = time.perf_counter()
    y = torch.empty((len(p), st.n_y))
    dy_dp = torch.empty((len(p), st.n_y, st.n_p))
    yb = torch.empty((batch, st.n_y), device=st.device)
    db = torch.empty((batch, st.n_y, st.n_p), device=st.device)
    for i in range(0, len(p), batch):
        q = torch.as_tensor(p[i : i + batch])
        st.compute(q, yb[: len(q)], db[: len(q)])
        y[i : i + len(q)] = yb[: len(q)].cpu()
        dy_dp[i : i + len(q)] = db[: len(q)].cpu()
    torch.save({"n_g": list(n_g), "y": y, "dy_dp": dy_dp}, file)
    print(
        f"grid data {n_g} computed in {time.perf_counter() - t0:.0f} s, saved to {file}"
    )
    return y, dy_dp


def register_extra(name):
    """Register the loss term "hs_sep" of this directory (hs_sep_loss.py), once."""
    if name == "hs_sep" and name not in twincher.registry.components("loss_terms"):
        from hs_sep_loss import HSSepLoss

        twincher.register("loss_terms", name, HSSepLoss)


class GridSampler:
    """Grid data as the stencil of a Learner, with a curriculum over some components
    only: compute(p, y, dy_dp) replaces the components `fixed` of the requested points
    by uniform values in [-fixed_range, fixed_range], interpolates y and dy/dp there
    (multilinear), and sets these components of p to 0 in the batch (the hs loss uses p
    only for its ramp and the promotion, so its box is that of the other components)."""

    def __init__(self, y, dy_dp, n_g, fixed, fixed_range, device, seed=3):
        self.data_y, self.data_dy_dp = y.to(device), dy_dp.to(device)
        self.n_p, self.n_y = len(n_g), y.shape[1]
        self.mli = TorchMultilinearInterpolator(self.n_p, n_g, device)
        self.fixed, self.fixed_range = list(fixed), fixed_range
        self.rng = torch.Generator(device=device).manual_seed(seed)

    def compute(self, p, y, dy_dp):
        q = p.clone()
        u = torch.empty((len(p), len(self.fixed)), dtype=p.dtype, device=p.device)
        u.uniform_(-self.fixed_range, self.fixed_range, generator=self.rng)
        q[:, self.fixed] = u
        self.mli.interpolate(q, y, self.data_y)
        self.mli.interpolate(q, dy_dp, self.data_dy_dp)
        p[:, self.fixed] = 0


class Check:
    """det J of the twincher at fixed random points of the box |p|_inf <= R_box."""

    def __init__(self, st, wiring, box, seed, n_solve=0, clamp=None, n_iter=20):
        box = torch.as_tensor(box, dtype=torch.float64)
        self.st, self.n_solve, self.clamp, self.box = st, n_solve, clamp, box
        self.n_iter = n_iter
        self.n_slices = 1 if st.n_p == 2 else 3
        self.wind = LoopWindings(st, wiring, N_WIND**2 * self.n_slices)
        self.wind_c = LoopWindings(st, wiring, 1, n_t=128)
        g = torch.Generator().manual_seed(seed)
        self.p = box * (2 * torch.rand((N_CHECK, st.n_p), generator=g,
                                       dtype=torch.float64) - 1)  # fmt: skip
        self.y = torch.empty((N_CHECK, st.n_y), device=st.device)
        self.dy_dp = torch.empty((N_CHECK, st.n_y, st.n_p), device=st.device)
        for i in range(0, N_CHECK, 512):
            st.compute(
                self.p[i : i + 512], self.y[i : i + 512], self.dy_dp[i : i + 512]
            )
        self.sub = (self.p.abs().amax(dim=1) <= 0.86).to(st.device)
        self.n_p, self.n_y = st.n_p, st.n_y
        self.SH = shuttle_for(wiring, N_CHECK, st.n_p)

    def windings(self, a, ranges):
        """(fraction of the tiling circles with winding != 1, winding of the circle of
        radius 0.8 R around the centre) in the plane of components 0, 1 of the box
        |p_k| <= ranges[k]; the other components at 0 and +-ranges/2."""
        ranges = torch.as_tensor(ranges, dtype=torch.float64)
        n_p = self.st.n_p
        x = (torch.arange(N_WIND, dtype=torch.float64) + 0.5) / N_WIND * 2 - 1
        X, Y = torch.meshgrid(x * ranges[0], x * ranges[1], indexing="ij")
        rho = max(0.75 * 2 * min(ranges[0], ranges[1]).item() / N_WIND, 0.01)
        centres = []
        for k in range(self.n_slices):
            c = torch.zeros((N_WIND**2, n_p), dtype=torch.float64)
            c[:, 0], c[:, 1] = X.ravel(), Y.ravel()
            c[:, 2:] = (k - (self.n_slices - 1) / 2) * ranges[2:] / 2
            centres.append(c)
        centres = torch.cat(centres)
        planes = torch.tensor([[0, 1]] * len(centres))
        w = self.wind(a, centres, planes, rho)
        w_c = self.wind_c(a, torch.zeros((1, n_p), dtype=torch.float64),
                          torch.tensor([[0, 1]]), 0.8 * min(ranges[0], ranges[1]))  # fmt: skip
        return float((w != 1).float().mean()), int(w_c[0])

    def solver_errors(self, twc):
        """max |p - p_true| (units of the full p) for n_solve views of the box."""
        err = solve(twc, self.st, self.box, self.n_iter, clamp=self.clamp,
                    n=self.n_solve)  # fmt: skip
        return err * getattr(self.st, "scale", 1.0)

    def __call__(self, a, clearance):
        SH, n_y = self.SH, self.n_y
        SH.a.copy_(a)
        SH.tw_update()
        SH.s[:n_y].copy_(self.y.T)
        SH.s[n_y:].fill_(0)
        SH.Q.fill_(0)
        SH.Q[:n_y].copy_(self.dy_dp.permute(1, 2, 0))
        SH.forward_query()
        d = torch.linalg.det(SH.Q[: self.n_p].permute(2, 1, 0))
        s = d[self.sub]
        return {
            "det_min": d.min().item(),
            "det_p01": torch.quantile(d, 0.01).item(),
            "frac_neg": (d <= 0).float().mean().item(),
            "frac_low": (d <= clearance).float().mean().item(),
            "frac_neg_0.86": (s <= 0).float().mean().item(),
            "frac_low_0.86": (s <= clearance).float().mean().item(),
        }


def make_learner(cfg, run_dir, st, pool, data=None):
    learner = twincher.Learner(arch="hs", n_s=cfg["n_s"], n_l=cfg["n_l"], lr=cfg["lr"],
                               n_c=cfg["n_c"], dir=run_dir, output_stride=cfg["stride"],
                               plot_progress=False, rng_seed=cfg["seed"])  # fmt: skip
    terms = {"hs_noise": None, "hs": cfg["loss_params"]}
    for name in cfg.get("extra_terms", {}):
        register_extra(name)
    terms.update(cfg.get("extra_terms", {}))
    learner.modify_params({"loss_terms": terms})
    if pool is not None:  # SamplePool or GridSampler
        learner.set_stencil(pool)
        source = pool.source
    else:  # grid data; the learner's interpolator is replaced after init()
        learner.set_data(
            *data, stencil_name=f"itodd_obj{cfg['obj']}_depth_{cfg['chart']}"
        )
        learner.n_grid = 2  # (only for the interpolator that init() creates)
        source = "x".join(map(str, cfg["grid"]))
    learner.init()
    if pool is None:
        learner.MLI = TorchMultilinearInterpolator(st.n_p, cfg["grid"], learner.device)
    if cfg.get("init"):  # the whole twincher (wiring and parameters), new optimizer
        learner.SH.load(str(cfg["init"]), n_c=cfg["n_c"], n_p=st.n_p, n_m=learner.n_m)
        if (learner.SH.n_s, learner.SH.n_l) != (cfg["n_s"], cfg["n_l"]):
            raise ValueError(
                f"{cfg['init']}: n_s, n_l {learner.SH.n_s}, {learner.SH.n_l}"
            )
        learner.optimizer = torch.optim.Adam([learner.SH.param], lr=learner.lr,
                                             **learner.adam_params)  # fmt: skip
        if not getattr(learner.time_monitor, "disable", False):
            learner.SH.enable_time_monitor(time_monitor=learner.time_monitor)
    if cfg.get("sgt") is not None:
        learner.SH.core.sgt = cfg["sgt"]
    learner.clip_ema, learner.g_raw, learner.n_clipped = None, 0.0, 0
    if cfg.get("clip"):
        add_clipping(learner, cfg["clip"])
    learner.SH.metadata["stencil"] = stencil_metadata(
        st, cfg["chart"], cfg["obj"], cfg["margin"], source
    )
    learner.SH.metadata["training"] = {k: cfg[k] for k in ("loss_params", "lr")}
    hs = next(term for term in learner.loss_terms if term.name == "hs")
    if cfg.get("first_reset"):
        hs.reset = FirstReset(**cfg["loss_params"]["momentum_reset"])
    return learner, hs


class FirstReset(MomentumReset):
    """The first version of the momentum reset (the study of README Section 18, and the
    run z0_reset of noise_charts.py), to reproduce these runs: a batch is rejected when
    more than neg_frac of its cases inside R have det < 0 (no running mean); the
    parameters go back to those before the previous update, also when the rejected
    batch was evaluated right after a reset (then to the state rejected before); the
    learning-rate factor has no floor; a resumed run starts the reset afresh. A state
    with a few det < 0 just above neg_frac in every batch then makes it reset in every
    iteration, alternating between two states, with the learning rate falling to zero:
    the reason for the revised version of the package (twincher.hs_loss.MomentumReset)."""

    def __init__(self, neg_frac, lr_factor=0.5, lr_up=1.1):
        super().__init__(neg_frac=neg_frac, ratio=0.0, lr_factor=lr_factor,
                         lr_up=lr_up, lr_floor=0.0)  # fmt: skip
        self.previous = None

    def load_state(self, state):
        super().load_state(None)
        self.previous = None

    def judge(self, dets):
        if self.paused:
            return
        self.boost = min(1.0, self.boost * self.lr_up)
        previous, self.previous = self.previous, self.current
        if dets.numel() and (dets < 0).float().mean().item() > self.neg_frac:
            self.target = self.current if previous is None else previous
            self.boost *= self.lr_factor
            self.n += 1


def add_clipping(learner, factor, beta=0.99):
    """Clip the summed gradient of the loss terms to factor times the running mean of
    its norm, before the damper and the optimizer step (Learner.step calls _damper
    between the two): a rare case with a huge loss otherwise makes Adam move every
    parameter coherently by a sizeable fraction of lr."""
    damper = learner._damper

    def clipped(make_output):
        norm = torch.linalg.vector_norm(learner.g).item()
        learner.g_raw = norm
        ema = learner.clip_ema
        if ema is not None and norm > factor * ema:
            learner.g.mul_(factor * ema / norm)
            norm = factor * ema
            learner.n_clipped += 1
        learner.clip_ema = norm if ema is None else beta * ema + (1 - beta) * norm
        damper(make_output)

    learner._damper = clipped


def trained_ranges(cfg, hs, n_p):
    """Range of each component of p trained so far (curriculum or fixed)."""
    ranges = torch.full((n_p,), hs.p_range, dtype=torch.float64)
    fixed = cfg.get("fixed")
    if fixed:
        ranges[fixed["components"]] = fixed["range"]
    return ranges


def save_checkpoint(file_name, learner, hs, pool, extra):
    state = {
        "a": learner.SH.a.clone(),
        "metadata": learner.SH.metadata,
        "optimizer": learner.optimizer.state_dict(),
        "rng": learner.rng.get_state(),
        "pool_rng": None
        if getattr(pool, "rng", None) is None
        else pool.rng.get_state(),
        "learner": {
            k: getattr(learner, k) for k in LEARNER_STATE if hasattr(learner, k)
        },
        "hs": {k: getattr(hs, k) for k in HS_STATE},
        **extra,
    }
    tmp = file_name.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(file_name)


def load_checkpoint(file_name, learner, hs, pool):
    state = torch.load(file_name, map_location="cuda")
    learner.SH.a.copy_(state["a"])
    learner.SH.tw_update()
    learner.SH.metadata.update(state["metadata"])
    learner.optimizer.load_state_dict(state["optimizer"])
    learner.rng.set_state(state["rng"].cpu())
    if pool is not None and state["pool_rng"] is not None:
        pool.rng.set_state(state["pool_rng"].cpu())
    for k, v in state["learner"].items():
        setattr(learner, k, v)
    for k, v in state["hs"].items():
        setattr(hs, k, v)
    return state


def read_csv(file, keep_until=None):
    if not file.exists():
        return []
    with open(file, newline="") as f:
        rows = list(csv.DictReader(f))
    if keep_until is not None:
        rows = [r for r in rows if int(r["iter"]) <= keep_until]
    return rows


def rewrite_csv(file, fields, rows):
    with open(file, "w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def train(cfg, run_dir, log=print):
    """Train (or continue) one run; returns the result dictionary."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = run_dir / "config.json"
    if cfg_file.exists() and json.loads(cfg_file.read_text()) != cfg:
        raise RuntimeError(f"{cfg_file} differs from the configuration; delete or "
                           "rename the directory")  # fmt: skip
    write_json(cfg_file, cfg)
    if (run_dir / "result.json").exists():
        return json.loads((run_dir / "result.json").read_text())
    name = run_dir.name
    st = stencil_of(cfg)
    if not (run_dir / "train.json").exists():
        pool, data = None, None
        if cfg.get("data", "pool") == "pool":
            pool_file = Path(cfg["pool_dir"]) / (f"pool_obj{cfg['obj']}_{cfg['chart']}_"
                                                f"{cfg['pool_size']}_m{cfg['margin']:g}_tent.pt")  # fmt: skip
            if cfg.get("dims") is not None:
                pool_file = pool_file.with_name(
                    f"pool_{data_tag(cfg)}_{cfg['pool_size']}.pt"
                )
            pool_file.parent.mkdir(parents=True, exist_ok=True)
            pool = SamplePool(st, cfg["pool_size"], pool_file)
            pool.name = f"itodd_obj{cfg['obj']}_depth_{cfg['chart']}"
            pool.source = f"pool {cfg['pool_size']}"
        elif cfg["data"] == "rollfb":  # grid without the roll, roll exact (FB stencil)
            data_dir = Path(cfg["data_dir"])
            data_dir.mkdir(parents=True, exist_ok=True)
            n_g = "x".join(map(str, cfg["grid"]))
            pool = RollFB(
                st, cfg["grid"], data_dir / f"rollfb_{data_tag(cfg)}_{n_g}.pt"
            )
            pool.name = f"itodd_obj{cfg['obj']}_depth_{cfg['chart']}"
            pool.source = f"rollfb {n_g}"
        else:
            data_dir = Path(cfg["data_dir"])
            data_dir.mkdir(parents=True, exist_ok=True)
            n_g = "x".join(map(str, cfg["grid"]))
            data = grid_data(
                st, cfg["grid"], data_dir / f"grid_{data_tag(cfg)}_{n_g}.pt"
            )
            if cfg.get("fixed"):
                pool = GridSampler(*data, cfg["grid"], cfg["fixed"]["components"],
                                   cfg["fixed"]["range"], "cuda")  # fmt: skip
                pool.name = f"itodd_obj{cfg['obj']}_depth_{cfg['chart']}"
                pool.source = n_g + f", fixed {cfg['fixed']}"
                data = None
        holder = [data]  # _train releases the CPU copy once the learner has it
        del data
        _train(cfg, run_dir, st, pool, name, log, holder)
        del pool
        torch.cuda.empty_cache()
    res = {"config": cfg, **json.loads((run_dir / "train.json").read_text())}
    res["best"] = json.loads((run_dir / "best.json").read_text())
    rows = read_csv(run_dir / "history.csv")
    for key, R in (("it_R0.8", 0.8), ("it_R0.9", 0.9), ("it_R_stop", cfg["r_stop"])):
        it_R = [int(r["iter"]) for r in rows if float(r["R"]) >= R - 1e-9]
        res[key] = it_R[0] if it_R else None
    res["min_R_stop"] = next(
        (
            float(r["time_s"]) / 60
            for r in rows
            if float(r["R"]) >= cfg["r_stop"] - 1e-9
        ),
        None,
    )
    per_stride = cfg["n_c"] * cfg["stride"]
    res["folds"] = sum(int(r["neg_dets"]) > 0.01 * per_stride for r in rows)
    p, dets = dets_exact(run_dir / "best.twc", st)
    norm = p.abs().amax(dim=1)
    res["eval"] = {}
    for R in cfg["eval_boxes"]:
        d = dets[norm <= R]
        err = solve(run_dir / "best.twc", st, R, cfg.get("solver_iters", 20),
                    clamp=cfg.get("solver_clamp")) * cfg.get("scale", 1.0)  # fmt: skip
        res["eval"][str(R)] = {
            "frac_det_neg": (d <= 0).float().mean().item(),
            "frac_det_low": (d <= 0.5).float().mean().item(),
            "det_min": d.min().item(),
            "solver_success": float((err < SOLVE_TOL).mean()),
            "solver_err_below_0.01": float((err < 0.01).mean()),
            "solver_err_above_0.1": float((err > 0.1).mean()),
        }
        e = res["eval"][str(R)]
        log(f"[{name}] best.twc, |p| <= {R}: det <= 0 {100 * e['frac_det_neg']:.3f} %, "
            f"solver: error < 1e-3 {100 * e['solver_success']:.1f} %, < 0.01 "
            f"{100 * e['solver_err_below_0.01']:.1f} %, > 0.1 "
            f"{100 * e['solver_err_above_0.1']:.1f} %")  # fmt: skip
    write_json(run_dir / "result.json", res)
    return res


def _train(cfg, run_dir, st, pool, name, log, holder):
    learner, hs = make_learner(cfg, run_dir, st, pool, holder.pop())
    domain = hs.trimmer_domain_amp > 0  # the trimmer over the whole domain
    reset = hs.reset
    noise = next((t for t in learner.loss_terms if t.name == "hs_sm_noise"), None)
    wiring = run_dir / "wiring.twc"  # layers and pairs of the twincher, for the checks
    learner.SH.save(str(wiring))
    check = Check(st, wiring, cfg["check_box"], seed=2026,
                  n_solve=cfg.get("solver_check", 0), clamp=cfg.get("solver_clamp"),
                  n_iter=cfg.get("solver_iters", 20))  # fmt: skip
    clearance = hs.det_clearance
    ckpt, hist, hist_e = (run_dir / "checkpoint.pt", run_dir / "history.csv",
                          run_dir / "history_exact.csv")  # fmt: skip
    if ckpt.exists():
        state = load_checkpoint(ckpt, learner, hs, pool)
        start, t_train, done = state["iter"] + 1, state["train_time"], state["done"]
        best = state["best"]
        rows = read_csv(hist, state["iter"])
        rows_e = read_csv(hist_e, state["iter"])
        log(f"[{name}] resumed after iteration {state['iter']}")
    else:
        start, t_train, done, best, rows, rows_e = 0, 0.0, 0, None, [], []
    if hs.p_range >= cfg["r_stop"] - 1e-9:  # R held (as in the run before)
        hs.p_range_step = 0.0
    if cfg.get("adam_betas"):  # also over the state of a checkpoint
        for group in learner.optimizer.param_groups:
            group["betas"] = tuple(cfg["adam_betas"])
    if reset and ckpt.exists():
        reset.load_state(state.get("reset"))
    rewrite_csv(hist, FIELDS, rows)
    rewrite_csv(hist_e, FIELDS_EXACT, rows_e)

    status, it, det_min, n_neg = "max_iters", start, math.inf, 0
    lr_min, g_max, trim_max, noise_max = math.inf, 0.0, 0.0, 0.0
    prime, warmup = cfg.get("adam_prime", 0), cfg.get("warmup", 0)
    step_R = hs.p_range_step
    rollback = cfg.get("rollback")
    lr_scale, n_rollbacks, clean = 1.0, 0, 0
    if ckpt.exists():
        lr_scale, n_rollbacks = state.get("lr_scale", 1.0), state.get("rollbacks", 0)
        learner.clip_ema = state.get("clip_ema")
        learner.n_clipped = state.get("n_clipped", 0)
        clean = state.get("clean_checks", 0)
    explosion = cfg.get("explosion_frac", 1e-3) * cfg["stride"] * cfg["n_c"]
    trace = collections.deque(maxlen=1000)  # per iteration, written at explosions
    armed, exploding = ckpt.exists(), False  # armed after the first quiet stride
    with open(hist, "a", newline="") as f, open(hist_e, "a", newline="") as fe:
        writer, writer_e = csv.DictWriter(f, FIELDS), csv.DictWriter(fe, FIELDS_EXACT)
        while it <= cfg["max_iters"]:
            assert learner.iter == it
            # cap of the damped learning rate: warm-up and reductions after rollbacks
            warm = min(1.0, max(0.01, (it - prime) / warmup)) if warmup else 1.0
            learner.lr = cfg["lr"] * lr_scale * warm  # (x the factor of the reset)
            learner.lr_dynamic = min(learner.lr_dynamic, learner.lr)
            if it < prime:
                hs.p_range_step = 0.0
                a0 = learner.SH.a.clone()
            elif it == prime and hs.p_range < cfg["r_stop"] - 1e-9:
                hs.p_range_step = step_R
            t0 = time.perf_counter()
            if reset:
                reset.paused = it < prime
            learner.step()
            if it < prime:  # only the moments of Adam move
                learner.SH.a.copy_(a0)
                learner.SH.tw_update()
            t_train += time.perf_counter() - t0
            g = torch.linalg.vector_norm(learner.g).item()
            lr_min, g_max = min(lr_min, learner.lr_dynamic), max(g_max, g)
            if domain:
                trim_max = max(trim_max, float(hs.domain_norm_max))
            if noise is not None:
                noise_max = max(noise_max, noise.vel_max.max().item())
            if hs.p_range >= cfg["r_stop"] - 1e-9:
                hs.p_range_step = 0.0  # hold R
            dets = hs.dets_inner  # without the cases beyond R
            if dets.numel():
                det_min = min(det_min, dets.min().item())
                n_neg += (dets < 0).sum().item()
            trace.append({"iter": it, "R": round(hs.p_range, 4),
                          "det_min": dets.min().item() if dets.numel() else math.nan,
                          "neg": int((dets < 0).sum()), "g": g,
                          "g_raw": learner.g_raw, "clipped": learner.n_clipped,
                          "resets": reset.n if reset else 0,
                          "hs_max": float(getattr(hs, "norm_max", math.nan)),
                          "trim_max": float(hs.domain_norm_max) if domain else math.nan,
                          "lr": learner.lr_dynamic, "loss": float(learner.loss)})  # fmt: skip
            if it % cfg["stride"] != 0:
                it += 1
                continue
            row = {"iter": it, "time_s": round(t_train, 1), "R": round(hs.p_range, 3),
                   "det_min": round(det_min, 4), "neg_dets": n_neg,
                   "loss": float(learner.loss), "lr": learner.lr_dynamic,
                   "lr_min": lr_min, "g_max": round(g_max, 5),
                   "trim_max": round(trim_max, 3),
                   "resets": reset.n if reset else 0}  # fmt: skip
            if noise is not None:
                row["noise_p75"] = round(torch.quantile(noise.vel_max, 0.75).item(), 4)
                row["noise_max"] = round(noise_max, 4)
            writer.writerow(row)
            f.flush()
            log(f"[{name}] {it:6d} {t_train:7.0f} s  R {hs.p_range:.3f}  min det "
                f"{det_min:+.3f}  neg {n_neg:7d}  loss {row['loss']:.2e}"
                + (f"  |dr/dy|_normal p75 {row['noise_p75']:.3f} max "
                   f"{row['noise_max']:.3f}" if noise is not None else ""))  # fmt: skip
            if n_neg <= explosion:
                armed = armed or it >= start + cfg["stride"]  # a full quiet stride
                exploding = False
            elif armed:
                if not exploding:  # the onset: the iterations before it
                    rewrite_csv(
                        run_dir / f"trace_{it}.csv", list(trace[0]), list(trace)
                    )
                    if ckpt.exists():  # the state at the start of this stride
                        shutil.copy(ckpt, run_dir / f"pre_explosion_{it}.pt")
                exploding = True
                if (rollback and n_rollbacks < rollback.get("max", 8)
                        and ckpt.exists()):  # fmt: skip
                    state = load_checkpoint(ckpt, learner, hs, pool)
                    learner.clip_ema = state.get("clip_ema")
                    if reset:  # its accepted states are later than the checkpoint
                        reset.load_state(state.get("reset"))
                    n_rollbacks += 1
                    lr_scale *= rollback.get("lr_factor", 0.5)
                    learner.rng.manual_seed(cfg["seed"] + 1000 * n_rollbacks)
                    it, t_back = state["iter"] + 1, it
                    done, best = state["done"], state["best"]
                    f.flush()
                    fe.flush()
                    rewrite_csv(hist, FIELDS, read_csv(hist, state["iter"]))
                    rewrite_csv(hist_e, FIELDS_EXACT, read_csv(hist_e, state["iter"]))
                    with open(run_dir / "rollbacks.csv", "a") as fr:
                        fr.write(f"{t_back},{state['iter']},{n_neg},{lr_scale}\n")
                    log(f"[{name}] explosion at {t_back} ({n_neg} det < 0): back to "
                        f"{state['iter']}, lr x {lr_scale:g}")  # fmt: skip
                    det_min, n_neg = math.inf, 0
                    lr_min, g_max, trim_max, noise_max = math.inf, 0.0, 0.0, 0.0
                    trace.clear()
                    exploding = False
                    continue
            if it % cfg["exact_stride"] == 0 and it > 0:
                e = {
                    "iter": it,
                    "R": round(hs.p_range, 3),
                    **check(learner.SH.a, clearance),
                }
                e["wind_bad"], e["wind_c"] = check.windings(
                    learner.SH.a, trained_ranges(cfg, hs, st.n_p)
                )
                if check.n_solve:
                    learner.SH.save(str(run_dir / "check.twc"))
                    err = check.solver_errors(run_dir / "check.twc")
                    e["solver_fail"] = float((err > 0.1).mean())
                    e["solver_nc"] = float((err > 0.01).mean())
                score = (e["frac_neg"], e["frac_low"])
                if check.n_solve:
                    score = (e["solver_nc"], *score)
                e["best"] = best is None or score < tuple(best["score"])
                if e["best"]:
                    best = {"iter": it, "R": e["R"], "score": list(score), **e}
                    learner.SH.save(str(run_dir / "best.twc"))
                    write_json(run_dir / "best.json", best)
                writer_e.writerow(e)
                fe.flush()
                log(f"[{name}] exact, |p| <= {cfg['check_box']}: det <= 0 "
                    f"{100 * e['frac_neg']:.3f} %, det <= {clearance} "
                    f"{100 * e['frac_low']:.3f} %, min {e['det_min']:+.2f}"
                    + (f", solver: not solved {100 * e['solver_nc']:.1f} %, failures "
                       f"{100 * e['solver_fail']:.1f} %" if check.n_solve else "")
                    + f", winding != 1 {100 * e['wind_bad']:.1f} %, centre {e['wind_c']}"
                    + ("  (best)" if e["best"] else ""))  # fmt: skip
                at_stop = hs.p_range >= cfg["r_stop"] - 1e-9
                clean = clean + 1 if at_stop and e["frac_neg"] == 0 and (
                    not check.n_solve or e["solver_nc"] == 0) else 0  # fmt: skip
            ok = hs.p_range >= cfg["r_stop"] - 1e-9 and det_min > clearance
            done = done + 1 if ok else 0
            det_min, n_neg = math.inf, 0
            lr_min, g_max, trim_max, noise_max = math.inf, 0.0, 0.0, 0.0
            if cfg.get("snapshot") and it % cfg["snapshot"] == 0:
                learner.SH.save(str(run_dir / f"iter_{it}.twc"))
            if not math.isfinite(row["loss"]):
                status = "diverged"
                break
            if done >= cfg["done_strides"]:
                status = "done"
                break
            if cfg.get("done_exact") and clean >= cfg["done_exact"]:
                status = "done_exact"
                break
            save_checkpoint(ckpt, learner, hs, pool, {"iter": it, "train_time": t_train,
                                                      "done": done, "best": best,
                                                      "lr_scale": lr_scale,
                                                      "rollbacks": n_rollbacks,
                                                      "clean_checks": clean,
                                                      "clip_ema": learner.clip_ema,
                                                      "n_clipped": learner.n_clipped,
                                                      "reset": reset.state()
                                                      if reset else None})  # fmt: skip
            it += 1
    it = min(it, cfg["max_iters"])
    learner.time_monitor.save_report()
    learner.SH.save(str(run_dir / "final.twc"))
    if best is None:  # no exact check yet: the final state
        e = {"iter": it, "R": round(hs.p_range, 3), **check(learner.SH.a, clearance)}
        best = {"iter": it, "score": [e["frac_neg"], e["frac_low"]], **e}
        learner.SH.save(str(run_dir / "best.twc"))
        write_json(run_dir / "best.json", best)
    write_json(run_dir / "train.json", {"status": status, "iters": it,
                                        "train_time_s": t_train, "R": hs.p_range,
                                        "rollbacks": n_rollbacks,
                                        "clipped": learner.n_clipped,
                                        "resets": reset.n if reset else 0})  # fmt: skip
    ckpt.unlink(missing_ok=True)
    log(f"[{name}] {status} after {it} iterations, {t_train:.0f} s, R {hs.p_range:.3f}; "
        f"best exact check at {best['iter']}: det <= 0 {100 * best['frac_neg']:.3f} %")  # fmt: skip


def summary_table(results):
    """Lines of a table of results (name -> result or None)."""
    head = (f"{'run':24s} {'status':>9s} {'iters':>6s} {'min':>5s} {'R':>5s} "
            f"{'R0.8@':>6s} {'R0.9@':>6s} {'Rstop@':>6s} {'min':>5s} {'folds':>5s} "
            f"{'best@':>6s} {'neg@box':>8s} | "
            f"{'det<=0':>7s} {'ok<1e-3':>8s} {'<0.01':>6s} {'>0.1':>5s}  (largest eval box)")  # fmt: skip
    lines = [head]
    for name, r in results.items():
        if r is None:
            lines.append(f"{name:24s} {'pending':>9s}")
            continue
        e = r["eval"][str(r["config"]["eval_boxes"][-1])]
        lines.append(
            f"{name:24s} {r['status']:>9s} {r['iters']:6d} {r['train_time_s'] / 60:5.0f} "
            f"{r['R']:5.2f} {r['it_R0.8']!s:>6s} {r['it_R0.9']!s:>6s} "
            f"{r.get('it_R_stop')!s:>6s} {fmt_min(r.get('min_R_stop')):>5s} {r['folds']:5d} {r['best']['iter']:6d} {100 * r['best']['frac_neg']:7.3f}% | "
            f"{100 * e.get('frac_det_neg', math.nan):6.3f}% "
            f"{100 * e.get('solver_success', math.nan):7.1f}% "
            f"{100 * e.get('solver_err_below_0.01', math.nan):5.1f}% "
            f"{100 * e.get('solver_err_above_0.1', math.nan):4.1f}%"
        )
    return lines


def fmt_min(x):
    return "-" if x is None else f"{x:.0f}"


def write_summary(results, out_dir):
    lines = summary_table(results)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


def run_series(runs, default_out, description):
    """Command line of a series of runs (name -> configuration): completed runs are
    skipped, an interrupted run continues from its checkpoint; --summary only prints the
    table; --set key=value (JSON value) overrides a setting of all runs (tests)."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--runs", nargs="+", help="only these runs (default: all)")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    parser.add_argument("--summary", action="store_true", help="no training")
    args = parser.parse_args()
    if args.runs:
        unknown = set(args.runs) - set(runs)
        if unknown:
            raise SystemExit(f"unknown runs {sorted(unknown)}; known: {list(runs)}")
        runs = {name: runs[name] for name in args.runs}
    for item in args.set:
        key, value = item.split("=", 1)
        for cfg in runs.values():
            if key not in cfg:
                raise SystemExit(f"unknown setting {key}")
            cfg[key] = json.loads(value)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def result(name):
        file = args.out_dir / name / "result.json"
        return json.loads(file.read_text()) if file.exists() else None

    todo = [] if args.summary else [n for n in runs if result(n) is None]
    for name in todo:
        t0 = time.perf_counter()
        train(runs[name], args.out_dir / name)
        print(f"[{name}] completed in {(time.perf_counter() - t0) / 60:.1f} min")
        torch.cuda.empty_cache()
        write_summary({n: result(n) for n in runs}, args.out_dir)
    if not todo:
        write_summary({n: result(n) for n in runs}, args.out_dir)
