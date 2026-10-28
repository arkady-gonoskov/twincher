"""Stage 4: solving with unknown pixels; solvers and twinchers compared on the same views.

Synthetic test: for n_cases random p_true in [-1, 1]^3, the depth map at p_true with 2-3
elongated elliptical regions of unknown pixels (NaN): axis ratio 1:aspect, centred at
random pixels of the object, random orientation, semi-major axis log-uniform in
[a_min, a_max] pixels (of the res x res map). Each run (MODE:TWINCHER) solves all views
from p = 0 with n_iter steps:
- fused: FusedHSSolver, unknown pixels filled from the rendering at the current p;
- newton: the same with the Newton step for the fused residual (fused_solver.py);
- naive: HSSolver with the fixed target y(observed map), unknown pixels as background.
TWINCHER: stage2 (output_train_cylinder/final.twc) or iter<N> (output_train_noise/
iter_<N>.twc, stage 3). Success: max_k |p_k - p_true,k| < tol. Failures are classified by
the misfit on the available pixels (RMS of the y-tone difference between the rendering at
the solution and the observation): "consistent" if it is not larger than for the successes
of the same run (95th percentile), i.e. the false solution explains the available data.

Outputs in OUT_DIR: unknown.npz, unknown_solvers.png (the runs on the stage-3 twincher
iter10000), unknown_twinchers.png (fused and newton runs for all twinchers).

Usage:
    python test_unknown.py [--runs MODE:TWINCHER,...] [--n-iter 10] [--n-cases 2048]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import twincher

from fused_solver import FusedHSSolver, solve_fused
from train_noise import stencil_from_metadata

HERE = Path(__file__).resolve().parent
TWINCHERS = ["stage2", "iter2000", "iter4000", "iter6000", "iter8000", "iter10000"]
DEFAULT_RUNS = [f"{m}:{t}" for t in TWINCHERS for m in ("fused", "newton")]
DEFAULT_RUNS.append("naive:iter10000")


def twc_file(label):
    if label == "stage2":
        return HERE / "output_train_cylinder/final.twc"
    return HERE / f"output_train_noise/iter_{label[4:]}.twc"


def unknown_regions(z, gen, aspect, a_min, a_max):
    """Mask (n, res, res) of 2-3 random ellipses centred on object pixels of z."""
    n, res, _ = z.shape
    dev = z.device
    obj = (z > 0).reshape(n, -1).float()
    centres = torch.multinomial(obj, 3, replacement=True, generator=gen)  # (n, 3)
    cy, cx = (centres // res).float() + 0.5, (centres % res).float() + 0.5
    k = 2 + torch.randint(0, 2, (n,), generator=gen, device=dev)  # 2 or 3 ellipses
    log_a = math.log(a_min) + (math.log(a_max) - math.log(a_min)) * torch.rand(
        (n, 3), generator=gen, device=dev
    )
    a = torch.exp(log_a)
    a[:, 2] = torch.where(k == 3, a[:, 2], 0)
    b = a / aspect
    t = math.pi * torch.rand((n, 3), generator=gen, device=dev)
    pix = torch.arange(res, device=dev, dtype=torch.float32) + 0.5
    mask = torch.zeros_like(z, dtype=torch.bool)
    for e in range(3):
        dx = pix[None, None, :] - cx[:, e, None, None]
        dy = pix[None, :, None] - cy[:, e, None, None]
        c, s = t[:, e].cos()[:, None, None], t[:, e].sin()[:, None, None]
        u = (dx * c + dy * s) / a[:, e, None, None].clamp(min=1e-6)
        v = (-dx * s + dy * c) / b[:, e, None, None].clamp(min=1e-6)
        mask |= (u * u + v * v <= 1) & (a[:, e, None, None] > 0)
    return mask


def tone(st, z):
    """y tones of depth maps (background and unknown: s_amp)."""
    y = st.s_amp * (2 * (z - st.z_lo) / (st.z_bg - st.z_lo) - 1)
    return torch.where(z > 0, y, st.s_amp)


def solve_run(st, mode, twc, z_avail, n_iter):
    """p after each iteration (n_iter, n_c, n_p) and the final p (tensor)."""
    n_c = len(z_avail)
    traj = []
    if mode == "naive":
        solver = twincher.solver(str(twc), n_c=n_c)
        y_avail, _ = st.y_from_depth(z_avail)
        solver.start_solution(y_avail)
        for _ in range(n_iter):
            st.compute(solver.p, solver.y, solver.dy_dp)
            solver.step(y=solver.y, dy_dp=solver.dy_dp)
            traj.append(solver.p.double().cpu().numpy())
    else:
        solver = FusedHSSolver(str(twc), n_c=n_c)

        def record(it, s):
            traj.append(s.p.double().cpu().numpy())

        newton = mode == "newton"
        solve_fused(solver, st, z_avail, n_iter, newton=newton, callback=record)
    return np.stack(traj), solver.p.double()


def run_batch(st, runs, p_true, gen, args):
    z_true = st.depth_image(p_true)
    unknown = unknown_regions(z_true, gen, args.aspect, args.a_min, args.a_max)
    obj = z_true > 0
    out = {
        "f_obj": ((unknown & obj).sum((1, 2)) / obj.sum((1, 2))).cpu().numpy(),
        "f_all": unknown.float().mean((1, 2)).cpu().numpy(),
    }
    z_avail = torch.where(unknown, torch.nan, z_true)
    avail = ~unknown
    tone_true = tone(st, z_true)
    del z_true
    for run in runs:
        mode, label = run.split(":")
        traj, p = solve_run(st, mode, twc_file(label), z_avail, args.n_iter)
        out[f"p:{run}"] = traj
        d = (tone(st, st.depth_image(p)) - tone_true).square() * avail
        out[f"misfit:{run}"] = (d.sum((1, 2)) / avail.sum((1, 2))).sqrt().cpu().numpy()
    return out


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(color="#e4e3df", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#52514e")


class Analysis:
    def __init__(self, res, tol, n_bin_min=10):
        self.res, self.tol = res, tol
        f = res["f_obj"]
        self.edges = np.arange(0, min(1.0, f.max()) + 0.1, 0.1)
        self.bins = np.clip(np.digitize(f, self.edges) - 1, 0, len(self.edges) - 2)
        self.centre = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.count = np.bincount(self.bins, minlength=len(self.centre))
        self.keep = self.count >= n_bin_min

    def ok(self, run, tol=None, it=-1):
        err = np.abs(self.res[f"p:{run}"][it] - self.res["p_true"]).max(-1)
        return err < (tol or self.tol)

    def rate(self, flags):
        return np.array(
            [flags[self.bins == b].mean() if self.keep[b] else np.nan
             for b in range(len(self.centre))]
        )  # fmt: skip

    def failures(self, run):
        """(consistent, stuck) flags of the failures of a run."""
        ok = self.ok(run)
        mis = self.res[f"misfit:{run}"]
        limit = np.quantile(mis[ok], 0.95)
        return ~ok & (mis <= limit), ~ok & (mis > limit)


def plot_solvers(file_name, A, label, tol_precise):
    import matplotlib.pyplot as plt

    runs = [  # run, legend label, colour (categorical, validated), marker
        (f"newton:{label}", "fused, Newton step", "#2a78d6", "o"),
        (f"fused:{label}", "fused", "#eb6834", "s"),
        (f"naive:{label}", "naive", "#1baf7a", "^"),
    ]
    runs = [r for r in runs if f"p:{r[0]}" in A.res]
    n_iter = A.res[f"p:{runs[0][0]}"].shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained",
                             facecolor="#fcfcfb")  # fmt: skip
    for run, name, col, mk in runs:
        axes[0].plot(A.centre, A.rate(A.ok(run)), color=col, marker=mk, ms=5,
                     label=f"{name}, |Δp| < {A.tol:g}")  # fmt: skip
        axes[0].plot(A.centre, A.rate(A.ok(run, tol_precise)), color=col, ls=":",
                     marker=mk, ms=4, label=f"{name}, |Δp| < {tol_precise:g}")  # fmt: skip
        its = np.arange(1, n_iter + 1)
        axes[1].plot(its, [A.ok(run, it=i).mean() for i in range(n_iter)], color=col,
                     marker=mk, ms=5, label=name)  # fmt: skip
        if not run.startswith("naive"):
            consistent, stuck = A.failures(run)
            axes[2].plot(A.centre, A.rate(consistent), color=col, marker=mk, ms=5,
                         label=f"{name}: consistent with the data")  # fmt: skip
            axes[2].plot(A.centre, A.rate(stuck), color=col, ls="--", marker=mk, ms=5,
                         label=f"{name}: inconsistent (stuck)")  # fmt: skip
    axes[0].set_title(f"after {n_iter} iterations ({label})", loc="left")
    axes[0].set_ylabel(f"success probability (max |Δp_k| < {A.tol:g})")
    axes[0].set_xlabel("fraction of unknown object pixels")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    axes[1].set_title("all cases, by iteration", loc="left")
    axes[1].set_ylabel(f"success probability (max |Δp_k| < {A.tol:g})")
    axes[1].set_xlabel("iteration")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    axes[2].set_title(
        f"failures (|Δp| >= {A.tol:g}) after {n_iter} iterations", loc="left"
    )
    axes[2].set_ylabel("fraction of cases")
    axes[2].set_xlabel("fraction of unknown object pixels")
    axes[2].legend(frameon=False, loc="upper left", fontsize=8)
    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        style(ax)
    for ax in (axes[0], axes[2]):
        ax.set_xlim(0, A.edges[-1])
    fig.savefig(file_name, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def plot_twinchers(file_name, A):
    import matplotlib.pyplot as plt

    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal, validated
    modes = [
        m for m in ("fused", "newton") if any(k.startswith(f"p:{m}:") for k in A.res)
    ]
    fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 4.8),
                             layout="constrained", squeeze=False, facecolor="#fcfcfb")  # fmt: skip
    for ax, mode in zip(axes[0], modes):
        labels = [t for t in TWINCHERS if f"p:{mode}:{t}" in A.res]
        colours = iter(ramp[-sum(t != "stage2" for t in labels) :])
        n_iter = 0
        for t in labels:
            run = f"{mode}:{t}"
            n_iter = A.res[f"p:{run}"].shape[0]
            if t == "stage2":
                kw = {
                    "color": "#8a8984",
                    "ls": "--",
                    "label": "stage 2 (no noise training)",
                }
            else:
                kw = {"color": next(colours), "label": f"stage 3, iteration {t[4:]}"}
            ax.plot(A.centre, A.rate(A.ok(run)), marker="o", ms=5, **kw)
        name = "fused, Newton step" if mode == "newton" else "fused"
        ax.set_title(f"{name}, {n_iter} iterations", loc="left")
        ax.set_ylabel(f"success probability (max |Δp_k| < {A.tol:g})")
        ax.set_xlabel("fraction of unknown object pixels")
        ax.set_xlim(0, A.edges[-1])
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, loc="lower left", fontsize=8)
        style(ax)
    fig.savefig(file_name, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def print_table(A, runs, tol_precise):
    print(f"\nsuccess after the last iteration: max |dp_k| < {A.tol:g} (in brackets "
          f"< {tol_precise:g}); unknown = fraction of unknown object pixels")  # fmt: skip
    print(" unknown | cases | " + " | ".join(f"{r:>20s}" for r in runs))
    oks = {r: (A.ok(r), A.ok(r, tol_precise)) for r in runs}
    for b in np.nonzero(A.keep)[0]:
        m = A.bins == b
        cells = [
            f"{oks[r][0][m].mean():5.3f} ({oks[r][1][m].mean():5.3f})" for r in runs
        ]
        row = " | ".join(f"{c:>20s}" for c in cells)
        print(f"  {A.centre[b]:5.2f}  | {A.count[b]:5d} | {row}")
    cells = [f"{oks[r][0].mean():5.3f} ({oks[r][1].mean():5.3f})" for r in runs]
    print(f"   all   | {len(A.bins):5d} | " + " | ".join(f"{c:>20s}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=HERE / "output_unknown_newton")
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--n-cases", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--aspect", type=float, default=5.0)
    parser.add_argument("--a-min", type=float, default=10.0, help="pixels (512^2 map)")
    parser.add_argument("--a-max", type=float, default=400.0)
    parser.add_argument("--tol", type=float, default=0.05, help="success: max |dp_k|")
    parser.add_argument("--tol-precise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = args.runs.split(",")

    meta = twincher.load_metadata(str(twc_file("iter10000")))
    st = stencil_from_metadata(meta["stencil"])
    gen = torch.Generator(device=st.device).manual_seed(args.seed)
    p_all = 2 * torch.rand((args.n_cases, st.n_p), generator=gen, device=st.device,
                           dtype=torch.float64) - 1  # fmt: skip
    parts = []
    for i in range(0, args.n_cases, args.batch):
        parts.append(run_batch(st, runs, p_all[i : i + args.batch], gen, args))
        print(f"{i + len(parts[-1]['f_obj'])} of {args.n_cases} cases done", flush=True)
    res = {k: np.concatenate([p[k] for p in parts], axis=1 if k.startswith("p:") else 0)
           for k in parts[0]}  # fmt: skip
    res["p_true"] = p_all.cpu().numpy()
    np.savez(args.out_dir / "unknown.npz", runs=np.array(runs), **res)

    import matplotlib

    matplotlib.use("Agg")
    A = Analysis(res, args.tol)
    plot_solvers(args.out_dir / "unknown_solvers.png", A, "iter10000", args.tol_precise)
    plot_twinchers(args.out_dir / "unknown_twinchers.png", A)
    print_table(A, runs, args.tol_precise)
    print(f"outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
