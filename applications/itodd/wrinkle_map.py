"""Maps of an n_p = 2 twincher of a reduced chart (train_chart.ReducedStencil) over p.

On a fine G x G grid of [-box, box]^2 (y and dy/dp rendered, no interpolation):
- r = the first two components of the twincher output (the mesh of HSMonitorP2);
- det J with the exact y and dy/dp, and with y and dy/dp interpolated (multilinear, as
  in training) from the training grid --grid;
- the tangent error of the interpolation: max_k |dy/dp_k(int) - dy/dp_k| / |dy/dp_k|;
- the winding number of r around small circles (radius 0.02, 24 points) in p centred
  on every second point of the grid: 1 for an embedding, otherwise a branch point
  (e.g. 2: r winds twice, like z -> z^2) or a fold;
- the solver (start p = 0, --n-iter iterations, p clamped to [-1, 1]) for the views
  of every second point of the grid: max |p - p_true| in units of the full p (times
  the scale of the reduced chart).
Prints statistics and writes a figure (--out, default TWC_STEM_wrinkles.png next to the
twincher).

The stencil is taken from the metadata of the twincher if it has one (chart_training),
otherwise from the options (twinchers of other scripts).

Usage:
    python wrinkle_map.py TWC --grid 8 8 [--scale 0.25] [--resolution 1024] [--dp 0.0125]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import twincher
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, TwoSlopeNorm

from multires_interpolator import TorchMultilinearInterpolator, grid_points
from train_chart import chart_stencil, stencil_from_metadata


def render(st, p, batch=512):
    y = torch.empty((len(p), st.n_y), device=st.device)
    d = torch.empty((len(p), st.n_y, st.n_p), device=st.device)
    for i in range(0, len(p), batch):
        st.compute(p[i : i + batch], y[i : i + batch], d[i : i + batch])
    return y, d


def r_and_det(twc, y, d):
    n, n_y, n_p = d.shape
    sh = twincher.Shuttle()
    sh.load(str(twc), n_c=n, n_p=n_p, n_m=n_p)
    sh.s.fill_(0)
    sh.s[:n_y].copy_(y.T)
    sh.Q.fill_(0)
    sh.Q[:n_y].copy_(d.permute(1, 2, 0))
    sh.forward_query()
    det = torch.linalg.det(sh.Q[:n_p].permute(2, 1, 0))
    return sh.s[:n_p].T.cpu().numpy(), det.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("twc", type=Path)
    parser.add_argument("--grid", type=int, nargs=2, required=True)
    parser.add_argument("--chart", default="Y0")
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--kernel", default="flat")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dp", type=float, default=0.025)
    parser.add_argument("--dims", type=int, nargs=2, default=[1, 2])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--G", type=int, default=96)
    parser.add_argument("--box", type=float, default=1.0)
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    meta = twincher.load_metadata(str(args.twc))
    if "stencil" in meta:
        st = stencil_from_metadata(meta["stencil"])
    else:
        st = chart_stencil(args.chart, 26, kernel=args.kernel, n_grid=args.probes,
                           resolution=args.resolution, dp=args.dp, dims=args.dims,
                           scale=args.scale)  # fmt: skip
    if st.n_p != 2:
        raise SystemExit("only for n_p = 2")
    G = args.G
    x = args.box * np.linspace(-1, 1, G)
    p = args.box * torch.as_tensor(grid_points([G, G]))  # p0 slow, p1 fast
    y, d = render(st, p)
    r, det = r_and_det(args.twc, y, d)
    yg, dg = render(st, torch.as_tensor(grid_points(args.grid)))
    mli = TorchMultilinearInterpolator(2, args.grid, st.device)
    yi, di = torch.empty_like(y), torch.empty_like(d)
    mli.interpolate(p.float().to(st.device), yi, yg)
    mli.interpolate(p.float().to(st.device), di, dg)
    _, det_i = r_and_det(args.twc, yi, di)
    tan = ((di - d).norm(dim=1) / d.norm(dim=1).clamp(min=1e-9)).amax(dim=1)
    tan = tan.cpu().numpy()
    # the solver for every second point of the fine grid
    sub = np.zeros((G, G), dtype=bool)
    sub[::2, ::2] = True
    ps = p[sub.ravel()]
    err = solve_at(args.twc, st, ps, args.n_iter) * st.scale
    wind = winding(args.twc, st, ps)

    det, det_i, tan = det.reshape(G, G), det_i.reshape(G, G), tan.reshape(G, G)
    print(f"{args.twc}  (grid {args.grid}, G {G})")
    print(f"  det <= 0: exact {100 * (det <= 0).mean():.2f} %, interpolated "
          f"{100 * (det_i <= 0).mean():.2f} %; det <= 0.5: exact "
          f"{100 * (det <= 0.5).mean():.2f} %, interpolated {100 * (det_i <= 0.5).mean():.2f} %")  # fmt: skip
    print(f"  tangent error: median {np.median(tan):.2f}, 90 % {np.quantile(tan, 0.9):.2f},"
          f" max {tan.max():.2f}; at det <= 0 (exact): "
          f"{np.median(tan[det <= 0]) if (det <= 0).any() else float('nan'):.2f} (median)")  # fmt: skip
    bins = [0, 1e-3, 1e-2, 0.1, np.inf]
    frac = np.histogram(err, bins)[0] / len(err)
    print("  solver error < 1e-3: {:.1f} %, 1e-3..1e-2: {:.1f} %, 1e-2..0.1: {:.1f} %, "
          "> 0.1: {:.1f} %".format(*(100 * frac)))  # fmt: skip
    values, counts = np.unique(wind, return_counts=True)
    print("  winding numbers (count): "
          + ", ".join(f"{v:+.0f} ({c})" for v, c in zip(values, counts)))  # fmt: skip
    out = args.out or args.twc.with_name(args.twc.stem + "_wrinkles.png")
    plot(out, x, r, det, det_i, tan,
         err, sub, args.grid, args.box, wind)  # fmt: skip


def winding(twc, st, centres, rho=0.02, n_loop=24, batch=4096):
    """Winding number of r around circles of radius rho in p centred at centres."""
    t = torch.linspace(0, 2 * np.pi, n_loop + 1, dtype=torch.float64)[:-1]
    off = rho * torch.stack([t.cos(), t.sin()], dim=-1)
    p = (centres[:, None] + off[None]).reshape(-1, 2)
    y = torch.cat([st.forward(p[i : i + batch])[0] for i in range(0, len(p), batch)])
    sh = twincher.Shuttle()
    sh.load(str(twc), n_c=len(p), n_p=2, n_m=1)
    sh.s.fill_(0)
    sh.s[: st.n_y].copy_(y.T)
    sh.forward_inference()
    r = sh.s[:2].T.double().cpu().reshape(len(centres), n_loop, 2)
    r = r - r.mean(dim=1, keepdim=True)
    angle = torch.atan2(r[..., 1], r[..., 0])
    step = torch.diff(torch.cat([angle, angle[:, :1]], dim=1), dim=1)
    step = (step + np.pi) % (2 * np.pi) - np.pi
    return (step.sum(dim=1) / (2 * np.pi)).round().numpy()


def solve_at(twc, st, p_true, n_iter, batch=1024):
    """max |p - p_true| of the solver (start p = 0, p clamped to [-1, 1])."""
    err = []
    for i in range(0, len(p_true), batch):
        q = p_true[i : i + batch]
        solver = twincher.solver(str(twc), n_c=len(q))
        y_true, _ = st.forward(q)
        solver.start_solution(y_true)
        for _ in range(n_iter):
            st.compute(solver.p, solver.y, solver.dy_dp)
            solver.step(y=solver.y, dy_dp=solver.dy_dp)
        err.append((solver.p.double().cpu() - q).abs().max(dim=1).values.numpy())
    return np.concatenate(err)


def plot(file, x, r, det, det_i, tan, err, sub, grid, box=1.0, wind=None):
    G = len(x)
    fig, axes = plt.subplots(2, 4, figsize=(21, 10.5))
    ax = axes[0, 0]  # mesh of r (lines of constant p0 and p1, every second point)
    rr = r.reshape(G, G, 2)[::2, ::2]
    segs = [rr[i, :] for i in range(len(rr))] + [rr[:, j] for j in range(len(rr))]
    ax.add_collection(LineCollection(segs, linewidths=0.4, colors="k", alpha=0.6))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title("r(y(p)), exact y")
    ax.set_xlabel("$r_0$")
    ax.set_ylabel("$r_1$")
    ext = (-box, box, -box, box)
    nodes = [v[np.abs(v) <= box] for v in (np.linspace(-1, 1, n) for n in grid)]

    def field(ax, f, title, **kw):
        im = ax.imshow(f, origin="lower", extent=ext, **kw)  # rows: p0, columns: p1
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel("$p_1$")
        ax.set_ylabel("$p_0$")
        for v in nodes[1]:
            ax.axvline(v, color="w", lw=0.3, alpha=0.6)
        for v in nodes[0]:
            ax.axhline(v, color="w", lw=0.3, alpha=0.6)

    lim = max(1.0, np.quantile(np.abs(det), 0.99))
    norm = TwoSlopeNorm(vcenter=0.5, vmin=min(det.min(), det_i.min(), 0.0) - 1e-6,
                        vmax=lim)  # fmt: skip
    field(axes[0, 1], det, "det J, exact y and dy/dp", cmap="RdBu", norm=norm)
    field(axes[0, 2], det_i, "det J, interpolated (training data)", cmap="RdBu",
          norm=norm)  # fmt: skip
    field(axes[1, 0], tan, "tangent error of the interpolation", cmap="magma",
          vmin=0, vmax=min(2.0, tan.max()))  # fmt: skip
    e = np.full((G, G), np.nan)
    e[sub] = err
    field(axes[1, 1], np.clip(e, 1e-5, 1), "solver error max|p - p_true| (full p)",
          cmap="viridis", norm=LogNorm(1e-5, 1))  # fmt: skip
    w = np.full((G, G), np.nan)
    w[sub] = wind
    field(axes[0, 3], w, "winding number of r around circles (radius 0.02)",
          cmap="coolwarm", vmin=-1, vmax=3)  # fmt: skip
    axes[1, 3].axis("off")
    ax = axes[1, 2]
    field(ax, det, "exact det J with solver errors > 1e-2", cmap="RdBu", norm=norm)
    pp = np.stack(np.meshgrid(x, x, indexing="ij"), -1)[sub]
    bad = err > 1e-2
    ax.scatter(pp[bad, 1], pp[bad, 0], s=6, c="k", marker="x", lw=0.6)
    fig.tight_layout()
    fig.savefig(file, dpi=110)
    plt.close(fig)
    print(f"  figure: {file}")


if __name__ == "__main__":
    main()
