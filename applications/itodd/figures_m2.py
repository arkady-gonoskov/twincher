"""Figures of milestone 2 in README.md (figures/m2_*.png), from the outputs of the runs.

Needs the outputs of y0_reduced.py, y0_fb.py, y0_full.py, y0_expand.py, y0_vortex.py
and train_charts_fb.py (history files, twinchers and traces) and a GPU for the figures
that render views (views, wrinkles, vortices, fb).

Usage:
    python figures_m2.py [views wrinkles vortices fb curves explosions charts]
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import twincher
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from depth_stencil import FB_DEFAULTS, fb_components
from train_chart import chart_stencil
from winding import LoopWindings
from wrinkle_map import TorchMultilinearInterpolator, grid_points, render
from wrinkle_map import r_and_det as r_det

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"
# diverging blue <-> red with a gray midpoint (det J: positive blue, negative red)
DIVERGING = LinearSegmentedColormap.from_list(
    "det", ["#d03b3b", "#f0efec", "#2a78d6"], N=256
)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": False, "font.size": 9,
    "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})  # fmt: skip


def save(fig, name):
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"figures/{name}")


def column(rows, key):
    return np.array([float(r[key]) for r in rows])


def read_csv(file):
    with open(file, newline="") as f:
        return list(csv.DictReader(f))


def plane(box, G):
    """Points of a G x G grid of the (elevation, azimuth) plane, |p| <= box."""
    x = np.linspace(-box, box, G)
    E, F = np.meshgrid(x, x, indexing="ij")  # rows: elevation, columns: azimuth
    return x, torch.as_tensor(np.stack([E.ravel(), F.ravel()], 1))


def mesh(ax, r, G, step=2, alpha=0.55):
    """The images of the lines of constant elevation and azimuth."""
    rr = r.reshape(G, G, 2)[::step, ::step]
    segs = [rr[i] for i in range(len(rr))] + [rr[:, j] for j in range(rr.shape[1])]
    ax.add_collection(LineCollection(segs, linewidths=0.4, colors=INK, alpha=alpha))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["left", "bottom"]].set_visible(False)


def det_map(ax, det, box, title, nodes=None):
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.5, vmax=1.5)  # the same in all panels
    im = ax.imshow(det, origin="lower", extent=(-box, box, -box, box), cmap=DIVERGING,
                   norm=norm)  # fmt: skip
    if nodes is not None:
        for v in nodes[1][np.abs(nodes[1]) <= box]:
            ax.axvline(v, color=MUTED, lw=0.3, alpha=0.6)
        for v in nodes[0][np.abs(nodes[0]) <= box]:
            ax.axhline(v, color=MUTED, lw=0.3, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("azimuth $p_2$")
    ax.set_ylabel("elevation $p_1$")
    return im


def fig_views():
    """Depth maps of the connector across the azimuth of ChartZ0 and ChartY0."""
    phis = [-0.6, -0.3, 0.0, 0.3, 0.6]
    fig, axes = plt.subplots(2, len(phis), figsize=(11, 4.6))
    for row, chart in zip(axes, ("Z0", "Y0")):
        st = chart_stencil(chart, 26, kernel="flat", n_grid=8, dp=0.025)
        p = torch.zeros((len(phis), 4), dtype=torch.float64)
        p[:, 2] = torch.tensor(phis)
        z = st.depth_image(p).cpu().numpy()
        for ax, zi, ph in zip(row, z, phis):
            ax.imshow(np.where(zi > 0, zi, np.nan), cmap="Greys_r", vmin=z[z > 0].min(),
                      vmax=z[z > 0].max())  # fmt: skip
            ax.set_axis_off()
            ax.set_title(
                f"{chart}: azimuth p2 = {ph:+.1f} ({ph * 150:+.0f}°)", fontsize=8
            )
    fig.suptitle("The connector (ITODD object 26) seen across the azimuth of two charts "
                 "(elevation and roll 0; lighter = farther)", fontsize=10)  # fmt: skip
    save(fig, "m2_views.png")


def fig_wrinkles():
    """Wrinkles: det J on the interpolated training data and on exact data, the tangent
    error of the interpolation; training grids of 8 and 16 nodes per component."""
    G, base = 96, HERE / "output_y0_reduced"
    _, p = plane(1.0, G)
    st = chart_stencil("Y0", 26, kernel="flat", n_grid=8, dp=0.025, dims=[1, 2],
                       scale=0.25)  # fmt: skip
    y, d = render(st, p, batch=128)
    grid = [8, 8]
    yg, dg = render(st, torch.as_tensor(grid_points(grid)), batch=128)
    mli = TorchMultilinearInterpolator(2, grid, st.device)
    yi, di = torch.empty_like(y), torch.empty_like(d)
    mli.interpolate(p.float().to(st.device), yi, yg)
    mli.interpolate(p.float().to(st.device), di, dg)
    tan = ((di - d).norm(dim=1) / d.norm(dim=1).clamp(min=1e-9)).amax(dim=1)
    _, det_i = r_det(base / "s25_g8" / "best.twc", yi, di)
    _, det8 = r_det(base / "s25_g8" / "best.twc", y, d)
    _, det16 = r_det(base / "s25_g16" / "best.twc", y, d)
    nodes8, nodes16 = [np.linspace(-1, 1, 8)] * 2, [np.linspace(-1, 1, 16)] * 2
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.0), layout="constrained")
    panels = [(axes[0], det_i, "8 x 8 nodes: det J on the\ntraining data (interpolated)",
               nodes8),
              (axes[1], det8, "8 x 8 nodes: det J on exact data", nodes8),
              (axes[3], det16, "16 x 16 nodes: det J on exact data", nodes16)]  # fmt: skip
    for ax, det, title, nodes in panels:
        im = det_map(ax, det.reshape(G, G), 1.0,
                     f"{title}\ndet J <= 0 on {100 * (det <= 0).mean():.1f} % of p",
                     nodes)  # fmt: skip
        fig.colorbar(im, ax=ax, location="bottom", shrink=0.8, label="det J")
    ax = axes[2]
    im = ax.imshow(tan.cpu().numpy().reshape(G, G), origin="lower",
                   extent=(-1, 1, -1, 1), cmap="Blues", vmin=0, vmax=0.8)  # fmt: skip
    for v in nodes8[0]:
        ax.axvline(v, color=MUTED, lw=0.3, alpha=0.6)
        ax.axhline(v, color=MUTED, lw=0.3, alpha=0.6)
    ax.set_title("8 x 8 nodes: tangent error of the\ninterpolation of dy/dp")
    ax.set_xlabel("azimuth $p_2$")
    ax.set_ylabel("elevation $p_1$")
    fig.colorbar(im, ax=ax, location="bottom", shrink=0.8,
                 label="max_k |dy/dp_k (interpolated) - dy/dp_k| / |dy/dp_k|")  # fmt: skip
    fig.suptitle("Wrinkles: ChartY0 reduced to elevation and azimuth over a quarter of "
                 "their ranges (64 layers); thin lines: the training nodes",
                 fontsize=10)  # fmt: skip
    save(fig, "m2_wrinkles.png")


def windings(twc, st, box, Gc=40):
    """Winding numbers of r around the circles of a tiling of the box (as in the checks
    of chart_training.py: centres of Gc x Gc cells, radius 0.75 cell widths)."""
    c = ((torch.arange(Gc, dtype=torch.float64) + 0.5) / Gc * 2 - 1) * box
    C = torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)
    sh = twincher.Shuttle()
    sh.load(str(twc), n_c=1, n_p=2, n_m=2)
    lw = LoopWindings(st, str(twc), len(C), n_t=24)
    w = lw(sh.a, C, torch.tensor([[0, 1]] * len(C)), 0.75 * 2 * box / Gc)
    return w.cpu().numpy().reshape(Gc, Gc)


def fig_vortices():
    """r meshes and winding numbers: a twincher with vortices and a clean one; a zoom on
    a vortex with the image of a small circle around it."""
    G = 64
    st = chart_stencil("Y0", 26, kernel="flat", n_grid=8, dp=0.025, dims=[1, 2])
    runs = [("c40", "final.twc", 0.72, "64 layers, stalled at R 0.735"),
            ("c40_l256", "best.twc", 0.86, "256 layers, R 0.86")]  # fmt: skip
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.2), layout="constrained")
    cmap = LinearSegmentedColormap.from_list(
        "w", ["#d03b3b", "#eda100", "#f0efec", "#2a78d6", "#1c5cab"], N=5
    )
    wmaps = []
    for col, (run, file, box, label) in zip((0, 2), runs):
        twc = HERE / "output_y0_reduced" / run / file
        _, p = plane(box, G)
        y, d = render(st, p, batch=128)
        r, _ = r_det(twc, y, d)
        mesh(axes[0, col], r, G)
        axes[0, col].set_title(f"r(y(p)) over |p| <= {box}, {label}")
        w = windings(twc, st, box)
        wmaps.append(w)
        im = axes[1, col].imshow(w, origin="lower", extent=(-box, box, -box, box),
                                 cmap=cmap, vmin=-1.5, vmax=3.5)  # fmt: skip
        axes[1, col].set_title(f"winding numbers: != 1 for {100 * (w != 1).mean():.1f} % "
                               "of the circles")  # fmt: skip
        axes[1, col].set_xlabel("azimuth $p_2$")
        axes[1, col].set_ylabel("elevation $p_1$")
    cb = fig.colorbar(im, ax=axes[1, 2], shrink=0.8, ticks=[-1, 0, 1, 2, 3])
    cb.set_label("winding number of r around a small circle")
    # zoom on a vortex (winding 2) of the 64-layer twincher
    twc, box, w = HERE / "output_y0_reduced/c40/final.twc", runs[0][2], wmaps[0]
    Gc = len(w)
    i, j = np.argwhere(w == 2)[0]
    c = ((np.array([i, j]) + 0.5) / Gc * 2 - 1) * box
    rho, h, Gz = 0.75 * 2 * box / Gc, 0.045, 31
    xs = np.linspace(-h, h, Gz)
    E, F = np.meshgrid(c[0] + xs, c[1] + xs, indexing="ij")
    y, d = render(st, torch.as_tensor(np.stack([E.ravel(), F.ravel()], 1)), batch=128)
    r, det = r_det(twc, y, d)
    t = np.linspace(0, 2 * np.pi, 241)
    pc = torch.as_tensor(np.stack([c[0] + rho * np.sin(t), c[1] + rho * np.cos(t)], 1))
    y, d = render(st, pc, batch=128)
    rc, _ = r_det(twc, y, d)
    ax = axes[0, 1]
    mesh(ax, r, Gz, step=1, alpha=0.25)
    ax.plot(rc[:, 0], rc[:, 1], color=SERIES[1], lw=1.6)
    for k in (0, 60, 120, 180):
        ax.annotate(
            "",
            rc[k + 5],
            rc[k],
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.6,
                "color": SERIES[1],
                "mutation_scale": 16,
            },
        )
    ax.set_title("zoom: r near a vortex; orange: the image\nof the small circle around it "
                 "(winds twice)")  # fmt: skip
    ax = axes[1, 1]
    im = ax.imshow(det.reshape(Gz, Gz), origin="lower", cmap=DIVERGING,
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-0.5, vmax=1.5),
                   extent=(c[1] - h, c[1] + h, c[0] - h, c[0] + h))  # fmt: skip
    ax.plot(c[1] + rho * np.cos(t), c[0] + rho * np.sin(t), color=SERIES[1], lw=1.6)
    ax.set_title(f"det J near the vortex (e {c[0]:+.2f}, azimuth {c[1]:+.2f})\n"
                 "orange: the small circle in p")  # fmt: skip
    ax.set_xlabel("azimuth $p_2$")
    ax.set_ylabel("elevation $p_1$")
    fig.colorbar(im, ax=ax, shrink=0.8, label="det J")
    fig.suptitle("Vortices (branch points): ChartY0 reduced to elevation and azimuth "
                 "(full ranges), flat 8 x 8 stencil", fontsize=10)  # fmt: skip
    save(fig, "m2_vortices.png")


def bessel_table(m, j, n=2001):
    import mpmath

    rho = np.linspace(0, 1, n)
    return rho, np.array([float(mpmath.besselj(m, j * r)) for r in rho])


def fig_fb():
    """The Fourier-Bessel stencil: heights with Z_bg and with the near background, and
    the heights reconstructed from the 64 components."""
    import mpmath

    p = torch.tensor([[0.0, 0.0, 0.25, 0.0]], dtype=torch.float64)
    st_far = chart_stencil("Y0", 26, dp=0.025, fb={"n": 64})
    st_near = chart_stencil("Y0", 26, dp=0.025, fb={"n": 64, "background": 20.0})
    z = st_far.depth_image(p)
    mask = z > 0
    cnt = mask.float().sum()
    xs = torch.arange(st_far.res, device=z.device) + 0.5
    cx = (mask.float().sum(dim=1) * xs).sum() / cnt
    cy = (mask.float().sum(dim=2) * xs).sum() / cnt
    centre = torch.stack([cx, cy])[None]
    scale = st_far.z_bg - st_far.z_lo
    zm = z[mask].mean()
    h_far = torch.where(mask, (st_far.z_bg - z) / scale, 0.0)[0].cpu().numpy()
    h_near = torch.where(mask, (zm + 20 - z).clamp(min=0) / scale, 0.0)[0].cpu().numpy()
    f, _ = st_near._fb_coeffs(z, centre)
    f = f[0].cpu().numpy()
    # reconstruction on the pixels of the disk about the centre of mass
    res = st_far.res
    yy, xx = np.mgrid[0:res, 0:res] + 0.5
    dx, dy = xx - cx.item(), yy - cy.item()
    rho, theta = np.hypot(dx, dy) / (0.5 * res), np.arctan2(dy, dx)
    rec = np.zeros((res, res))
    for i, (n, m) in enumerate(st_near._fbt.modes):
        j = float(mpmath.besseljzero(m, n))
        norm = 1 / (math.sqrt(math.pi) * abs(float(mpmath.besselj(m + 1, j))))
        r_t, J = bessel_table(m, j)
        radial = norm * np.interp(np.clip(rho, 0, 1), r_t, J) * (rho <= 1)
        term = f[i] * radial * np.exp(1j * m * theta)
        rec += term.real if m == 0 else 2 * term.real
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    zz = z[0].cpu().numpy()
    vmax = max(h_far.max(), 1e-6)
    panels = [(np.where(zz > 0, zz, np.nan), "depth map (lighter = farther)", "Greys_r",
               None),
              (h_far, "heights above the far plane Z_bg", "Greys", (0, vmax)),
              (h_near, "heights above mean depth + 20 mm", "Greys", (0, h_near.max())),
              (rec, "from the 64 Fourier-Bessel components", "Greys",
               (0, h_near.max()))]  # fmt: skip
    for ax, (img, title, cmap, lim) in zip(axes, panels):
        kw = {} if lim is None else {"vmin": lim[0], "vmax": lim[1]}
        ax.imshow(img, cmap=cmap, **kw)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    circle = plt.Circle((cx.item(), cy.item()), 0.5 * res, fill=False, color=SERIES[1],
                        lw=0.8, ls="--")  # fmt: skip
    axes[3].add_patch(circle)
    fig.suptitle("ChartY0 view (azimuth +38°): the inner wall of the bore, deep behind "
                 "the front, is dropped by the near background", fontsize=10)  # fmt: skip
    save(fig, "m2_fb_stencil.png")
    print("fb components:", len(fb_components(64)), FB_DEFAULTS)


def r_curve(file):
    rows = read_csv(file)
    return (np.array([int(r["iter"]) for r in rows]),
            np.array([float(r["R"]) for r in rows]))  # fmt: skip


STYLES = ["-", "--", "-.", ":"]  # second encoding of the series (besides the colour)


def curves_panel(ax, runs, title, xmax=None):
    for k, (label, it, R) in enumerate(runs):
        ax.plot(it, R, color=SERIES[k], ls=STYLES[k], lw=1.8, label=label)
    ax.axhline(0.86, color=MUTED, lw=0.8, ls=":")
    ax.text(0, 0.87, "target R = 0.86", color=MUTED, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("curriculum range R")
    ax.set_ylim(0, 0.95)
    if xmax:
        ax.set_xlim(0, xmax)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower right", fontsize=8)


def out(group, run):
    return HERE / f"output_{group}" / run / "history.csv"


def fig_curves():
    """Growth of R for the main steps in 2D, 3D and 4D."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    curves_panel(axes[0], [
        ("flat 8x8, 64 layers", *r_curve(out("y0_reduced", "c40"))),
        ("flat 8x8, 256 layers", *r_curve(out("y0_reduced", "c40_l256"))),
        ("FB, 64 layers", *r_curve(out("y0_fb", "fb_bg20_l64"))),
        ("FB + trimmer everywhere, 64", *r_curve(out("y0_fb", "fbtrim_bg20_l64"))),
    ], "2D (elevation, azimuth)", 40000)  # fmt: skip
    curves_panel(axes[1], [
        ("flat 8x8, 256 layers", *r_curve(out("y0_expand", "d123"))),
        ("flat 8x8, 512 layers", *r_curve(out("y0_expand", "d123_l512"))),
        ("FB, 256 layers", *r_curve(out("y0_fb", "fb_bg20_d123"))),
        ("FB + trimmer everywhere", *r_curve(out("y0_fb", "fbtrim_bg20_d123"))),
    ], "3D (elevation, azimuth, roll)", 40000)  # fmt: skip
    it0, R0 = r_curve(out("y0_fb", "fbtrim_full"))
    it1, R1 = r_curve(out("y0_full", "l384"))
    keep = it1 <= 7500
    curves_panel(axes[2], [
        ("256 layers (exploded at 0.78)", it0, R0),
        ("from 25000: 384 layers", it1[keep] + 25000, R1[keep]),
    ], "4D ChartY0 (+ distance)", 60000)  # fmt: skip
    fig.suptitle("ChartY0: the curriculum range R during training (flat runs are stopped "
                 "where they stalled)", fontsize=10)  # fmt: skip
    save(fig, "m2_curves.png")


def fig_explosions():
    """What precedes an explosion: traces of the last 1000 (Y1: 500) iterations."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    cases = [(HERE / "output_y0_full/l384_clip/trace_3000.csv",
              "ChartY0, R held at 0.80: slow run-away"),
             (HERE / "output_charts_fb/Y1/trace_26500.csv",
              "ChartY1, step R 0.8325 -> 0.835 at 26029: fast onset")]  # fmt: skip
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.8), sharex="col")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for col, (file, title) in enumerate(cases):
        rows = read_csv(file)
        it = np.array([int(r["iter"]) for r in rows])
        ax = axes[0, col]
        ax.semilogy(it, column(rows, "trim_max"), color=SERIES[0], lw=1.0,
                    label="over the whole domain [-1, 1]^4 (trimmer batch)")  # fmt: skip
        ax.semilogy(it, column(rows, "hs_max"), color=SERIES[1], lw=1.0, ls="--",
                    label="inside the box [-R, R]^4 (training batch)")  # fmt: skip
        ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("largest |dr/dp_k| in the batch")
        ax.set_title(title)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.legend(fontsize=8, loc="upper left")
        ax = axes[1, col]
        neg = column(rows, "neg")
        ax.bar(it[neg > 0], neg[neg > 0], width=1.5, color=SERIES[0])
        ax.set_yscale("log")
        ax.set_ylim(0.7, 5120)
        ax.yaxis.set_major_formatter(plain)
        ax.set_ylabel("cases with det J < 0\n(of 5120 per batch)")
        ax.set_xlabel("iteration")
        ax.grid(axis="y", color=GRID, lw=0.5)
    fig.suptitle("Traces written by chart_training.py at the onset of an explosion "
                 "(one value per iteration)", fontsize=10)  # fmt: skip
    save(fig, "m2_explosions.png")


def fig_charts():
    """The six charts of the final recipe."""
    base = HERE / "output_charts_fb"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    curves_panel(axes[0], [(c, *r_curve(base / c / "history.csv"))
                           for c in ("Z0", "Z1", "Z2")], "Z charts", 33000)  # fmt: skip
    it0, R0 = r_curve(HERE / "output_y0_fb/fbtrim_full/history.csv")
    it1, R1 = r_curve(HERE / "output_y0_full/l384/history.csv")
    keep0, keep1 = it0 <= 25000, it1 <= 7500
    y0 = (np.concatenate([it0[keep0], it1[keep1] + 25000]),
          np.concatenate([R0[keep0], R1[keep1]]))  # fmt: skip
    curves_panel(axes[1], [("Y0 (256, then 384 layers)", *y0),
                           *[(c, *r_curve(base / c / "history.csv"))
                             for c in ("Y1", "Y2")]], "Y charts", 33000)  # fmt: skip
    axes[1].set_ylabel("")
    fig.suptitle("All six charts of the connector reach R = 0.86", fontsize=10)
    save(fig, "m2_six_charts.png")


def fig_momentum():
    """The Adam momentum study (adam_momentum.py): from the state before an explosion,
    the control, beta_1 0.9 and the momentum reset; gray: the replay before it."""
    base = HERE / "output_momentum"
    cases = [("l384", "ChartY0, 384 layers, R held at 0.86", 9500),
             ("y2", "ChartY2, R growing", 24500)]  # fmt: skip
    variants = [("ctrl", "control (beta_1 0.99)"), ("b09", "beta_1 0.9"),
                ("reset", "momentum reset")]  # fmt: skip
    keys = [("neg_dets", "cases with det J < 0\nper stride (2.56 M cases)"),
            ("g_max", "largest gradient norm\nin the stride"),
            ("R", "curriculum range R")]  # fmt: skip
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.4), sharex="col")
    for col, (case, title, start) in enumerate(cases):
        rows = [r for r in read_csv(base / f"replay_{case}" / "history.csv")
                if start - 3000 <= int(r["iter"]) <= start]  # fmt: skip
        series = [("replay before the start", MUTED, "-", rows)]
        for k, (v, label) in enumerate(variants):
            rows = read_csv(base / f"{case}_{v}" / "history.csv")
            series.append((label, SERIES[k], STYLES[k], rows))
        for label, color, ls, rows in series:
            it = np.array([int(r["iter"]) for r in rows])
            for row, (key, _) in enumerate(keys):
                val = column(rows, key)
                if key == "neg_dets":
                    val = np.maximum(val, 0.5)  # zero drawn at the bottom of the axis
                axes[row, col].plot(it, val, color=color, ls=ls, lw=1.6, marker=".",
                                    ms=4, label=label)  # fmt: skip
        axes[0, col].set_title(title)
        for row, (key, ylabel) in enumerate(keys):
            ax = axes[row, col]
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color=GRID, lw=0.5)
            ax.axvline(start, color=MUTED, lw=0.8, ls=":")
            if key != "R":
                ax.set_yscale("log")
        axes[0, col].set_ylim(0.4, 3e6)
        axes[2, col].set_xlabel("iteration (end of each stride of 500)")
        axes[0, col].legend(fontsize=8, loc="upper left")
    fig.suptitle("Adam's momentum and the explosions: three variants from the state before "
                 "an explosion (no rollback)", fontsize=10)  # fmt: skip
    save(fig, "m2_momentum.png")


FIGURES = {"views": fig_views, "wrinkles": fig_wrinkles, "vortices": fig_vortices,
           "fb": fig_fb, "curves": fig_curves, "explosions": fig_explosions,
           "charts": fig_charts, "momentum": fig_momentum}  # fmt: skip

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
        torch.cuda.empty_cache()
