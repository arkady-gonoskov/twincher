"""Reduced ChartY0 of the connector: resolution of the training grid against wrinkles.

ChartY0 restricted to some components of p (train_chart.ReducedStencil; the others at
the centre of their ranges), stencil 8 x 8 probes with flat kernels (n_y 64), finite-
difference step 0.025 (in the full p), grid data interpolated multilinearly, the hs loss
of twincher 0.1.0 (HS_0_1_0) with its strict promotion, n_s 128, n_l 64, lr 1e-3, n_c
5120. R is held at r_stop once reached; exact checks
(det J at 8192 points, the solver with 40 iterations on 256 views) every 2000
iterations (chart_training.py).

Runs (name: dims, scale, grid):
- n_p = 2, elevation and azimuth (dims 1, 2):
  - s25_g8, s25_g12, s25_g16: a quarter of the ranges, 8^2, 12^2, 16^2 nodes (R to 1);
  - g16, g16x48, g24x48, g32x96: the full ranges (R to 0.86);
  - pool: the full ranges, 2^16 exact samples instead of a grid (the limit of an
    infinitely fine grid);
  - g16x48_c, g24x48_c, g32x96_c, pool_c: the curriculum of the quarter-range runs in
    units of the full ranges: ramp_size 0.05 (start at R 0.05), p_range_step 0.0025
    (the full-range runs with the defaults stayed at R 0.2, the start of the default
    curriculum);
  - c40, c40_lr5, c40_l256, c40_l256_lr5: as g32x96_c (which folded at R 0.53, as
    did g24x48_c and pool_c) with 40000 iterations: n_l 64 or 256, lr 1e-3 or 0.5e-3;
  - c40_pt50: as c40 with promotion_threshold 50 (before the folds, batches with a few
    det < 0 kept recurring while R grew: 11 clean batches in a row were enough);
  - z0_c40: control, ChartZ0 (trains in 4D) in the setting of c40;
  - sep, sep_a30: c40 with the antipodal separation loss (hs_sep_loss.py;
    kappa 0.3, |d| in [0.01, 0.05], amp 3 or 30) against branch points (winding 2 of r
    around the views along a bore axis, found in c40 and c40_pt50);
  - g32x96_sep: the default curriculum (start at R 0.2, where g32x96 stalled with a
    branch point) with the separation loss (amp 30).

The script can be interrupted at any time and started again (completed runs are
skipped, an interrupted run continues from its last checkpoint).

Usage:
    nohup python -u y0_reduced.py > output_y0_reduced/series.log 2>&1 &
    python y0_reduced.py --summary
"""

from pathlib import Path

from chart_training import HS_0_1_0, run_series

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_y0_reduced"
BASE = {
    "obj": 26,
    "chart": "Y0",
    "margin": 1.0,
    "dims": [1, 2],
    "scale": 1.0,
    "data": "grid",
    "grid": [16, 16],
    "data_dir": str(OUT / "data"),
    "kernel": "flat",
    "probes": 8,
    "spectral": None,
    "dp": 0.025,
    "n_s": 128,
    "n_l": 64,
    "n_c": 5120,
    "lr": 1e-3,
    "seed": 42,
    "loss_params": HS_0_1_0,
    "max_iters": 12000,
    "stride": 500,
    "exact_stride": 2000,
    "r_stop": 0.86,
    "done_strides": 6,
    "check_box": 0.86,
    "solver_check": 256,
    "solver_iters": 40,
    "solver_clamp": 0.86,
    "eval_boxes": [0.5, 0.86],
}
QUARTER = {**BASE, "scale": 0.25, "r_stop": 1.0, "check_box": 1.0,
           "solver_clamp": 1.0, "eval_boxes": [0.5, 1.0]}  # fmt: skip
RUNS = {
    "s25_g8": {**QUARTER, "grid": [8, 8]},
    "s25_g12": {**QUARTER, "grid": [12, 12]},
    "s25_g16": {**QUARTER, "grid": [16, 16]},
    "g16": BASE,
    "g16x48": {**BASE, "grid": [16, 48]},
    "g24x48": {**BASE, "grid": [24, 48]},
    "g32x96": {**BASE, "grid": [32, 96]},
    "pool": {**BASE, "data": "pool", "pool_size": 2**16, "pool_dir": str(OUT / "data")},
}
CURRICULUM = {**HS_0_1_0, "ramp_size": 0.05, "p_range_step": 0.0025}
for name in ("g16x48", "g24x48", "g32x96", "pool"):
    RUNS[f"{name}_c"] = {**RUNS[name], "loss_params": CURRICULUM}
C40 = {**RUNS["g32x96_c"], "max_iters": 40000, "exact_stride": 4000}
RUNS["c40"] = C40
RUNS["c40_lr5"] = {**C40, "lr": 0.5e-3}
RUNS["c40_l256"] = {**C40, "n_l": 256}
RUNS["c40_l256_lr5"] = {**C40, "n_l": 256, "lr": 0.5e-3}
RUNS["c40_pt50"] = {**C40, "loss_params": {**CURRICULUM, "promotion_threshold": 50}}
RUNS["z0_c40"] = {**C40, "chart": "Z0"}
SEP = {"amp": 3.0, "kappa": 0.3, "d_min": 0.01, "d_max": 0.05}
RUNS["sep"] = {**C40, "extra_terms": {"hs_sep": SEP}}
RUNS["sep_a30"] = {**C40, "extra_terms": {"hs_sep": {**SEP, "amp": 30.0}}}
RUNS["g32x96_sep"] = {**C40, "loss_params": HS_0_1_0,
                      "extra_terms": {"hs_sep": {**SEP, "amp": 30.0}}}  # fmt: skip

if __name__ == "__main__":
    run_series(RUNS, OUT, __doc__.split("\n\n")[0])
