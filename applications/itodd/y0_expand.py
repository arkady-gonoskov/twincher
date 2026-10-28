"""ChartY0 of the connector: from the reduced charts to all four components of p.

Setting found with the reduced charts (y0_reduced.py: n_p = 2, elevation and azimuth):
stencil 8 x 8 probes with flat kernels (n_y 64), finite-difference step 0.025; grid
data interpolated multilinearly with about 32 nodes for the elevation, 96 for the
azimuth, 24 for the roll and 8 for the distance (90 % of the tangent errors of the
interpolation below ~0.1-0.2, grid_tangent.py); the hs loss of twincher 0.1.0
(HS_0_1_0) with a fine curriculum (ramp_size 0.05, i.e. start at R 0.05, p_range_step
0.0025); n_s 128, n_l
256 (with n_l 64 branch points at the views along the bores, winding number 2, kept R
below 0.75); lr 1e-3, n_c 5120. R held at 0.86, stop after 6 clean strides; exact checks
(det J at 8192 points, the solver with 40 iterations on 256 views) every 2500
iterations (chart_training.py).

Runs:
- d123: n_p = 3, elevation, azimuth and roll (distance at the centre of its range),
  grid (32, 96, 24);
- full: n_p = 4, grid (8, 32, 96, 24);
- d123_l512: as d123 with n_l 512 (d123 folded at R 0.515 and stayed there with
  branch lines at e 0, phi +-0.25 along the roll);
- d123_lr5: as d123 with lr 0.5e-3 (in 2D the largest fold had 8787 instead of 520634
  negative determinants per stride; d123_l512 folded at R 0.532 too).

The script can be interrupted at any time and started again (completed runs are
skipped, an interrupted run continues from its last checkpoint).

Usage:
    nohup python -u y0_expand.py > output_y0_expand/series.log 2>&1 &
    python y0_expand.py --summary
"""

from pathlib import Path

from chart_training import HS_0_1_0, run_series

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_y0_expand"
BASE = {
    "obj": 26,
    "chart": "Y0",
    "margin": 1.0,
    "dims": None,
    "scale": 1.0,
    "data": "grid",
    "grid": [8, 32, 96, 24],
    "data_dir": str(OUT / "data"),
    "kernel": "flat",
    "probes": 8,
    "spectral": None,
    "dp": 0.025,
    "n_s": 128,
    "n_l": 256,
    "n_c": 5120,
    "lr": 1e-3,
    "seed": 42,
    "loss_params": {**HS_0_1_0, "ramp_size": 0.05, "p_range_step": 0.0025},
    "max_iters": 60000,
    "stride": 500,
    "exact_stride": 2500,
    "r_stop": 0.86,
    "done_strides": 6,
    "check_box": 0.86,
    "solver_check": 256,
    "solver_iters": 40,
    "solver_clamp": 0.86,
    "eval_boxes": [0.5, 0.86],
}
RUNS = {
    "d123": {**BASE, "dims": [1, 2, 3], "grid": [32, 96, 24], "max_iters": 40000},
    "full": BASE,
}
RUNS["d123_l512"] = {**RUNS["d123"], "n_l": 512}
RUNS["d123_lr5"] = {**RUNS["d123"], "lr": 0.5e-3}

if __name__ == "__main__":
    run_series(RUNS, OUT, __doc__.split("\n\n")[0])
