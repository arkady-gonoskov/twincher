"""Stencil variants for the embedding of ChartZ0 of the connector: a resumable series.

All runs: data interpolated from a (16, 16, 32, 16) grid of p (D, elevation, azimuth,
roll; multires_interpolator.py), finite-difference step dp = 0.025
(study_fd_step_chart.py), n_l 512, lr 0.5e-3, the hs loss of twincher 0.1.0, R held at 0.86
once reached (the six charts cover all views from |p|_inf <= 0.86 on) and training
continued until the batches inside R are clean for 6 strides (3000 iterations); exact
checks every 2500 iterations (det J at 8192 points and the solver, p clamped to the
trained box, on 256 views of |p|_inf <= 0.86); best.twc by the fewest solver failures
(chart_training.py).

Runs:
- flat16: flat kernels (mean depth of each cell), 16 x 16 probes (n_y 256), n_s 256,
  n_c 2048 (baseline);
- flat12, flat8: 12 x 12 (n_y 144) and 8 x 8 (n_y 64) probes, n_s = n_y;
- flat8_nc8192: as flat8 with n_c 8192;
- spec128, spec64, spec32, spec16: the lowest Fourier components of the 16 x 16 flat
  cells (spiral order), scaled by 0.2 (with the orthonormal FFT alone the lowest
  components vary 2-6 times faster with p than the probes of the flat kernel and the
  training did not start), n_s = n_y;
- spec64_Y0: as spec64 for ChartY0 (stopped: stayed at R 0.2);
- spec64_Y0_tol: as spec64_Y0 with a tolerant promotion (promotion_tol 0.002, no ramp).

The script can be interrupted at any time and started again (completed runs are
skipped, an interrupted run continues from its last checkpoint).

Usage:
    nohup python -u stencil_sweep.py > stencil_sweep.log 2>&1 &
    python stencil_sweep.py --summary
"""

from pathlib import Path

from chart_training import HS_0_1_0, run_series

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_stencil_sweep"
BASE = {
    "obj": 26,
    "chart": "Z0",
    "margin": 1.0,
    "data": "grid",
    "grid": [16, 16, 32, 16],
    "data_dir": str(OUT / "data"),
    "kernel": "flat",
    "probes": 16,
    "spectral": None,
    "dp": 0.025,
    "n_s": 256,
    "n_l": 512,
    "n_c": 2048,
    "lr": 0.5e-3,
    "seed": 42,
    "loss_params": HS_0_1_0,
    "max_iters": 40000,
    "stride": 500,
    "exact_stride": 2500,
    "r_stop": 0.86,
    "done_strides": 6,
    "check_box": 0.86,
    "solver_check": 256,
    "solver_clamp": 0.86,
    "eval_boxes": [0.8, 0.86],
}
RUNS = {
    "flat16": BASE,
    "flat12": {**BASE, "probes": 12, "n_s": 144},
    "flat8": {**BASE, "probes": 8, "n_s": 64},
    "flat8_nc8192": {**BASE, "probes": 8, "n_s": 64, "n_c": 8192},
    **{
        f"spec{n}": {**BASE, "spectral": n, "spectral_scale": 0.2, "n_s": n}
        for n in (128, 64, 32, 16)
    },
    # ChartY0 (non-injective with an exact-sample pool) with spec64
    "spec64_Y0": {
        **BASE,
        "chart": "Y0",
        "spectral": 64,
        "spectral_scale": 0.2,
        "n_s": 64,
    },
    # the same with a tolerant promotion (hs loss without a ramp): spec64_Y0
    # stayed at R 0.2, its rare det <= 0.5 (0.06 %) blocked the strict rule
    "spec64_Y0_tol": {
        **BASE,
        "chart": "Y0",
        "spectral": 64,
        "spectral_scale": 0.2,
        "n_s": 64,
        "loss_params": {
            "ramp_boost": 1.0,
            "promotion_tol": 0.002,
            "trimmer_domain_amp": 0.0,
            "momentum_reset": None,
        },
    },
}

if __name__ == "__main__":
    run_series(RUNS, OUT, __doc__.split("\n\n")[0])
