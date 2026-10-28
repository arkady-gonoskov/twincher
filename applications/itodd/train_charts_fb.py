"""The charts of the connector (obj 26) with the Fourier-Bessel recipe: a resumable series.

Recipe (from ChartY0, y0_fb.py and y0_full.py):
- stencil: 64 Fourier-Bessel components of the heights about the centre of mass of the
  mask (shell order, damping exp(-(n_r / 3)^2), ISRU with C 12), the background brought
  to 20 mm behind the mean depth of the mask, a first component for the distance
  (ISRU of 6 (Z_mean - Z_mid) / (Z_bg - Z_lo)); finite-difference step 0.025;
- data: roll-aware (roll_fb.RollFB): grid 8 x 32 x 96 over (distance, elevation,
  azimuth) at the roll centre, the roll applied exactly (2 min to compute per chart);
- twincher: n_s 128, n_l 384; Adam lr 1e-3, n_c 5120;
- loss: the hs loss with ramp_boost 1.0 (no boost at the border), fine curriculum
  (start at R 0.05, steps 0.0025), the trimmer also over the whole domain (without it
  the training collapsed when the box grew into regions of large |dr/dp|), no momentum
  reset;
- rollback: after a stride with more than 0.1 % det < 0 back to the previous stride,
  lr halved, generator reseeded (at most 8 times);
- R held at 0.86 (the six charts cover all views from |p|_inf <= 0.857 on); exact checks
  every 2500 iterations (det J at 8192 points, the solver on 256 views, windings);
  stop after 2 consecutive checks at R 0.86 without det J <= 0 and with all views
  solved (the batch criterion, 6 strides with all det J > 0.5 over 2.5 million cases
  each, is rarely met), or after 60000 iterations; best.twc by the fewest views not
  solved.

Outputs per chart in output_charts_fb/<chart>/ (chart_training.py): best.twc, final.twc,
history.csv, history_exact.csv, result.json, traces of explosions; summary.txt.

The script can be interrupted at any time and started again (completed charts are
skipped, an interrupted chart continues from its last checkpoint). In a loop that
restarts it after a crash (e.g. CUDA error 999 of the WSL GPU driver, which kills the
process; the GPU works again right after), it resumes by itself; the loop ends when the
script exits normally.

Usage:
    mkdir -p output_charts_fb
    nohup bash -c 'until python -u train_charts_fb.py --runs Z1 Z2 Y1 Y2 \
        >> output_charts_fb/series.log 2>&1; do sleep 60; done' > /dev/null 2>&1 &
    python train_charts_fb.py --summary
"""

from pathlib import Path

from chart_training import run_series

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_charts_fb"
RECIPE = {
    "obj": 26,
    "chart": "Z0",
    "margin": 1.0,
    "dims": None,
    "scale": 1.0,
    "data": "rollfb",
    "grid": [8, 32, 96],
    "data_dir": str(OUT / "data"),
    "kernel": "flat",
    "probes": 8,
    "spectral": None,
    "fb": {"n": 64, "scale": 12.0, "background": 20.0, "depth": 6.0},
    "dp": 0.025,
    "n_s": 128,
    "n_l": 384,
    "n_c": 5120,
    "lr": 1e-3,
    "seed": 42,
    "loss_params": {
        "ramp_boost": 1.0,
        "ramp_size": 0.05,
        "p_range_step": 0.0025,
        "trimmer_domain_amp": 1.0,
        "momentum_reset": None,
    },
    "extra_terms": {},
    "rollback": {"lr_factor": 0.5, "max": 8},
    "max_iters": 60000,
    "stride": 500,
    "exact_stride": 2500,
    "r_stop": 0.86,
    "done_strides": 6,
    "done_exact": 2,
    "check_box": 0.86,
    "solver_check": 256,
    "solver_iters": 40,
    "solver_clamp": 0.86,
    "eval_boxes": [0.5, 0.86],
}
RUNS = {
    chart: {**RECIPE, "chart": chart} for chart in ("Z0", "Z1", "Z2", "Y0", "Y1", "Y2")
}

if __name__ == "__main__":
    run_series(RUNS, OUT, __doc__.split("\n\n")[0])
