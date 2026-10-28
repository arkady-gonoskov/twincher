"""Post-training of the chart twinchers for noise tolerance (hs_sm_noise).

Starts from the twincher of a chart trained with train_charts_fb.py (best.twc) and
continues the training with its recipe (same stencil, roll-aware data, hs loss with the
trimmer over the whole domain), R held at 0.86 from the start, plus the single-mode noise
loss of the package (hs_sm_noise): for each case, the direction of y perpendicular to the
manifold y(p) in which r changes most is penalized, also at random points up to y_range
off the manifold. The determinant and trimmer terms act as guard rails meanwhile.

As every restart of a trained twincher: 1000 iterations that only update Adam's moments
(adam_prime), then the learning rate warmed up over 2000 iterations (a fresh Adam
otherwise wrecks the twincher, README section 13.3). No rollback: the negative
determinants of the first phase (seen with the cylinder) are to be observed, not undone.
Snapshots iter_<n>.twc every 1000 iterations for noise_check_chart.py.

Runs (name: chart, settings):
- z0: ChartZ0, the standard parameters of milestone 1 (amp 1e-3, y_range 0.05), lr 1e-3,
  20000 iterations: clean to 10000 (|dr/dy|_normal p75 5.7 -> 1.83), exploded at 10500
  (363668 det < 0) and recovered only slowly (4 % of the views not solved at 14000);
- z0_reset: z0 continued from its state before the explosion (pre_explosion_10500.pt,
  with Adam) with the first version of the momentum reset (README Section 18,
  chart_training.FirstReset: when more than 0.1 % of a batch has det < 0, the last
  update reverted, Adam's first moment zeroed, lr x 0.5 and then x 1.1 per iteration),
  up to 30000 iterations: clean to 14000 (p75 1.56), then a state with ~6 det < 0 per
  batch triggered a reset in every iteration, which alternated between two states while
  the learning rate fell to 0 (stopped at 18000; snapshots up to 14000 usable);
- z0_reset2: as z0_reset with the revised reset (option momentum_reset of the hs loss,
  twincher.hs_loss.MomentumReset: relative trigger, revert to the last accepted state,
  learning-rate floor 0.05): clean to 30000, the spikes at 10103 and 14500 caught by 1
  and 4 resets;
- z1, z2, y0, y1, y2: the other charts with the procedure of z0 and the revised reset
  from the start, 20000 iterations (y0: the 384-layer twincher of y0_full.py, with the
  roll-aware data of the recipe).
Runs with "start_from" begin with that checkpoint (copied into the run directory under
the default output directory, without its best state).

Usage:
    nohup bash -c 'until python -u noise_charts.py --runs z0 \
        >> output_noise_charts/series.log 2>&1; do sleep 60; done' > /dev/null 2>&1 &
    python noise_charts.py --summary
"""

import sys
from pathlib import Path

import torch

from chart_training import run_series
from train_charts_fb import OUT as CHARTS_OUT
from train_charts_fb import RUNS as CHARTS

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_noise_charts"


def post_training(chart, noise, reset=None, **settings):
    base = CHARTS[chart]
    return {
        **base,
        "init": str(CHARTS_OUT / chart / "best.twc"),
        "loss_params": {
            **base["loss_params"],
            "p_range_init": base["r_stop"],
            "momentum_reset": reset,
        },
        "extra_terms": {**base["extra_terms"], "hs_sm_noise": noise},
        "rollback": None,
        "adam_prime": 1000,
        "warmup": 2000,
        "max_iters": 20000,
        "done_strides": 10**9,  # no early stop: a fixed length
        "done_exact": None,
        "exact_stride": 2000,
        "snapshot": 1000,
        **settings,
    }


RESET = {"neg_frac": 1e-3, "lr_factor": 0.5, "lr_up": 1.1, "ratio": 3.0, "beta": 0.99,
         "lr_floor": 0.05, "depth": 4}  # fmt: skip
RUNS = {
    "z0": post_training("Z0", {"amp": 1e-3, "y_range": 0.05}),
    "z0_reset": post_training(
        "Z0", {"amp": 1e-3, "y_range": 0.05}, max_iters=30000,
        reset={"neg_frac": 1e-3, "lr_factor": 0.5, "lr_up": 1.1}, first_reset=True,
        start_from="output_noise_charts/z0/pre_explosion_10500.pt",
    ),
    "z0_reset2": post_training(
        "Z0", {"amp": 1e-3, "y_range": 0.05}, max_iters=30000,
        reset=RESET,
        start_from="output_noise_charts/z0/pre_explosion_10500.pt",
    ),
    **{chart.lower(): post_training(
        chart, {"amp": 1e-3, "y_range": 0.05}, reset=RESET,
        **({"init": str(CHARTS_OUT / "Y0_l384_best7500.twc")} if chart == "Y0" else {}),
    ) for chart in ("Z1", "Z2", "Y0", "Y1", "Y2")},
}  # fmt: skip


def prepare(name, cfg):
    """Copy the checkpoint of start_from into the run directory (once)."""
    run_dir = OUT / name
    if (
        not cfg.get("start_from")
        or (run_dir / "checkpoint.pt").exists()
        or (run_dir / "train.json").exists()
    ):
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(HERE / cfg["start_from"], map_location="cpu")
    state["best"] = None  # the run keeps its own best state
    torch.save(state, run_dir / "checkpoint.pt")


if __name__ == "__main__":
    if "--summary" not in sys.argv:
        for name, cfg in RUNS.items():
            if "--runs" not in sys.argv or name in sys.argv:
                prepare(name, cfg)
    run_series(RUNS, OUT, __doc__.split("\n\n")[0])
