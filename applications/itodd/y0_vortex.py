"""ChartY0 of the connector: vortices (branch points), a decreasing ramp, a
component-wise curriculum and the s-gate threshold.

Base: the 2D setting of y0_reduced.py (elevation, azimuth; grid 32 x 96; flat 8 x 8
stencil; n_s 128; lr 1e-3; n_c 5120; R held at 0.86; 40000 iterations). The ramp (hs
loss with ramp_boost 0, without the options added later: RAMP): the amplitude of the det
loss decreases linearly from amp at |p|_inf = R - ramp_size to 0 at R, so that the
penalty around a vortex near the border is anisotropic and pushes the vortex out of the
box.

Runs:
- 2D, n_l 64 (y0_reduced c40 kept vortices at the centre and at phi +-0.6):
  - restart_r02, restart_r05: from c40/final.twc (R reset), ramp, ramp_size 0.2 (start
    at R 0.2, step 0.01) or 0.05 (step 0.0025);
  - restart_noramp: the same with the hs loss of twincher 0.1.0 (fine curriculum),
    control;
  - scratch_r05: the ramp from scratch (ramp_size 0.05, step 0.0025).
- 3D component-wise curriculum: from the 2D twincher c40_l256 (n_l
  256, clean), elevation and azimuth drawn over [-0.86, 0.86], the curriculum only on
  the roll (start 0.05, step 0.0025), grid (32, 96, 24):
  - cw3_ramp: with the ramp; cw3_pkg: hs loss of 0.1.0; cw3_ramp_sgt2: ramp and s-gate
    threshold 2.0 (instead of 1.3);
  - cw3_cube_ramp: from c40_l256 too, but the curriculum box grows in all three
    components from 0.05 (cw3_ramp stayed at roll R 0.05: the orientation of the new
    direction, arbitrary in the 2D twincher, has to be fixed over the whole (e, phi)
    range at once, folds across that plane cannot leave the box).

The script can be interrupted at any time and started again (completed runs are
skipped, an interrupted run continues from its last checkpoint).

Usage:
    nohup python -u y0_vortex.py --runs restart_r02 ... > output_y0_vortex/a.log 2>&1 &
    python y0_vortex.py --summary
"""

from pathlib import Path

from chart_training import HS_0_1_0, run_series
from y0_reduced import C40

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_y0_vortex"
RAMP = {"ramp_boost": 0.0, "trimmer_domain_amp": 0.0, "momentum_reset": None}
FINE = {"ramp_size": 0.05, "p_range_step": 0.0025}
BASE2 = {**C40, "data_dir": str(HERE / "output_y0_reduced/data"), "exact_stride": 2500}
C40_FINAL = str(HERE / "output_y0_reduced/c40/final.twc")
RUNS = {
    "restart_r02": {**BASE2, "init": C40_FINAL,
                    "loss_params": {**RAMP, "ramp_size": 0.2, "p_range_step": 0.01}},
    "restart_r05": {**BASE2, "init": C40_FINAL, "loss_params": {**RAMP, **FINE}},
    "restart_noramp": {**BASE2, "init": C40_FINAL, "loss_params": {**HS_0_1_0, **FINE}},
    "scratch_r05": {**BASE2, "loss_params": {**RAMP, **FINE}},
}  # fmt: skip
BASE3 = {**BASE2, "dims": [1, 2, 3], "grid": [32, 96, 24],
         "data_dir": str(HERE / "output_y0_expand/data"), "n_l": 256,
         "init": str(HERE / "output_y0_reduced/c40_l256/best.twc"),
         "fixed": {"components": [0, 1], "range": 0.86}}  # fmt: skip
RUNS["cw3_ramp"] = {**BASE3, "loss_params": {**RAMP, **FINE}}
RUNS["cw3_pkg"] = {**BASE3, "loss_params": {**HS_0_1_0, **FINE}}
RUNS["cw3_ramp_sgt2"] = {**RUNS["cw3_ramp"], "sgt": 2.0}
RUNS["cw3_cube_ramp"] = {**RUNS["cw3_ramp"], "fixed": None}

if __name__ == "__main__":
    run_series(RUNS, OUT, __doc__.split("\n\n")[0])
