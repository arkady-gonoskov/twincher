"""ChartY0 of the connector, all four components: continuing the 4D run to R 0.86.

y0_fb.py fbtrim_full (FB stencil with the near background and the depth component, grid
(8, 32, 96, 24), n_l 256, trimmer over the whole domain, fine curriculum) grew to R
0.765 by iteration 25000 (best checkpoint: solver failures 0.8 %, all windings 1), then
exploded twice (at the step to R 0.782 and at iteration 49000 without a change of R;
445k-472k det < 0 per stride) and ended broken (centre winding 0) at R 0.782.

Every run starts from that best checkpoint (output_y0_fb/best25000.twc; R from 0.765),
with the hs loss with ramp_boost 1.0 (no boost at the border; fbtrim_full had the ramp
of twincher 0.1.0 with 2.0), fine steps (0.0025), the trimmer over the whole domain, R
held at 0.86, 40000 iterations; a fresh Adam wrecked the restored twincher within 50
iterations even at lr 1e-5, so 1000 iterations update only Adam's moments (parameters
kept), then the learning rate is warmed up over 2000 iterations; per stride the history
also records the largest |dr/dp_k| over the whole domain (trim_max), the largest
gradient norm and the smallest damped learning rate.

Runs:
- cont: n_l 256 as before (does it just need more iterations?);
- l384: n_l 384, a zero-twist layer inserted after every second layer (twc_extend.py;
  the map is unchanged to 4e-14), more capacity;
- sgt2: s-gate threshold 2.0 instead of 1.3 (with det J ~ 1 the corners of the
  4D box lie sqrt(4) R from the centre, beyond 1.3);
- trim3: trimmer amplitude 3 (inside, hs loss, and over the whole domain);
- lr5: lr 0.5e-3;
- l384_rb: l384 with rollback (chart_training: back to the previous stride, lr halved,
  generator reseeded, after a stride with more than 0.1 % det < 0; at most 8 times).
- l384_clip: l384 with gradient clipping at 5 times the running mean of |g| (l384
  reached R 0.86 at 6500, clean at 7500, then collapsed at 10000 without a change of R:
  |g| 8787, |dr/dp| 73).
- l384_trim3: l384 with the trimmer amplitude 3 (inside and over the whole domain):
  l384_clip had an event at R 0.80 (22868 det < 0) preceded by a slow rise of the
  largest |dr/dp| over ~300 iterations (median per batch 1.7 -> 2.9, then 7.4; inside R
  1.6 -> 2.6 -> 89), so clipping (stopped at 3500) does not address it.
sgt2 exploded at R 0.787 (iteration 3000, 995k det < 0) and was stopped at 5000.

The script can be interrupted at any time and started again (completed runs are
skipped, an interrupted run continues from its last checkpoint).

Usage:
    nohup python -u y0_full.py --runs l384 sgt2 > output_y0_full/a.log 2>&1 &
    python y0_full.py --summary
"""

from pathlib import Path

from chart_training import run_series
from y0_fb import RUNS as FB_RUNS

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_y0_full"
BEST = HERE / "output_y0_fb"
LOSS = {"ramp_boost": 1.0, "ramp_size": 0.05, "p_range_step": 0.0025,
        "p_range_init": 0.765, "trimmer_domain_amp": 1.0, "momentum_reset": None}  # fmt: skip
BASE = {**FB_RUNS["fbtrim_full"], "init": str(BEST / "best25000.twc"),
        "loss_params": LOSS, "max_iters": 40000, "adam_prime": 1000,
        "warmup": 2000}  # fmt: skip
RUNS = {
    "cont": BASE,
    "l384": {**BASE, "n_l": 384, "init": str(BEST / "best25000_l384.twc")},
    "sgt2": {**BASE, "sgt": 2.0},
    "trim3": {**BASE, "loss_params": {**LOSS, "trimmer_amp": 3.0,
                                      "trimmer_domain_amp": 3.0}},
    "lr5": {**BASE, "lr": 0.5e-3},
}  # fmt: skip
RUNS["l384_rb"] = {**RUNS["l384"], "rollback": {"lr_factor": 0.5, "max": 8}}
RUNS["l384_clip"] = {**RUNS["l384"], "clip": 5.0}
RUNS["l384_trim3"] = {**RUNS["l384"], "loss_params": {**LOSS, "trimmer_amp": 3.0,
                                                      "trimmer_domain_amp": 3.0}}  # fmt: skip

if __name__ == "__main__":
    run_series(RUNS, OUT, __doc__.split("\n\n")[0])
