"""ChartY0 of the connector with the Fourier-Bessel stencil.

y: the 64 lowest Fourier-Bessel components (shell index si = m + 2 (n_r - 1)) of the
heights, centred at the centre of mass of the mask, damped by exp(-(n_r / 3)^2), mapped
by ISRU(C x) (depth_stencil.py, fb options). Before training (stencil_metrics.py) the
variant with the background brought to 20 mm behind the mean depth of the mask had the
mildest near-cusps of y(phi) (chord/arc min 0.45 vs 0.26 for flat 8 x 8) and the
smallest interpolation errors; the variant with heights above Z_bg (as flat) behaved like
flat 8 x 8.

Runs in the settings where flat 8 x 8 failed (grid data, fine curriculum ramp_size 0.05,
p_range_step 0.0025, the hs loss of twincher 0.1.0, n_s 128, lr 1e-3, n_c 5120, R held
at 0.86):
- fb_bg20_l64, fb_c2_l64: 2D (elevation, azimuth), grid 32 x 96, n_l 64 (flat: c40,
  vortices at the bore views, R 0.735 after 40000 iterations);
- fb_bg20_d123: 3D (elevation, azimuth, roll), grid (32, 96, 24), n_l 256 (flat: d123,
  folded at R 0.515 and stalled).
All of them (and flat) collapsed suddenly at some R (fb_c2_l64 0.60, fb_bg20_l64 0.70,
fb_bg20_d123 0.667: loss from 1e-3 to 1-2): beyond R the derivatives |dr/dp_k| reach
45-70 (inside <= 1.7), so the cases taken in by the growing box carry huge trimmer and
det losses. With the trimmer applied over the whole domain (option trimmer_domain_amp
of the hs loss, TRIM):
- fbtrim_bg20_l64, fbtrim_bg20_d123: as fb_bg20_l64, fb_bg20_d123;
- flat_trim_l64: flat 8 x 8 (y0_reduced c40) with it;
- fbtrim_full: the full chart (n_p 4), FB with the near background and the depth
  component (C_D 6: with the near background the distance is otherwise only in the
  scale), grid (8, 32, 96, 24), n_l 256, trimmer over the whole domain, 60000
  iterations.

The script can be interrupted at any time and started again (completed runs are
skipped, an interrupted run continues from its last checkpoint).

Usage:
    nohup python -u y0_fb.py > output_y0_fb/series.log 2>&1 &
    python y0_fb.py --summary
"""

from pathlib import Path

from chart_training import run_series
from y0_reduced import C40

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_y0_fb"
FB_BG20 = {"n": 64, "scale": 12.0, "background": 20.0}
FB_C2 = {"n": 64, "scale": 2.0}
BASE2 = {**C40, "data_dir": str(OUT / "data"), "exact_stride": 2500, "fb": FB_BG20}
RUNS = {
    "fb_bg20_l64": BASE2,
    "fb_c2_l64": {**BASE2, "fb": FB_C2},
    "fb_bg20_d123": {**BASE2, "dims": [1, 2, 3], "grid": [32, 96, 24], "n_l": 256},
}
TRIM = {**C40["loss_params"], "trimmer_domain_amp": 1.0}
RUNS["fbtrim_bg20_l64"] = {**RUNS["fb_bg20_l64"], "loss_params": TRIM}
RUNS["fbtrim_bg20_d123"] = {**RUNS["fb_bg20_d123"], "loss_params": TRIM}
RUNS["flat_trim_l64"] = {**C40, "data_dir": str(HERE / "output_y0_reduced/data"),
                         "exact_stride": 2500, "loss_params": TRIM}  # fmt: skip
RUNS["fbtrim_full"] = {**RUNS["fbtrim_bg20_d123"], "dims": None, "grid": [8, 32, 96, 24],
                       "fb": {**FB_BG20, "depth": 6.0}, "max_iters": 60000}  # fmt: skip

if __name__ == "__main__":
    run_series(RUNS, OUT, __doc__.split("\n\n")[0])
