"""Widths of the twinches of trained twinchers, against their initialization (README Section 16).

The parameters a of every twinch are mapped to its width w and twist t as twinch_update
in include/twinch.h of the package does, with the width range (wl, ws, wu) stored in the
file. A cold start (set_cold in twincher.h, also used by twc_extend.py for new layers)
draws a2 uniformly from [-0.333, 0.5] and sets t = 0. Twinches whose twist stays small
(|t| <= 0.05) barely act on the map, so the quantiles of w are given separately for the
twinches with a noticeable twist, for the others, and for the initial distribution
(sampled with the width range of the first file). The last column is the share of the
twisted twinches that are broader than any twinch at initialization (a2 = 0.5).

Usage:
    python twinch_widths.py output_charts_fb/Z0/best.twc [more .twc files]
"""

import argparse

import numpy as np

from twc_extend import read_twc

T_MIN = 0.05
QUANTILES = (0.05, 0.5, 0.95)


def width(a2, wl, ws, wu):
    root = np.sqrt(a2**2 * (wu - wl) ** 2 + 4 * (wu - ws) * (ws - wl))
    return (2 * ws + a2**2 * (wu + wl) + a2 * root) / (2 * (1 + a2**2))


def quantiles(w):
    return " / ".join(f"{v:.3f}" for v in np.quantile(w, QUANTILES))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    q = " / ".join(f"{100 * v:g} %" for v in QUANTILES)
    print(f"{'file':42s} {'n_l':>4s} {'|t| > ' + str(T_MIN):>10s}   "
          f"w ({q}): |t| > {T_MIN}, others; broader than at initialization")  # fmt: skip
    for k, file in enumerate(args.files):
        core = read_twc(file)[1]
        a, (wl, ws, wu) = core["a"].astype(float), core["twv"].astype(float)
        w, t = width(a[2::4], wl, ws, wu), 0.9 * a[3::4] / np.sqrt(1 + a[3::4] ** 2)
        active, w_max = np.abs(t) > T_MIN, width(0.5, wl, ws, wu)
        if k == 0:
            a2 = np.random.default_rng(0).uniform(-0.333, 0.5, 10**6)
            cold = width(a2, wl, ws, wu)
            print(f"{'initialization':42s} {'':4s} {0:10.0%}   {quantiles(cold)}")
        print(f"{file:42s} {core['n_l']:4d} {active.mean():10.0%}   "
              f"{quantiles(w[active])}, {quantiles(w[~active])}; "
              f"{np.mean(w[active] > w_max):.0%}")  # fmt: skip


if __name__ == "__main__":
    main()
