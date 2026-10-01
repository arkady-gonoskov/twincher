# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

import torch
import numpy as np
import matplotlib.pyplot as plt
import twincher
#plt.style.use("dark_background")

def plot_example(stencil, file_name):
    fig, ax = plt.subplots(dpi=300)
    n_y = 32
    p0, p1 = -0.6, 0.3
    p = np.array([p0, p1]).astype(np.float64)
    y = np.empty((n_y,), dtype=np.float64)
    x = np.linspace(-1, 1, n_y)
    stencil(p, y)
    ax.plot(x, y)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y_i$')
    ax.set_xlim(-1, 1)
    ax_top = ax.twiny()
    ax_top.set_xlim(0, 31)
    ax_top.set_xticks(8*np.arange(4))
    ax_top.set_xlabel("$i$ (index of $y$ components)")

    # Vertical lines
    ax.axvline(p0, color="gray", linestyle="--", linewidth=1.5)
    ax.axvline(p1, color="gray", linestyle="--", linewidth=1.5)
    ax.text(
        p0+0.05, 0.1,
        r"$x=p_0$",
        color="gray",
        ha="left",
        va="center"
    )

    ax.text(
        p1+0.05, 0.4,
        r"$x=p_1$",
        color="gray",
        ha="left",
        va="center"
    )

    fig.savefig(file_name)