# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

import tempfile
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import twincher

#COLOR_SCHEME = "dark"
COLOR_SCHEME = "light"
if COLOR_SCHEME == "dark":
    plt.style.use("dark_background")

class StateMonitor: # plots r_0 as a function of s_0, s_1 

    def __init__(self, stencil, dir, n_c = 128, n_g = 128):
        if stencil.n_p != 1 or stencil.n_y != 2:
            raise ValueError("This tool is designed for n_p=1, n_y=2.")
        self.stencil = stencil
        self.dir = dir
        self.fig, self.ax = plt.subplots(dpi=100)
        self.ax.set_xlim(-1.0, 1.0)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_aspect("equal", adjustable="box")
        self.n_g = n_g
        n_c = 128 # number of cases for the plot
        p = np.linspace(-1, 1, n_c)
        p = p[:, None]
        y = np.zeros((n_c, 2), dtype=np.float32)
        for i_c in range(n_c):
            stencil(p[i_c, :], y[i_c, :])        
        self.curve = self.ax.plot(y[:, 0], y[:, 1])

    def make_plot(self, file_name, iter, shuttle = None):
        n_g = self.n_g
        dir = self.dir
        if shuttle is not None:
            if not hasattr(self, "shuttle"):
                # copy of the twincher (on the same device) for evaluating r_0 on a grid of y
                with tempfile.TemporaryDirectory() as tmp_dir:
                    twc_file = str(Path(tmp_dir) / "shuttle.twc")
                    shuttle.save(twc_file)
                    self.shuttle = twincher.Shuttle(data_type=shuttle.dtype)
                    self.shuttle.load(twc_file, n_c=n_g**2, n_p=1, n_m=1)
                x = torch.linspace(-1, 1, n_g)
                y = torch.linspace(-1, 1, n_g)
                X, Y = torch.meshgrid(x, y, indexing='xy')
                self.y = torch.stack((X.flatten(), Y.flatten()), dim=1)
                self.contours = None

            self.shuttle.a.copy_(shuttle.a)
            self.shuttle.tw_update()
            
            self.shuttle.s[:2, :].permute(1, 0).copy_(self.y)
            self.shuttle.s[2:, :].zero_()
            self.shuttle.forward_inference()
            r0 = self.shuttle.s[0, :].reshape(n_g, n_g).detach().cpu().numpy()
            if not hasattr(self, "image"):
                from matplotlib.colors import Normalize
                self.image = self.ax.imshow(r0,
                    origin="lower",
                    extent=(-1, 1, -1, 1),
                    aspect="equal",
                    cmap=plt.colormaps["magma"],
                    norm=Normalize(vmin=-1, vmax=1),
                    interpolation="nearest",
                )
                bar = self.fig.colorbar(self.image, ax=self.ax, label="$r_0$")
                colors = "white" if COLOR_SCHEME == "dark" else "black"
                bar.ax.tick_params(colors=colors)
                bar.ax.yaxis.label.set_color(colors)
                bar.outline.set_edgecolor(colors)
                bar.outline.set_linewidth(1.0)
            else:
                self.image.set_data(r0)
            if self.contours is not None:
                self.contours.remove()
            self.contours = self.ax.contour(r0,
                levels=np.arange(-1.0, 1.0, 2/16.0),
                extent=(-1, 1, -1, 1),
                origin="lower",
                colors="white",
                linewidths=0.5,
                linestyles="solid",
                alpha=0.3,
            )
            self.ax.set_title(f'State evolution: "{self.stencil.name}" (iter={iter})')

        self.ax.set_xlabel(r'$y_0$')
        self.ax.set_ylabel(r'$y_1$')
        self.fig.savefig(file_name)

def make_parity_plot(dir, stencil, file_name):
    n_c = 1024
    solver = twincher.solver(
        file_name=str(dir / "final.twc"), 
        n_c = n_c,
    )
    rng = np.random.default_rng(42)
    p_true = rng.uniform(-1, 1, (n_c, 1))
    y_true = np.empty((n_c, 2), dtype=np.float64)
    for i_c in range(n_c):
        stencil(p=p_true[i_c, :], y=y_true[i_c, :])
    p_solution = np.empty((n_c, 1), dtype=np.float64)
    solver.solve(stencil, y_true, n_iter=8, out=p_solution)

    fig, ax = plt.subplots(dpi=300)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel(r'$p_{true}$')
    ax.set_ylabel(r'$p_{sol}$')
    ax.set_aspect("equal", adjustable="box")
    ax.scatter(x=p_true[:, 0], y=p_solution[:, 0], s=5, alpha=0.5, label="twincher", edgecolors='none')

    # y-space descent 
    p = np.random.uniform(-1, 1, size=(n_c, 1)).astype(np.float64)
    y = np.empty((n_c, 2), dtype=np.float64)
    dy_dp = np.empty((n_c, 2, 1), dtype=np.float64)
    p_step = 0.2
    for _ in range(16):
        for i_c in range(n_c):
            twincher.estimate_J(
                stencil=stencil,
                p=p[i_c, :],
                y=y[i_c, :],
                dy_dp=dy_dp[i_c, :, :],
                d_eval=1e-8
            )
        norms = (dy_dp[:, :, 0]*dy_dp[:, :, 0]).sum(axis=1)
        norms = np.maximum(norms, 1e-8)
        delta_p = (dy_dp[:, :, 0]*(y_true - y)).sum(axis=1)
        p[:, 0] += delta_p*np.minimum(p_step, norms**-1)
    ax.scatter(x=p_true[:, 0], y=p[:, 0], s=20, alpha=0.5, zorder=0, label="y-space descent", edgecolors='none')
    ax.legend()
    ax.set_title("Parity plot (well-posed case of spiral curve)")
    fig.savefig(dir / file_name)