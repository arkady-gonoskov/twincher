# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

import numpy as np
from .tw_util import estimate_J, copy_to, plot_style, save_figure
from .solver import solver
from pathlib import Path

def noise_check( 
        file_name, 
        stencil, 
        device="auto",
        output_file_name = "noise_tol.png", 
        rng_seed = 42,
        noise_max = 1.0,
        ax_ylim = "auto",
        n_c = 1024,
    ):
    dir = Path(file_name).resolve().parent
    CS = solver(
        file_name = file_name,
        n_c = n_c,
        device = device,
        epsilon = 1e-8,
    )

    n_p, n_y, n_s = stencil.n_p, stencil.n_y, CS.SH.n_s
    rng = np.random.default_rng(rng_seed)
    noise_amps = np.empty((n_c), dtype=np.float64)
    noise_amps[...] = rng.uniform(0, noise_max, size=noise_amps.shape)
    p_true = np.empty((n_c, n_p), dtype=np.float64)
    p = np.empty((n_c, n_p), dtype=np.float64)
    p_true[...] = rng.uniform(-1, 1, size=p_true.shape)
    y_true = np.empty((n_c, n_y), dtype=np.float64)
    y = np.empty((n_c, n_y), dtype=np.float64)
    dy_dp = np.empty((n_c, n_y, n_p), dtype=np.float64)
    for i_c in range(n_c):
        estimate_J(
            stencil=stencil, 
            p=p_true[i_c, :], 
            y=y_true[i_c, :],
            dy_dp=dy_dp[i_c, :, :],
            d_eval=1e-8,
        )
    pinv_dy_dp = np.linalg.pinv(dy_dp)
    
    noise = np.empty((n_c, n_y), dtype=np.float64)
    noise[...] = rng.uniform(-1, 1, size=noise.shape)
    noise[:, :] = noise[:, :] - np.matmul(pinv_dy_dp[:, :, :].transpose(0, 2, 1), np.matmul(dy_dp[:, :, :].transpose(0, 2, 1), noise[:, :, None]))[:, :, 0]
    noise[:, :] /= np.linalg.norm(noise, axis=1, keepdims=True).clip(1e-8)

    # for i_p in range(n_p):
    #     for i_c in range(n_c):
    #         print("dot=", np.dot(noise[i_c, :], dy_dp[i_c, :, i_p]))

    noise[:, :] *= noise_amps[:, None]

    y_true[:, :] += noise[:, :]
    y_true.clip(min=-1.0, max=1.0, out=y_true)

    CS.start_solution(y_true=y_true)
    for _ in range(32):
        copy_to(p, CS.p)
        for i_c in range(n_c):
            estimate_J(
                stencil=stencil,
                p=p[i_c, :],
                y=y[i_c, :],
                dy_dp=dy_dp[i_c, :, :],
                d_eval=1e-8
            )
        CS.step(y=y, dy_dp=dy_dp)

    diffs = np.linalg.norm((y - y_true), axis=1)

    if output_file_name is not None:
        import matplotlib.pyplot as plt
        style = "dark_background" # applied only to this figure, not to the settings of the user
        with plot_style(style):
            fig, ax = plt.subplots()
            ax.set_xlim(0, noise_max)
            if ax_ylim == "auto":
                ax.set_ylim(bottom=0, top=max(max(diffs), noise_max*2))
            else:
                ax.set_ylim(bottom=0, top=ax_ylim)
            ax.scatter(
                x=noise_amps,
                y=diffs,
                color="cyan",   # marker color
                s=10,          # marker size
                alpha=0.5,       # transparency (0 = transparent, 1 = opaque)
                edgecolor="none",
            )
            ax.set_xlabel(r"Input noise: $\|y(p_{\text{true}}) - y_{\text{provided}}\|$")
            ax.set_ylabel(r"Residual: $\|y(p_{\text{solution}}) - y_{\text{provided}}\|$")
            ax.set_title("Noise tolerance")
        save_figure(fig, str(dir) + f"/{output_file_name}", style=style)
        plt.close(fig) # figures of pyplot stay in memory until they are closed

    return noise_amps, diffs
