# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

from pathlib import Path
import tempfile
import numpy as np
import torch
from .shuttle import Shuttle
from .tw_util import select_device, output_dir, plot_style, save_figure

class HSMonitorP2:
    def __init__(self, 
        SH: Shuttle, 
        stencil,
        n_g: int = 32,
        dir = None, # directory for images (str or Path; default: ./twincher_output)
        device = "auto",
        color_scheme = "dark",
    ):
        device = select_device(device)
        if device == "cuda":
            data_type = "torch"
            dtype = torch.float32
        else:
            data_type = "torch_cpu"
            dtype = torch.float64

        self.SH = SH
        if SH.n_p != 2 or stencil.n_p != 2:
            raise RuntimeError("HSMonitorP2 is designed for n_p = 2")
        self.stencil = stencil
        self.n_g = n_g
        self.dir = output_dir(dir)
        self.dtype = dtype
        n_y = self.stencil.n_y
        # create identical shuttle for computing r[0] for grid points
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = str(Path(tmp_dir) / "shuttle.twc")
            SH.save(file_name)
            self.SH_grid = Shuttle(data_type=data_type)
            self.SH_grid.load(file_name, n_c=n_g**2, n_p=1, n_m=1)

        self.d_y = torch.empty((n_y, n_g**2), dtype=dtype, device=device)
        self.h_y = np.zeros((n_y, n_g**2), dtype=np.float32)
        self.h_p = np.zeros((2, n_g**2), dtype=np.float32)
        self.h_r_out = np.zeros((2, n_g**2), dtype=np.float32)

        x = np.linspace(-1, 1, n_g)
        y = np.linspace(-1, 1, n_g)
        self.node_x, self.node_y = np.meshgrid(x, y)

        for i_x in range(n_g):
            for i_y in range(n_g):
                self.h_p[0, i_y*n_g + i_x] = -1 + 2*i_x/(n_g - 1)
                self.h_p[1, i_y*n_g + i_x] = -1 + 2*i_y/(n_g - 1)
                stencil(self.h_p[:, i_y*n_g + i_x], self.h_y[:, i_y*n_g + i_x])
        self.d_y.copy_(torch.as_tensor(self.h_y, device=self.d_y.device))

        # for plot:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        if color_scheme == "dark":
            self.style = "dark_background"
            color = "white"
        else:
            self.style = None # current matplotlib settings
            color = "black"
        # the style is applied only while the figure is created and saved, so that the
        # matplotlib settings of the user are not changed
        with plot_style(self.style):
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-1.3, 1.3)
            self.ax.set_ylim(-1.3, 1.3)
            self.ax.set_xlabel(r'$r_0$')
            self.ax.set_ylabel(r'$r_1$')
            self.ax.set_aspect("equal")
            # elements such as ticks are created on demand: create them in this style
            self.fig.draw_without_rendering()

        self.collection = LineCollection(
            [],
            color=color,
            linewidths=0.5,
            alpha=0.3,
        )
        self.ax.add_collection(self.collection)

    def draw(self, SH: Shuttle, file_name = "test.png"):
        self.SH_grid.a.copy_(SH.a)
        self.SH_grid.tw_update()
        n_y = self.stencil.n_y
        
        self.SH_grid.s.fill_(0)
        self.SH_grid.s[:n_y, :].copy_(self.d_y[:, :])
        self.SH_grid.forward_inference()
        self.h_r_out[:, :] = self.SH_grid.s[:2, :].detach().cpu().numpy()
        for i_x in range(self.n_g):
            for i_y in range(self.n_g):
                self.node_x[i_x, i_y] = self.h_r_out[0, i_y*self.n_g + i_x]
                self.node_y[i_x, i_y] = self.h_r_out[1, i_y*self.n_g + i_x]
        segments = []
        # Horizontal grid lines.
        for j in range(self.n_g):
            segments.extend(
                [
                    [(self.node_x[j, i], self.node_y[j, i]),
                        (self.node_x[j, i + 1], self.node_y[j, i + 1])]
                    for i in range(self.n_g - 1)
                ]
            )

        # Vertical grid lines.
        for i in range(self.n_g):
            segments.extend(
                [
                    [(self.node_x[j, i], self.node_y[j, i]),
                        (self.node_x[j + 1, i], self.node_y[j + 1, i])]
                    for j in range(self.n_g - 1)
                ]
            )
        self.collection.set_segments(segments)

        save_figure(self.fig, str(self.dir) + "/" + file_name, style=self.style)

