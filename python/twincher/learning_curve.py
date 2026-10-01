# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

import os
import numpy as np
import matplotlib.pyplot as plt
from .tw_util import plot_style, save_figure

class LearningCurve:
    """
    Figure with the learning curve and the diagnostics of the loss terms, saved as lc.png.

    Loss terms can draw diagnostics into the axes given to them as attribute ax: they plot
    curves with labels, and the legends are added here in the style of the figure.
    color_scheme: "dark", or "light" for the current matplotlib settings.
    """
    def __init__(self, stride, color_scheme="dark"):
        self.stride = stride
        self.count = 0
        self.style = "dark_background" if color_scheme == "dark" else None
    def __call__(self, learner):
        n_losses = len(learner.loss_terms)
        
        # Create figure and plots only once
        if not hasattr(self, "fig"):
            # the style is applied only while the figure is created and saved, so that the
            # matplotlib settings of the user are not changed
            with plot_style(self.style):
                self.fig, self.ax = plt.subplots(
                    nrows=n_losses + 1,
                    ncols=1,
                    figsize=(10, 5 * (n_losses + 1)),
                    sharex=True,
                    squeeze=False,
                )
                self.loss = [[] for _ in range(n_losses + 1)]

                self.ax = self.ax[:, 0]
                self.lines = []

                ax0 = self.ax[0]
                ax0.set_ylabel("Loss")
                ax0.set_yscale("log")
                ax0.set_xscale("linear")
                ax0.set_ylim(1e-7, 0.1)
                ax0.xaxis.tick_top()
                ax0.tick_params(axis="x", labeltop=True, labelbottom=False)
                ax0.set_xlabel("Iteration")
                ax0.xaxis.set_label_position("top")
                ax0.grid(True, alpha=0.3)

                for i in range(n_losses):
                    ax = self.ax[i+1]
                    ax.set_ylabel(learner.loss_terms[i].name)
                    ax.tick_params(axis="x", labelbottom=False)
                    learner.loss_terms[i].ax = ax

                self.loss_curves = []
                for i in range(n_losses):
                    line, = ax0.plot([], [], linewidth=1.0, alpha=0.5, label=learner.loss_terms[i].name)
                    self.loss_curves.append(line)
                line, = ax0.plot([], [], linewidth=1.0, alpha=0.5, label="total")
                self.loss_curves.append(line)            

                ax0.legend()

                # elements such as ticks are created on demand: create them in this style
                self.fig.draw_without_rendering()

        # update content
        total = 0
        for i in range(n_losses):
            self.loss[i].append(learner.loss_terms[i].loss.item())
            total += self.loss[i][-1]
        self.loss[-1].append(total)

        # plot
        if self.count % self.stride == 0:
            dir = learner.dir
            for i in range(len(self.loss_curves)):
                loss_curve = self.loss_curves[i]
                loss = np.array(self.loss[i], dtype=float)
                iterations = np.arange(len(loss))
                valid = np.isfinite(loss) & (loss > 0)
                loss_curve.set_data(iterations[valid], loss[valid])

            # Automatic limits
            ax0 = self.ax[0]
            ax0.relim()
            ax0.autoscale_view()

            # legends of the diagnostics of the loss terms, created in the style of the figure
            with plot_style(self.style):
                for ax in self.ax[1:]:
                    if ax.get_legend_handles_labels()[0]:
                        ax.legend()

            self.fig.canvas.draw_idle()

            save_figure(self.fig, os.path.join(dir, "lc.png"), style=self.style)
        self.count += 1 
