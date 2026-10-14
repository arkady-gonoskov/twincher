# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

import numpy as np
import torch

from .hs_noise_loss import _pinv, _warn_without_normal_space


def _grad_kernel_u(v, u, c, amp, amps, losses):
    x = (v - u).relu_()
    s = torch.sqrt(x.square() + c * c + 1e-15)
    amps[:] = amp * x / (s * v)  # observe: extra v is for further use
    losses[:] += amp * (s - c)


class HSSMNoiseLoss:
    """Single-mode noise loss of the hyper-sail architecture, an alternative to HSNoiseLoss.

    For each case, it finds the direction of y perpendicular to the manifold y(p) in which r
    changes most per unit change of y: the top right singular vector of (dr/dy) P_N, where
    P_N projects onto the directions perpendicular to the columns of dy/dp. Only this single
    mode is penalized; its gradient is computed with the direction held fixed (Danskin's
    theorem). With y_range > 0, the cases are shifted from the manifold by random
    perpendicular offsets of norm up to y_range.
    """

    def __init__(
        self,
        vel_u=0.0,  # upper bound on velocities
        amp=1e-3,  # loss amplitude
        vel_scale=1.0,  # scale for loss transition from quadratic to linear
        y_range=0.0,  # range of picking y-deviations during training
    ):
        self.name = "hs_sm_noise"
        self.vel_u = vel_u
        self.amp = amp
        self.vel_scale = vel_scale
        self.y_range = y_range

        self.ax = None  # request of axis for reporting progress

    def init(self, learner):
        self.learner = learner
        n_c, n_p, n_y, dtype, device, time_monitor = (
            learner.n_c,
            learner.n_p,
            learner.n_y,
            learner.dtype,
            learner.device,
            learner.time_monitor,
        )
        _warn_without_normal_space(self.name, n_y, n_p)
        time_monitor.start(self.name + ".init", torch_sync=True)
        self.A_pinv = torch.empty((n_c, n_p, n_y), dtype=dtype, device=device)

        self.y = torch.empty((n_c, n_y), dtype=dtype, device=device)
        self.A = torch.empty((n_c, n_y, n_p), dtype=dtype, device=device)
        self.D = torch.empty((n_c, n_p, n_p), dtype=dtype, device=device)
        self.B = torch.empty((n_c, n_y, n_p), dtype=dtype, device=device)
        self.E = torch.empty((n_c, n_y, n_p), dtype=dtype, device=device)

        self.u = torch.empty((n_c, n_p), dtype=dtype, device=device)
        self.y_max = torch.empty((n_c, n_y), dtype=dtype, device=device)
        self.y_offset = torch.empty((n_c, n_y), dtype=dtype, device=device)
        self.p_offset = torch.empty((n_c, n_p), dtype=dtype, device=device)
        self.y_tang = torch.empty((n_c, n_y), dtype=dtype, device=device)
        self.offset_amps = torch.empty((n_c), dtype=dtype, device=device)

        self.vel_max = torch.empty((n_c), dtype=dtype, device=device)
        self.norms = torch.empty((n_c), dtype=dtype, device=device)
        self.amps = torch.empty((n_c), dtype=dtype, device=device)
        self.losses = torch.empty((n_c), dtype=dtype, device=device)

        self.maximal_vel = 0.0
        self.vel_p75 = 0

        time_monitor.stop(self.name + ".init", torch_sync=True)
        self.state = "initiated"

    def compute_grad(self, update_state=False, test=False):
        # borrow tools:
        SH, time_monitor, rng = (
            self.learner.SH,
            self.learner.time_monitor,
            self.learner.rng,
        )

        n_p, n_y, n_c = self.learner.n_p, self.learner.n_y, self.learner.n_c

        time_monitor.start(self.name, torch_sync=True)

        # copy batch:
        self.y.copy_(self.learner.y)
        self.A.copy_(self.learner.dy_dp)

        _pinv(self.A, out=self.A_pinv)

        if self.y_range > 0:
            self.y_offset.uniform_(-1, 1, generator=rng)
            torch.matmul(
                self.A_pinv, self.y_offset[:, :, None], out=self.p_offset[:, :, None]
            )
            torch.matmul(self.A, self.p_offset[:, :, None], out=self.y_tang[:, :, None])
            self.y_offset[:, :] -= self.y_tang[:, :]
            self.y_offset[:, :] /= torch.linalg.vector_norm(
                self.y_offset[:, :], dim=1, keepdim=True
            ).clamp_min(1e-8)

            self.offset_amps.uniform_(0, self.y_range, generator=rng)
            self.y_offset[:, :] *= self.offset_amps[:, None]
            self.y[:, :] += self.y_offset[:, :]
            self.y.clamp_min_(-1.0)
            self.y.clamp_max_(1.0)

        SH.s[:n_y, :].permute(1, 0).copy_(self.y[:, :])
        SH.s[n_y:, :].fill_(0)
        SH.compute_V()
        self.B[:, :, :].copy_(SH.V[:, :n_y, :].permute(2, 1, 0))
        torch.matmul(self.A_pinv[:, :, :], self.B[:, :, :], out=self.D[:, :, :])
        torch.matmul(self.A[:, :, :], self.D[:, :, :], out=self.E[:, :, :])
        self.B[:, :, :] -= self.E[:, :, :]
        torch.matmul(
            self.B[:, :, :].permute(0, 2, 1), self.B[:, :, :], out=self.D[:, :, :]
        )
        _, U = torch.linalg.eigh(self.D)  # eigenvalues in ascending order
        self.u[:, :] = U[:, :, -1]
        torch.matmul(self.B[:, :, :], self.u[:, :, None], out=self.y_max[:, :, None])

        torch.linalg.vector_norm(self.y_max, out=self.norms, dim=1, keepdim=False)
        self.norms.clamp_min_(1e-15)

        self.y_max /= self.norms[..., None]

        n_m = SH.n_m  # restored after the passes with the single mode
        SH.resize_n_m(1)

        SH.s[:n_y, :].permute(1, 0).copy_(self.y[:, :])
        SH.s[n_y:, :].fill_(0)

        SH.Q.fill_(0)
        SH.Q[:n_y, 0, :].permute(1, 0).copy_(self.y_max[:, :])
        SH.forward_query()

        torch.linalg.vector_norm(
            SH.Q[:n_p, 0, :], out=self.vel_max, dim=0, keepdim=False
        )
        self.vel_max.clamp_min_(1e-8)

        SH.b.fill_(0)
        SH.B.fill_(0)

        SH.B[:n_p, 0, :] = SH.Q[:n_p, 0, :]
        self.losses[:] = 0
        _grad_kernel_u(
            v=self.vel_max,
            u=self.vel_u,
            c=self.vel_scale,
            amp=self.amp,
            amps=self.amps,
            losses=self.losses,
        )
        self.loss = self.losses.sum() / n_c
        SH.B[:n_p, 0, :] *= self.amps[None, :]

        SH.backward_query()

        SH.resize_n_m(n_m)

        self.maximal_vel = max(self.maximal_vel, self.vel_max.max())

        if update_state:
            self.vel_p75 = torch.quantile(self.vel_max, 0.75)
            self.state = f"<{self.vel_p75:4.2f}|{self.maximal_vel:4.2f}>"

            if self.ax is not None:
                if not hasattr(self, "ax_iter"):
                    self.ax_iter = []
                    self.ax_vel_p75 = []
                    self.ax_maximal_vel = []
                    (self.ax_line_vel_p75,) = self.ax.plot(
                        [],
                        [],
                        color="red",
                        lw=0.8,
                        label=r"P75($\|\partial r / \partial p^\perp\|$)",
                    )
                    (self.ax_line_maximal_vel,) = self.ax.plot(
                        [],
                        [],
                        color="yellow",
                        lw=0.8,
                        label=r"max($\|\partial r /\partial p^\perp\|$)",
                    )
                self.ax_iter.append(self.learner.iter * 1.0)
                self.ax_vel_p75.append(self.vel_p75.item())
                self.ax_maximal_vel.append(self.maximal_vel.item())

                self.ax.grid(alpha=0.2)
                self.ax.set_ylim(0.01, 1000.0)
                self.ax.set_yscale("log")
                self.ax_line_vel_p75.set_data(
                    np.array(self.ax_iter), np.array(self.ax_vel_p75)
                )
                self.ax_line_maximal_vel.set_data(
                    np.array(self.ax_iter), np.array(self.ax_maximal_vel)
                )

                self.ax.relim()
                self.ax.autoscale_view()

            self.maximal_vel = 0.0

        time_monitor.stop(self.name, torch_sync=True)
