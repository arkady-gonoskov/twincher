# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

import numpy as np
import torch

def _grad_kernel_u(v, u, c, amp, amps, losses):
    x = (v - u).relu_()
    s = torch.sqrt(x.square() + c*c + 1e-15)
    amps[:, :] = amp * x / (s * v) # observe: extra v if for further use
    losses[:, :] += amp*(s - c)

class HSNoiseLoss:
    def __init__(self,
        vel_u = 0.0, # upper bound on velocities
        amp = 1e-4, # loss amplitude
        vel_scale = 1.0, # scale for loss transition from quadratic to linear
        y_range = 0.0, # range of picking y-deviations during training
    ):
        self.name = "hs_noise"
        self.vel_u = vel_u
        self.amp = amp
        self.vel_scale = vel_scale
        self.y_range = y_range

        self.ax = None # request of axis for reporting progress

    def init(self, learner):
        self.learner = learner
        n_c, n_p, n_y, dtype, device, time_monitor = learner.n_c, learner.n_p, learner.n_y, learner.dtype, learner.device, learner.time_monitor
        time_monitor.start(self.name + ".init", torch_sync=True)
        self.A_pinv = torch.empty((n_c, n_p, n_y), dtype=dtype, device=device)

        self.p = torch.empty((n_c, n_p), dtype=dtype, device=device)
        self.y = torch.empty((n_c, n_y), dtype=dtype, device=device)
        self.A = torch.empty((n_c, n_y, n_p), dtype=dtype, device=device)
        self.B = torch.empty((n_c, n_y, n_p), dtype=dtype, device=device)
        self.E = torch.empty((n_c, n_y, n_p), dtype=dtype, device=device)

        self.u = torch.empty((n_c, n_p), dtype=dtype, device=device)
        self.y_offset = torch.empty((n_c, n_y), dtype=dtype, device=device)
        self.offset_p = torch.empty((n_c, n_p), dtype=dtype, device=device)
        self.offset_y = torch.empty((n_c, n_y), dtype=dtype, device=device)
        self.offset_amps = torch.empty((n_c), dtype=dtype, device=device)

        self.y_pert = torch.empty((n_c, n_p, n_y), dtype=dtype, device=device)
        self.y_pert_ = torch.empty((n_c, n_p, n_y), dtype=dtype, device=device)
        self.p_pert = torch.empty((n_c, n_p, n_p), dtype=dtype, device=device)

        self.vels = torch.empty((n_c, n_p), dtype=dtype, device=device)
        self.amps = torch.empty((n_c, n_p), dtype=dtype, device=device)
        self.losses = torch.empty((n_c, n_p), dtype=dtype, device=device)

        self.maximal_vel = 0.0
        self.vel_p75 = 0

        time_monitor.stop(self.name + ".init", torch_sync=True)
        self.state = "initiated"

    def compute_grad(self, update_state = False, test = False):
        # borrow tools:
        SH, time_monitor, rng = self.learner.SH, self.learner.time_monitor, self.learner.rng
        
        n_p, n_y, n_s, n_c = self.learner.n_p, self.learner.n_y, self.learner.n_s, self.learner.n_c

        time_monitor.start(self.name, torch_sync=True)

        # copy batch:
        self.p.copy_(self.learner.p)
        self.y.copy_(self.learner.y)
        self.A.copy_(self.learner.dy_dp)

        torch.linalg.pinv(self.A, out=self.A_pinv)

        if self.y_range > 0:
            self.y_offset.uniform_(-1, 1, generator=rng)
            torch.matmul(
                self.A_pinv, 
                self.y_offset[:, :, None],
                out=self.offset_p[:, :, None]
            )
            torch.matmul(
                self.A, 
                self.offset_p[:, :, None],
                out=self.offset_y[:, :, None]
            )
            self.y_offset[:, :] -= self.offset_y[:, :]
            self.offset_y[:, :] /= torch.linalg.vector_norm(
                self.offset_y[:, :], dim=1, keepdim=True
            ).clamp_min(1e-8)

            self.offset_amps.uniform_(0, self.y_range, generator=rng)
            self.offset_y[:, :] *= self.offset_amps[:, None]
            self.y[:, :] += self.offset_y[:, :]
            self.y.clamp_min_(-1.0)
            self.y.clamp_max_(1.0)

        self.y_pert.uniform_(-1, 1, generator=rng)
        torch.matmul(
            self.y_pert[:, :, :],
            self.A_pinv.permute(0, 2, 1),
            out=self.p_pert[:, :, :]
        )
        torch.matmul(
            self.p_pert[:, :, :], 
            self.A.permute(0, 2, 1),
            out=self.y_pert_[:, :, :]
        )
        self.y_pert[:, :, :] -= self.y_pert_[:, :, :]
        self.y_pert[:, :, :] /= torch.linalg.vector_norm(
            self.y_pert[:, :, :], dim=2, keepdim=True
        ).clamp_min(1e-8)
      
        SH.s[:n_y, :].permute(1, 0).copy_(self.y[:, :])
        SH.s[n_y:, :].fill_(0)

        SH.Q.fill_(0)
        SH.Q[:n_y, :, :].permute(2, 1, 0).copy_(self.y_pert[:, :, :])

        SH.forward_query()

        torch.linalg.vector_norm(SH.Q[:n_p, :, :].permute(2, 0, 1), out=self.vels, dim=1, keepdim=False)
        self.vels.clamp_min_(1e-8)

        SH.b.fill_(0)
        SH.B.fill_(0)

        SH.B[:n_p, :, :] = SH.Q[:n_p, :, :]
        self.losses[:, :] = 0

        _grad_kernel_u(v=self.vels, u=self.vel_u, c=self.vel_scale, amp=self.amp, amps=self.amps, losses=self.losses)
        self.loss = self.losses.sum(dim=(0, 1))/n_c
        SH.B[:n_p, :, :] *= self.amps[:, None, :].permute(1, 2, 0)
        
        SH.backward_query()

        self.maximal_vel = max(self.maximal_vel, self.vels.max())

        if update_state:
            self.vel_p75 = torch.quantile(self.vels, 0.75)        
            self.state = f"<{self.vel_p75:4.2f}|{self.maximal_vel:4.2f}>"

            if self.ax is not None:
                if hasattr(self, "ax_iter") == False:
                    self.ax_iter = []
                    self.ax_vel_p75 = []
                    self.ax_maximal_vel = []
                    self.ax_line_vel_p75, = self.ax.plot([], [], color="red", lw=0.8, label=r'P75($\|\partial r / \partial p^\perp\|$)')
                    self.ax_line_maximal_vel, = self.ax.plot([], [], color="yellow", lw=0.8, label=r'max($\|\partial r /\partial p^\perp\|$)')
                self.ax_iter.append(self.learner.iter*1.0)
                self.ax_vel_p75.append(self.vel_p75.item())
                self.ax_maximal_vel.append(self.maximal_vel.item())
                
                self.ax.grid(alpha=0.2)
                self.ax.set_ylim(0.01, 1000.0)
                self.ax.set_yscale("log")
                self.ax_line_vel_p75.set_data(np.array(self.ax_iter), np.array(self.ax_vel_p75))
                self.ax_line_maximal_vel.set_data(np.array(self.ax_iter), np.array(self.ax_maximal_vel))
                
                self.ax.relim()
                self.ax.autoscale_view()

            self.maximal_vel = 0.0

        time_monitor.stop(self.name, torch_sync=True)
