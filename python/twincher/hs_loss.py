# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

import numpy as np
import torch

class HSLoss: # hyper-sail loss controller
    def __init__(self,
        det_margin = 0.8, # target margin for determinants
        amp = 3.0, # amplitude of loss
        det_clearance = 0.5, # threshold for triggering expansion
        p_range_step = 0.01, # increment for p_range
        ramp_boost = 2.0, # loss amplitude boost at the periphery 
        ramp_size = 0.2, # size of the ramp for the loss boost 
        trimmer_threshold = 1.3, # threshold for the loss on velocities
        trimmer_amp = 1.0, # amplitude of loss for velocities
        promotion_threshold = 10, # the number of valid-state iterations to pass until promotion
    ):
        self.name = "hs"
        self.det_margin = det_margin
        self.amp = amp
        self.det_clearance = det_clearance
        self.p_range_step = p_range_step
        self.ramp_boost = ramp_boost
        self.ramp_size = ramp_size
        self.trimmer_threshold = trimmer_threshold
        self.trimmer_amp = trimmer_amp
        self.promotion_threshold = promotion_threshold
         
        self.ax = None # request of axis for reporting progress

    def init(self, learner):
        self.learner = learner
        n_c, n_s, n_p, n_y, dtype, device = learner.n_c, learner.n_s, learner.n_p, learner.n_y, learner.dtype, learner.device

        self.dets = torch.zeros((n_c,), dtype=dtype, device=device)
        self.J = torch.empty((n_c, n_p, n_p), dtype=dtype, device=device)
        self.D = torch.empty((n_c, n_p, n_p), dtype=dtype, device=device)
        self.B = torch.empty((n_c, n_p, n_p), dtype=dtype, device=device)

        self.p_range = self.ramp_size
        learner.p_range = self.p_range

        self.neg_dets_count = 0
        self.stats_det_min = float('inf')

        self.state = "initiated"

    def compute_grad(self, update_state = False, test = False):
        # import tensors' sizes:
        n_p, n_y, n_s, n_c = self.learner.n_p, self.learner.n_y, self.learner.n_s, self.learner.n_c

        # borrow tools:
        learner = self.learner
        SH, time_monitor, rng = learner.SH, learner.time_monitor, learner.rng

        # set metadata
        if self.state == "initiated":
            learner.SH.metadata["twincher_type"] = "hs"
            learner.SH.metadata["stencil_name"] = learner.stencil_name
            learner.SH.metadata["n_p"] = n_p
            learner.SH.metadata["n_y"] = n_y
            self.iter = 0
            self.valid_det_count = 0
            self.state = "learning"            

        time_monitor.start(self.name, torch_sync=True)

        if test: 
            print(self.name, "in test mode")
            self.p_range = self.ramp_size # do not alter to reproduce loss
            rng.manual_seed(42) # setting for repeatability of batch selection

        # import batch:
        p = self.learner.p
        y = self.learner.y
        dy_dp = self.learner.dy_dp

        # forward pass:
        SH.s[:n_y, :].permute(1, 0).copy_(y[:, :])
        SH.s[n_y:, :].fill_(0)
        SH.Q.fill_(0)
        SH.Q[:n_y, :, :].permute(2, 0, 1).copy_(dy_dp[:, :, :])

        SH.forward_query()

        # set b and B tensors
        SH.b.fill_(0)
        SH.B.fill_(0)

        # det loss:
        self.J.copy_(SH.Q[:n_p, :, :].permute(2, 1, 0))
        self.dets = torch.linalg.det(self.J)

        active = (self.dets.abs() > 1e-7) & (self.dets < self.det_margin)

        J_a = self.J[active]
        inv_J = torch.linalg.inv(J_a)

        case_boost = p[active, :]
        case_boost.abs_()
        case_boost[:, :] -= (self.p_range - self.ramp_size)
        case_boost[:, :].relu_()
        amps = case_boost.sum(dim=1)
        amps[:] *= (self.ramp_boost - 1.0)*self.amp
        amps[:] += self.amp
        amps.relu_()

        f = self.dets[active]
        f -= self.det_margin
        losses = amps[:]*f[:]*f[:]
        self.det_loss = losses.sum() / n_c
        f *= 2.0
        f *= amps[:]
        f *= self.dets[active]
        inv_J[:, :] *= f[:, None, None]

        self.B.fill_(0)
        self.B[active, :, :] = inv_J
        SH.B[:n_p, :, :].permute(2, 0, 1).copy_(self.B)

        # trimmer loss:
        self.D.copy_(SH.Q[:n_p, :, :].permute(2, 0, 1))
        norms = torch.linalg.vector_norm(self.D, dim=1)
        a = norms.clone()
        a -= self.trimmer_threshold
        a.relu_()
        self.trimmer_loss = self.trimmer_amp*(a*a).sum()/n_c
        self.loss = self.det_loss + self.trimmer_loss
        torch.clamp_min_(norms, 1e-10)
        a *= 2*self.trimmer_amp
        a /= norms

        self.D[:, :, :] *= a[:, None, :]

        SH.B[:n_p, :, :].permute(2, 0, 1)[:, :, :] += self.D[:, :, :]

        # make backward pass:
        SH.backward_query()

        self.neg_dets_count += (self.dets[:] < 0).to(torch.int32).sum()
        self.stats_det_min = min(self.stats_det_min, self.dets.min())

        # p_range controller:
        if self.dets.min() > self.det_clearance:
            self.valid_det_count += 1
        else:
            self.valid_det_count = 0
        if self.valid_det_count > self.promotion_threshold:
            self.p_range = min(1.0, self.p_range + self.p_range_step)
            self.learner.p_range = self.p_range
        
        if update_state:
            self.stats_grad_max = norms.max()
            self.state = f"<{self.stats_det_min:4.2f}|{self.stats_grad_max:4.2f}>"
            self.state += f"[{self.neg_dets_count}]({self.p_range:6.3f})"

            if self.ax is not None:
                if hasattr(self, "ax_iter") == False:
                    self.ax_iter = []
                    self.ax_min = []
                    self.ax_grad_max = []
                    self.ax_line_min, = self.ax.plot([], [], color="red", lw=0.8, label=r'min($\det(\partial r / \partial p)$)')
                    self.ax_line_grad_max, = self.ax.plot([], [], color="yellow", lw=0.8, label=r'max($\|\partial r /\partial p\|$)')
                self.ax_iter.append(self.learner.iter*1.0)
                self.ax_min.append(self.stats_det_min.item())
                self.ax_grad_max.append(self.stats_grad_max.item())
                
                self.ax.grid(alpha=0.2)
                self.ax.set_ylim(-1.0, 3.0)
                self.ax_line_min.set_data(np.array(self.ax_iter), np.array(self.ax_min))
                self.ax_line_grad_max.set_data(np.array(self.ax_iter), np.array(self.ax_grad_max))
                
                self.ax.relim()
                self.ax.autoscale_view()

            self.neg_dets_count = 0
            self.stats_det_min = 100.0
        
        self.iter += 1
        time_monitor.stop(self.name, torch_sync=True)

if __name__ == "__main__":
    L = HSLoss()
    print(L.name)
