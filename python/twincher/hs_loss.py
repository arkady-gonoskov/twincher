# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

import collections

import numpy as np
import torch


class HSLoss:  # hyper-sail loss controller
    """Loss term of the hs architecture: det(dr/dp) above det_margin and |dr/dp_k| below
    trimmer_threshold (the trimmer) on the cases of the box |p|_inf <= p_range, which
    grows (curriculum) while the determinants stay above det_clearance.

    - Ramp: the amplitude of the determinant loss changes linearly from amp to
      amp * ramp_boost over the margin p_range - ramp_size < |p|_inf <= p_range of the
      box. ramp_mode "sum" is the ramp of version 0.1.0, kept to reproduce trainings
      made with it: amp * (1 + (ramp_boost - 1) * sum_k relu(|p_k| - p_range +
      ramp_size)). With ramp_ext > 0, the cases are drawn up to |p|_inf <= p_range +
      ramp_ext, with an amplitude decreasing linearly to 0 there.
    - Promotion: p_range grows by p_range_step after more than promotion_threshold
      consecutive valid batches: all cases inside the box with det > det_clearance or,
      with promotion_tol > 0, none with det <= 0 and at most this fraction with
      det <= det_clearance. The box starts at p_range_init (default: ramp_size).
    - Trimmer over the whole domain (trimmer_domain_amp > 0): in a second pass, the
      trimmer alone acts on n_c extra cases drawn uniformly in [-1, 1]^n_p. Nothing
      else constrains r outside the box, and its derivatives there can become so large
      that the training collapses when the box grows into such a region.
    - Momentum reset (momentum_reset: True or a dict of parameters of MomentumReset;
      default None: off): a batch with a sudden rise of the number of det < 0 sets the
      parameters back and Adam's first moment to zero, so that a spike of the gradient
      cannot carry the map away. Meant for the late stages of a training and for the
      continued training of a trained twincher, when the map should stay clean. While
      the box grows from a small start, isolated cases with det < 0 at its new border
      are normal, and resets there slow the training down or stop it.
    """

    def __init__(
        self,
        det_margin=0.8,  # target margin for determinants
        amp=3.0,  # amplitude of loss
        det_clearance=0.5,  # threshold for triggering expansion
        p_range_step=0.01,  # increment for p_range
        ramp_boost=1.0,  # loss amplitude factor at the border of the box
        ramp_size=0.2,  # width of the ramp of the loss amplitude (and initial p_range)
        trimmer_threshold=1.3,  # threshold for the loss on velocities
        trimmer_amp=1.0,  # amplitude of loss for velocities
        promotion_threshold=10,  # the number of valid-state iterations to pass until promotion
        trimmer_domain_amp=1.0,  # trimmer amplitude over the whole domain (0: off)
        momentum_reset=None,  # True or dict of parameters of MomentumReset (None: off)
        ramp_mode="max",  # "max", or "sum" (the ramp of version 0.1.0)
        ramp_ext=0.0,  # cases also drawn beyond the box, up to p_range + ramp_ext
        promotion_tol=0.0,  # allowed fraction of det <= det_clearance for a promotion
        p_range_init=None,  # initial p_range (None: ramp_size)
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
        self.trimmer_domain_amp = trimmer_domain_amp
        self.momentum_reset = momentum_reset
        self.ramp_mode = ramp_mode
        self.ramp_ext = ramp_ext
        self.promotion_tol = promotion_tol
        self.p_range_init = p_range_init

        self.ax = None  # request of axis for reporting progress

    def init(self, learner):
        self.learner = learner
        n_c, n_s, n_p, n_y, dtype, device = learner.n_c, learner.n_s, learner.n_p, learner.n_y, learner.dtype, learner.device
        if self.ramp_mode not in ("max", "sum"):
            raise ValueError(
                f'unknown ramp_mode "{self.ramp_mode}" (expected "max" or "sum")'
            )

        self.dets = torch.zeros((n_c,), dtype=dtype, device=device)
        self.J = torch.empty((n_c, n_p, n_p), dtype=dtype, device=device)
        self.D = torch.empty((n_c, n_p, n_p), dtype=dtype, device=device)
        self.B = torch.empty((n_c, n_p, n_p), dtype=dtype, device=device)

        # cases of the trimmer over the whole domain
        if self.trimmer_domain_amp > 0:
            self.p_domain = torch.zeros((n_c, n_p), dtype=dtype, device=device)
            self.y_domain = torch.zeros((n_c, n_y), dtype=dtype, device=device)
            self.dy_dp_domain = torch.zeros((n_c, n_y, n_p), dtype=dtype, device=device)
        self.domain_loss = 0.0
        self.domain_norm_max = 0.0

        self.reset = None
        if self.momentum_reset:
            params = {} if self.momentum_reset is True else dict(self.momentum_reset)
            self.reset = MomentumReset(**params)

        self.p_range = self.ramp_size if self.p_range_init is None else (
            self.p_range_init
        )
        learner.p_range = min(1.0, self.p_range + self.ramp_ext)

        self.neg_dets_count = 0
        self.stats_det_min = float("inf")

        self.state = "initiated"

    def compute_grad(self, update_state=False, test=False):
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
            self.p_range = self.ramp_size  # do not alter to reproduce loss
            rng.manual_seed(42)  # setting for repeatability of batch selection

        reset = None if test else self.reset
        if reset is not None:
            reset.before(learner)

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

        if self.ramp_mode == "max":
            # amp inside, changing linearly to amp*ramp_boost in the margin
            # p_range - ramp_size < |p|_inf <= p_range; beyond p_range (ramp_ext > 0)
            # decreasing linearly from amp*ramp_boost to 0 at p_range + ramp_ext
            p_max = p[active, :].abs().amax(dim=1)
            t = ((p_max - (self.p_range - self.ramp_size)) / self.ramp_size).clamp(0, 1)
            amps = self.amp * (1.0 + (self.ramp_boost - 1.0) * t)
            if self.ramp_ext > 0:
                beyond = (p_max - self.p_range).clamp(min=0)
                amps *= (1.0 - beyond / self.ramp_ext).clamp(min=0)
        else:  # "sum": the ramp of version 0.1.0
            case_boost = p[active, :]
            case_boost.abs_()
            case_boost[:, :] -= self.p_range - self.ramp_size
            case_boost[:, :].relu_()
            amps = case_boost.sum(dim=1)
            amps[:] *= (self.ramp_boost - 1.0) * self.amp
            amps[:] += self.amp
        amps.relu_()

        f = self.dets[active]
        f -= self.det_margin
        losses = amps[:] * f[:] * f[:]
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
        self.norm_max = norms.max()  # largest |dr/dp_k| of the batch (diagnostics)
        a = norms.clone()
        a -= self.trimmer_threshold
        a.relu_()
        self.trimmer_loss = self.trimmer_amp * (a * a).sum() / n_c
        self.loss = self.det_loss + self.trimmer_loss
        torch.clamp_min_(norms, 1e-10)
        a *= 2 * self.trimmer_amp
        a /= norms

        self.D[:, :, :] *= a[:, None, :]

        SH.B[:n_p, :, :].permute(2, 0, 1)[:, :, :] += self.D[:, :, :]

        # make backward pass:
        SH.backward_query()

        # trimmer over the whole domain (second pass, gradients added):
        if self.trimmer_domain_amp > 0:
            self._trimmer_domain()
            self.loss = self.loss + self.domain_loss

        self.neg_dets_count += (self.dets[:] < 0).to(torch.int32).sum()
        self.stats_det_min = min(self.stats_det_min, self.dets.min())

        # p_range controller (cases beyond p_range, with ramp_ext > 0, do not count):
        dets = self.dets_inner = self.dets[p.abs().amax(dim=1) <= self.p_range]
        if self.promotion_tol > 0:
            low = (dets <= self.det_clearance).float().mean()
            valid = dets.numel() == 0 or (dets.min() > 0 and low <= self.promotion_tol)
        else:
            valid = dets.numel() == 0 or dets.min() > self.det_clearance
        if valid:
            self.valid_det_count += 1
        else:
            self.valid_det_count = 0
        if self.valid_det_count > self.promotion_threshold:
            self.p_range = min(1.0, self.p_range + self.p_range_step)
            self.learner.p_range = min(1.0, self.p_range + self.ramp_ext)

        if reset is not None:
            reset.judge(self.dets_inner)

        if update_state:
            self.stats_grad_max = norms.max()
            self.state = f"<{self.stats_det_min:4.2f}|{self.stats_grad_max:4.2f}"
            if self.trimmer_domain_amp > 0:
                self.state += f"|{self.domain_norm_max:4.2f}"
            self.state += f">[{self.neg_dets_count}]({self.p_range:6.3f})"
            if self.reset is not None:
                self.state += f"resets:{self.reset.n}"

            if self.ax is not None:
                if hasattr(self, "ax_iter") == False:
                    self.ax_iter = []
                    self.ax_min = []
                    self.ax_grad_max = []
                    self.ax_line_min, = self.ax.plot([], [], color="red", lw=0.8, label=r'min($\det(\partial r / \partial p)$)')
                    self.ax_line_grad_max, = self.ax.plot([], [], color="yellow", lw=0.8, label=r'max($\|\partial r /\partial p\|$)')
                self.ax_iter.append(self.learner.iter * 1.0)
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

    def _trimmer_domain(self):
        """The trimmer alone on n_c cases drawn uniformly in [-1, 1]^n_p; the gradient
        of the first pass (in SH.g) is added to that of this one."""
        learner = self.learner
        SH, n_p, n_y, n_c = learner.SH, learner.n_p, learner.n_y, learner.n_c
        g_box = SH.g.clone()
        p, y, dy_dp = self.p_domain, self.y_domain, self.dy_dp_domain
        p.uniform_(-1.0, 1.0, generator=learner.rng)
        if hasattr(learner, "stencil"):
            learner.stencil.compute(p, y, dy_dp)
        else:
            learner.MLI.interpolate(p, y, learner.data_y)
            learner.MLI.interpolate(p, dy_dp, learner.data_dy_dp)
        SH.s[:n_y].copy_(y.T)
        SH.s[n_y:].fill_(0)
        SH.Q.fill_(0)
        SH.Q[:n_y].copy_(dy_dp.permute(1, 2, 0))
        SH.forward_query()
        D = SH.Q[:n_p].permute(2, 0, 1)  # (n_c, n_p components of r, n_p columns k)
        norms = torch.linalg.vector_norm(D, dim=1)  # |dr/dp_k|
        a = (norms - self.trimmer_threshold).relu()
        self.domain_loss = self.trimmer_domain_amp * (a * a).sum() / n_c
        self.domain_norm_max = norms.max()  # largest |dr/dp_k| here (diagnostics)
        # B = d loss / d Q per case (the core divides by n_c), as for the trimmer above
        scale = 2 * self.trimmer_domain_amp * a / norms.clamp(min=1e-10)
        SH.b.fill_(0)
        SH.B.fill_(0)
        SH.B[:n_p].copy_((D * scale[:, None, :]).permute(1, 2, 0))
        sga, SH.core.sga = SH.core.sga, 0.0  # (the s-gate regularization: first pass)
        SH.backward_query()
        SH.core.sga = sga
        SH.g.add_(g_box)


class MomentumReset:
    """Momentum reset of HSLoss (option momentum_reset), against spikes of the loss.

    Adam's first moment carries a spike of the gradient for ~1 / (1 - beta1) steps,
    while its second moment keeps the normalization for ~1 / (1 - beta2) steps: each
    affected parameter moves by up to ~1 / sqrt(1 - beta2) (about 30) learning rates in
    the same direction, which can break the map within a few iterations ("explosion").

    Each batch is judged by the number of its cases inside the box with det < 0,
    evaluated with the parameters before its update. The batch is rejected when this
    number exceeds both neg_frac of these cases and ratio times its running mean over
    the previous batches (factor beta; all batches, so that a persistent level becomes
    the reference after ~1 / (1 - beta) iterations): in a clean state this is the
    absolute criterion, in a state with a few det < 0 only a clear deterioration. After
    the optimizer step of a rejected batch, the parameters are set back to the last
    accepted state (the one before the update that led to this batch; when the batch of
    the state set back to is rejected too, one accepted state further back, up to
    depth), Adam's first moment is zeroed (the second moment, raised by the spike, is
    kept and damps the following steps), and the factor boost of the learning rate is
    multiplied by lr_factor, not below lr_floor. boost rises by lr_up per iteration back
    to 1; the learner's lr is multiplied by it during each step. While paused is True,
    batches are not judged (e.g. while only the moments of Adam are primed).
    """

    def __init__(
        self,
        neg_frac=1e-3,  # fraction of the cases with det < 0 that can trigger a reset
        ratio=3.0,  # ... and factor over their running mean
        lr_factor=0.5,  # factor of the learning rate at a reset
        lr_up=1.1,  # factor per iteration back to the nominal learning rate
        lr_floor=0.05,  # smallest factor of the learning rate
        beta=0.99,  # factor of the running mean of the number of det < 0
        depth=4,  # number of accepted states kept
    ):
        self.neg_frac, self.ratio = neg_frac, ratio
        self.lr_factor, self.lr_up, self.lr_floor = lr_factor, lr_up, lr_floor
        self.beta = beta
        self.good = collections.deque(maxlen=depth)  # accepted states
        self.boost, self.n, self.ema = 1.0, 0, 0.0
        self.reverted, self.current, self.target = False, None, None
        self.paused = False
        self.learner, self.optimizer, self.hook = None, None, None
        self.lr_nominal, self.armed = None, False

    def state(self):
        """The state to keep with a checkpoint (without the accepted parameters)."""
        return {"boost": self.boost, "n": self.n, "ema": self.ema}

    def load_state(self, state):
        """Continue from state() (None: from the start); accepted states are lost."""
        state = state or {"boost": 1.0, "n": 0, "ema": 0.0}
        self.boost, self.n, self.ema = state["boost"], state["n"], state["ema"]
        self.good.clear()
        self.reverted, self.current, self.target = False, None, None

    def before(self, learner):
        """Before the batch is evaluated: keep the parameters, scale the learning
        rate."""
        if self.optimizer is not learner.optimizer:  # also after a replacement
            if self.hook is not None:
                self.hook.remove()
            self.hook = learner.optimizer.register_step_post_hook(self._after_step)
            self.optimizer = learner.optimizer
        self.learner = learner
        self.current = learner.SH.a.clone()
        self.lr_nominal = learner.lr
        learner.lr = self.lr_nominal * self.boost
        learner.lr_dynamic = min(learner.lr_dynamic, learner.lr)
        self.armed = True

    def judge(self, dets):
        """After the batch is evaluated (det J of its cases inside the box): accept the
        parameters or reject them (they are set back after the optimizer step)."""
        if self.paused:
            return
        self.boost = min(1.0, self.boost * self.lr_up)
        n_neg = int((dets < 0).sum().item())
        limit = max(self.neg_frac * dets.numel(), self.ratio * self.ema)
        self.ema = self.beta * self.ema + (1 - self.beta) * n_neg
        if n_neg <= limit:
            if not self.reverted:  # (a state set back to is in good already)
                self.good.append(self.current)
            self.reverted = False
            return
        if self.reverted and len(self.good) > 1:
            self.good.pop()  # the state set back to is rejected as well
        self.target = self.good[-1] if self.good else self.current
        self.boost = max(self.lr_floor, self.boost * self.lr_factor)
        self.n += 1
        self.reverted = True

    def _after_step(self, optimizer, args, kwargs):
        if not self.armed:
            return
        self.armed = False
        learner = self.learner
        learner.lr = self.lr_nominal
        if self.target is not None:
            learner.SH.a.copy_(self.target)  # (followed by the learner's tw_update())
            state = optimizer.state[learner.SH.param]
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            self.target = None


if __name__ == "__main__":
    L = HSLoss()
    print(L.name)
