"""Antipodal separation loss for the hs architecture (experiment).

The hs loss constrains det J at single points. A branch point (r winds twice around a
point of p, like z -> z^2; the "persistent fold") has det J >= 0 everywhere and det J < 0.5
only in a small disc, so the hs loss hardly sees it, and the map stays non-injective.
This term compares r at pairs of points p_c + d and p_c - d with small random offsets
d (|d| uniform in [d_min, d_max], random direction, p_c uniform in [-R, R]^n_p): around
a branch point both map to nearly the same r. Loss per pair: amp * relu(kappa |2 d| -
|r(p_c + d) - r(p_c - d)|)^2, summed and divided by the number of pairs. A healthy
embedding (|dr/dp| ~ 1, singular values of J >= ~0.4 with the hs loss) is not penalized
for kappa below that.

One extra forward and backward pass per iteration on n_c cases (n_c / 2 pairs), with y
from the learner's data (grid interpolation) or stencil. The gradient is passed through
b (n_c d loss / d s: the core divides by n_c), B = 0; the s-gate regularization of the
core is switched off during this pass (it acts in the pass of the hs loss).
"""

import torch


class HSSepLoss:
    def __init__(
        self,
        amp=3.0,  # amplitude of the loss
        kappa=0.3,  # minimal ratio |r(p_c + d) - r(p_c - d)| / |2 d|
        d_min=0.01,  # range of |d|
        d_max=0.05,
    ):
        self.name = "hs_sep"
        self.amp, self.kappa, self.d_min, self.d_max = amp, kappa, d_min, d_max
        self.ax = None

    def init(self, learner):
        self.learner = learner
        n_c, n_p, n_y = learner.n_c, learner.n_p, learner.n_y
        dtype, device = learner.dtype, learner.device
        self.n_pairs = n_c // 2
        self.p = torch.zeros((n_c, n_p), dtype=dtype, device=device)
        self.y = torch.zeros((n_c, n_y), dtype=dtype, device=device)
        self.dy_dp = torch.zeros((n_c, n_y, n_p), dtype=dtype, device=device)
        self.loss = 0.0
        self.frac_active = 0.0
        self.state = "initiated"

    def compute_grad(self, update_state=False, test=False):
        learner = self.learner
        SH, rng, n_p, n_y = learner.SH, learner.rng, learner.n_p, learner.n_y
        n = self.n_pairs
        learner.time_monitor.start(self.name, torch_sync=True)
        R = learner.p_range
        c = torch.empty((n, n_p), dtype=self.p.dtype, device=self.p.device)
        c.uniform_(-R, R, generator=rng)
        u = torch.randn(c.shape, dtype=c.dtype, device=c.device, generator=rng)
        u /= u.norm(dim=1, keepdim=True).clamp(min=1e-12)
        size = torch.empty((n, 1), dtype=c.dtype, device=c.device)
        size.uniform_(self.d_min, self.d_max, generator=rng)
        d = u * size
        self.p[:n] = (c + d).clamp(-1, 1)
        self.p[n : 2 * n] = (c - d).clamp(-1, 1)
        if hasattr(learner, "stencil"):
            learner.stencil.compute(self.p, self.y, self.dy_dp)
        else:
            learner.MLI.interpolate(self.p, self.y, learner.data_y)

        SH.s[:n_y].copy_(self.y.T)
        SH.s[n_y:].fill_(0)
        SH.Q.fill_(0)
        SH.forward_query()
        r = SH.s[:n_p].T  # (n_c, n_p)
        diff = r[:n] - r[n : 2 * n]
        dist = diff.norm(dim=1).clamp(min=1e-12)
        gap = (self.kappa * (self.p[:n] - self.p[n : 2 * n]).norm(dim=1) - dist).relu()
        self.loss = (self.amp * (gap**2).sum() / n).item()
        # n_c d loss / d r(p_c + d); d loss / d r(p_c - d) is its negative
        g = (-2 * self.amp * learner.n_c / n) * (gap / dist)[:, None] * diff
        SH.b.fill_(0)
        SH.B.fill_(0)
        SH.b[:n_p, :n].copy_(g.T)
        SH.b[:n_p, n : 2 * n].copy_(-g.T)
        sga, SH.core.sga = SH.core.sga, 0.0
        SH.backward_query()
        SH.core.sga = sga
        self.frac_active = (gap > 0).float().mean().item()
        if update_state:
            self.state = f"<{self.frac_active:.4f}>"
        learner.time_monitor.stop(self.name, torch_sync=True)
