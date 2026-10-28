"""Winding numbers of r = twincher(y(p)) around small circles in coordinate planes of p.

A circle p(t) = c + rho (cos t e_a + sin t e_b) maps to a closed curve r(t) in R^n_p.
Its winding number is taken in the plane of the curve (the two leading principal axes
u1, u2 of r(t) - mean r), with the orientation of that plane given by the images of
e_a, e_b: sign of u1^T W u2, W = mean over t of (J e_a)(J e_b)^T - (J e_b)(J e_a)^T. An
embedding gives +1 for every small circle; a branch point inside the circle (r winds
twice around it, like z -> z^2; det J >= 0 around it) gives 2, a fold 0 or -1. For
n_p = 2 the result equals the winding number of r itself.

LoopWindings(st, wiring, n_loops, n_t) holds a shuttle for n_loops circles of n_t points
(wiring: a .twc file whose layers and pairs the shuttle takes, or (n_s, n_l) for the
default wiring); __call__(a, centres, planes, rho) returns the winding numbers
(n_loops,), with y and dy/dp of the stencil (exact).
"""

import math

import torch
import twincher


def shuttle_for(wiring, n_c, n_p):
    """A GPU shuttle for n_c cases with the layers and pairs of the .twc file wiring, or
    the default wiring for wiring = (n_s, n_l)."""
    if isinstance(wiring, tuple):
        n_s, n_l = wiring
        return twincher.Shuttle(data_type="torch", n_s=n_s, n_l=n_l, n_c=n_c, n_p=n_p,
                                n_m=n_p)  # fmt: skip
    sh = twincher.Shuttle(data_type="torch")
    sh.load(str(wiring), n_c=n_c, n_p=n_p, n_m=n_p)
    return sh


class LoopWindings:
    def __init__(self, st, wiring, n_loops, n_t=16):
        self.st, self.n_loops, self.n_t = st, n_loops, n_t
        self.n_p, self.n_y = st.n_p, st.n_y
        n = n_loops * n_t
        self.SH = shuttle_for(wiring, n, st.n_p)
        self.y = torch.empty((n, st.n_y), device=st.device)
        self.dy_dp = torch.empty((n, st.n_y, st.n_p), device=st.device)
        t = torch.arange(n_t, dtype=torch.float64) * (2 * math.pi / n_t)
        self.cos, self.sin = t.cos(), t.sin()

    def __call__(self, a, centres, planes, rho):
        """Winding numbers (n_loops,) of the circles of radius rho (scalar or (n_loops,))
        centred at centres (n_loops, n_p) in the coordinate planes planes (n_loops, 2)."""
        n_t, n_p, n_y, SH = self.n_t, self.n_p, self.n_y, self.SH
        rho = torch.as_tensor(rho, dtype=torch.float64).expand(len(centres))
        p = centres.double()[:, None, :].repeat(1, n_t, 1)  # (loops, t, n_p)
        rows = torch.arange(len(centres))
        p[rows, :, planes[:, 0]] += rho[:, None] * self.cos
        p[rows, :, planes[:, 1]] += rho[:, None] * self.sin
        p = p.reshape(-1, n_p)
        for i in range(0, len(p), 512):
            self.st.compute(
                p[i : i + 512], self.y[i : i + 512], self.dy_dp[i : i + 512]
            )
        SH.a.copy_(a)
        SH.tw_update()
        SH.s[:n_y].copy_(self.y.T)
        SH.s[n_y:].fill_(0)
        SH.Q.fill_(0)
        SH.Q[:n_y].copy_(self.dy_dp.permute(1, 2, 0))
        SH.forward_query()
        r = SH.s[:n_p].T.double().reshape(-1, n_t, n_p)
        J = SH.Q[:n_p].permute(2, 0, 1).double().reshape(-1, n_t, n_p, n_p)  # dr/dp
        idx = planes.to(J.device)[:, None, None, :].expand(-1, n_t, n_p, 2)
        A, B = torch.gather(J, 3, idx).unbind(dim=3)  # (loops, t, n_p): J e_a, J e_b
        W = (
            A[..., :, None] * B[..., None, :] - B[..., :, None] * A[..., None, :]
        ).mean(1)
        d = r - r.mean(dim=1, keepdim=True)
        u = torch.linalg.svd(d, full_matrices=False).Vh[:, :2]  # (loops, 2, n_p)
        xy = torch.einsum("ltp,lkp->ltk", d, u)
        angle = torch.atan2(xy[..., 1], xy[..., 0])
        step = torch.diff(torch.cat([angle, angle[:, :1]], dim=1), dim=1)
        step = (step + math.pi) % (2 * math.pi) - math.pi
        w = step.sum(dim=1) / (2 * math.pi)
        sign = torch.sign(torch.einsum("lp,lpq,lq->l", u[:, 0], W, u[:, 1]))
        return (w * sign).round().cpu()
