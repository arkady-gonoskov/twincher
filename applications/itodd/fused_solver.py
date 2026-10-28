"""Solver for depth maps with unknown pixels (structured-light shadows, occlusions).

The hs solver iterates p towards a fixed target r_true = r(y_observed). With unknown
pixels, the target is instead "chased": at each step, the unknown pixels of the observed
depth map are filled with the depth map rendered at the current p (map_fused), and the
step goes from r_curr = r(y(p)) towards r_fused = r(y(map_fused)). At a fixed point the
rendering agrees with all available pixels, and r_fused = r_curr; with all pixels
available, this is the hs solver. The residual F(p) = r_fused - r_curr is a function of
p alone, so that the damper of the hs solver (and dp_cap) apply unchanged.

Step (default, newton=True): Newton's for F(p) = 0, dp = (J_curr - J_fused)^-1 F, where
J_fused = dr_fused/dp accounts for the change of the filled pixels with p (J = V(y) dy/dp,
V = dr/dy; the derivatives dy/dp of the rendered and of the fused maps by finite
differences from the same renderings). With newton=False: dp = J_curr^-1 F, which treats
r_fused as fixed (more stalls of the damped iteration, see test_unknown.py).
"""

import torch
from twincher.hs_solver import HSSolver
from twincher.tw_util import copy_to


class FusedHSSolver(HSSolver):
    """HSSolver whose target can be replaced between steps (set_target) without
    restarting the solution: p, step sizes and the state of the damper are kept."""

    def set_target(self, y_target):
        n_p, n_y = self.n_p, self.n_y
        copy_to(self.y_true, y_target)
        self.SH.s[:n_y, :].copy_(self.y_true.permute(1, 0))
        self.SH.s[n_y:, :].fill_(0)
        self.SH.forward_inference()
        self.r_true.copy_(self.SH.s[:n_p, :].permute(1, 0))

    def _V(self, y):
        """dr/dy (n_c, n_p, n_y) at y (n_c, n_y)."""
        n_y = self.n_y
        self.SH.s[:n_y, :].copy_(y.permute(1, 0))
        self.SH.s[n_y:, :].fill_(0)
        self.SH.compute_V()
        return self.SH.V[:, :n_y, :].permute(2, 0, 1).clone()

    def newton_dy_dp(self, y_curr, dy_curr, y_fused, dy_fused):
        """dy/dp to pass to step(y_curr, ...) so that its Jacobian V(y_curr) dy/dp is
        the Jacobian of F: J_curr - J_fused = V(y_curr) dy_curr - V(y_fused) dy_fused."""
        V_c = self._V(y_curr)
        J_f = self._V(y_fused) @ dy_fused
        G = V_c @ V_c.transpose(1, 2)  # (n_c, n_p, n_p), V_c has full row rank
        V_c_pinv = V_c.transpose(1, 2) @ torch.linalg.inv(G)
        return dy_curr - V_c_pinv @ J_f


def fused_values(stencil, p, depth_avail, unknown, fused_derivative):
    """y and dy/dp of the rendered maps (y_curr) and of the fused maps (y_fused) at p.

    The maps rendered at p and at p +- dp_k e_k (finite differences, as in
    DepthStencil.compute) are used as they are and with the available pixels of
    depth_avail; dy_fused/dp only if fused_derivative (else None)."""
    p = p.to(stencil.device, torch.float64)
    n_p = p.shape[1]
    steps = torch.diag(stencil.dp)
    points = (
        [p] + [p + steps[k] for k in range(n_p)] + [p - steps[k] for k in range(n_p)]
    )
    y_c, y_f = [], []
    for i, q in enumerate(points):
        z = stencil.depth_image(q)
        y_c.append(stencil.y_from_depth(z)[0])
        if i == 0 or fused_derivative:
            y_f.append(stencil.y_from_depth(torch.where(unknown, z, depth_avail))[0])
        del z

    def derivative(y):
        # float32 differences divided in float64, as in DepthStencil.compute
        diff = [
            (y[1 + k] - y[1 + n_p + k]).double() / (2 * stencil.dp[k])
            for k in range(n_p)
        ]
        return torch.stack(diff, dim=-1).float()

    dy_f = derivative(y_f) if fused_derivative else None
    return y_c[0], derivative(y_c), y_f[0], dy_f


def solve_fused(
    solver, stencil, depth_avail, n_iter, p_start=None, newton=True, callback=None
):
    """Solve for p from depth maps with unknown pixels, chasing the fused target.

    solver: FusedHSSolver with n_c cases; stencil: DepthStencil (forward model);
    depth_avail: (n_c, res, res) on the device of the stencil, Z in mm, 0 for
    background, NaN for unknown pixels; p_start: (n_c, n_p) or None (p = 0); newton:
    Newton step for the fused residual (see the module docstring); callback(iteration,
    solver) is called after each step.
    """
    unknown = torch.isnan(depth_avail)
    if p_start is None:
        solver.p.fill_(0)
    else:
        copy_to(solver.p, p_start)
    for it in range(n_iter):
        y_c, dy_c, y_f, dy_f = fused_values(
            stencil, solver.p, depth_avail, unknown, newton
        )
        if it == 0:
            solver.start_solution(y_f, p_start=solver.p.clone())
        else:
            solver.set_target(y_f)
        dy = solver.newton_dy_dp(y_c, dy_c, y_f, dy_f) if newton else dy_c
        solver.step(y=y_c, dy_dp=dy)
        if callback is not None:
            callback(it + 1, solver)
