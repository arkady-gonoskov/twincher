"""Checks of DepthStencil: geometry, conventions, view_from_pose, speed.

Usage:
    python check_stencil.py
"""

import math
import time

import numpy as np
import torch

from depth_stencil import DepthStencil, view_from_pose
from itodd_data import model

MODEL = model(12)


def check_geometry(st):
    c = st.res // 2
    for view, z_expected in [
        ((700.0, 0.0, 0.0), 700 - 20),  # side view: radius 20
        ((700.0, math.pi / 2, 0.0), 700 - 50),  # top view (outside the chart): z = 50
    ]:
        p = st.p_from_view(torch.tensor([view], dtype=torch.float64, device=st.device))
        z = st.depth_image(p)[0]
        z_c = z[c - 1 : c + 1, c - 1 : c + 1].mean().item()
        print(f"view {np.round(view, 3)}: central Z {z_c:.3f}, expected {z_expected}")
    # +Z end (the nearer one for a camera above the XY plane): up for psi = 0,
    # turned counterclockwise (to the left) for psi > 0
    for psi, where in [(0.0, "top"), (1.0, "upper left")]:
        p = st.p_from_view(
            torch.tensor([[700.0, 0.5, psi]], dtype=torch.float64, device=st.device)
        )
        z = st.depth_image(p)[0]
        rows, cols = torch.nonzero(z > 0, as_tuple=True)
        near = z[z > 0] < z[z > 0].median()
        print(
            f"psi {psi:.1f}: nearer half at (row, col) = ({rows[near].float().mean():.0f},"
            f" {cols[near].float().mean():.0f}), centre {c}, expected: {where}"
        )


def check_pose_round_trip(st, n=200):
    """view -> BOP pose (object on the optical axis) -> view_from_pose."""
    g = torch.Generator().manual_seed(1)
    p = 2 * torch.rand((n, st.n_p), generator=g, dtype=torch.float64) - 1
    C, E = st._frames(p)
    v = st.view(p.to(st.device)).cpu().numpy()
    err = 0.0
    for c in range(n):
        Ec, Cc = E[c].double().cpu().numpy(), C[c].double().cpu().numpy()
        D, e, psi, _ = view_from_pose(Ec.T, -Ec.T @ Cc)  # R_m2c, t_m2c
        err = max(err, abs(D - v[c, 0]) / v[c, 0], abs(e - v[c, 1]), abs(psi - v[c, 2]))
    print(f"view -> pose -> view: largest error {err:.1e} (float32 frames)")


def check_speed(n_rep=3):
    for res in (256, 512, 1024):
        st = DepthStencil(MODEL, resolution=res)
        for n_c in (1, 1024):
            p = 2 * torch.rand((n_c, st.n_p), device="cuda") - 1
            y = torch.empty((n_c, st.n_y), device="cuda")
            dy = torch.empty((n_c, st.n_y, st.n_p), device="cuda")
            st.compute(p, y, dy)
            torch.cuda.synchronize()
            n = 20 * n_rep if n_c == 1 else n_rep
            t0 = time.perf_counter()
            for _ in range(n):
                st.compute(p, y, dy)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / n
            print(
                f"compute() with dy/dp, resolution {res}, n_c {n_c:4d}: "
                f"{dt * 1e3:7.1f} ms per call, {dt / n_c * 1e3:.3f} ms per case"
            )


if __name__ == "__main__":
    st = DepthStencil(MODEL)
    print(
        f"bounding radius {st.radius:.2f} mm, square half-side {st.half_width:.5f}, "
        f"pixel {st.pix * 2992:.2f} ITODD camera pixels, kernel K {st.kernel_size} px"
    )
    print(
        f"Z_lo {st.z_lo:.1f} mm -> y = -{st.s_amp}, Z_bg {st.z_bg:.1f} mm -> y = {st.s_amp}"
    )
    check_geometry(st)
    check_pose_round_trip(st)
    check_speed()
