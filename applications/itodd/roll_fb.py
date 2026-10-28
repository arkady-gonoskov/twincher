"""Roll-aware data of the Fourier-Bessel stencil.

Rolling the stencil camera about its optical axis rotates the depth map about the
principal point, and with it the heights about their centre of mass; the Fourier-Bessel
coefficients then change only by a phase: f_nm(psi) = f_nm(psi_0) exp(-i m sigma (psi -
psi_0)), sigma = +-1 by the conventions (determined against rendered views). So a grid
over the other components of p at the roll centre (p_roll = 0) suffices: RollFB stores
the complex coefficients f, their derivatives df/dp_k (central differences with the
steps of the stencil) and the argument x_D of the depth component (independent of the
roll), interpolates them multilinearly, applies the phase of the roll exactly, the
damping factors and the ISRU, and gives dy/dp_roll analytically:
    dx/dp_roll = factor * (Re, Im) of (-i m sigma dpsi/dp_roll f).
Used as the stencil of a Learner (compute(p, y, dy_dp)), like train_chart.SamplePool.

st: the chart stencil with fb options, full (roll = component 3) or a ReducedStencil
whose dims contain 3.
"""

import time
from pathlib import Path

import torch

from multires_interpolator import TorchMultilinearInterpolator, grid_points
from train_chart import ReducedStencil


class RollFB:
    def __init__(self, st, n_g, file=None, batch=256):
        reduced = isinstance(st, ReducedStencil)
        self.base = st.st if reduced else st
        if self.base.fb is None:
            raise ValueError("RollFB needs a stencil with the fb option")
        dims = st.dims if reduced else list(range(st.n_p))
        if 3 not in dims:
            raise ValueError("the stencil has no roll component")
        self.st, self.n_p, self.n_y, self.device = st, st.n_p, st.n_y, st.device
        self.name = st.name + "_rollfb"
        self.roll = dims.index(3)
        self.others = [k for k in range(self.n_p) if k != self.roll]
        self.scale = st.scale if reduced else 1.0
        lo, hi = self.base._lo[3].item(), self.base._hi[3].item()
        self.dpsi = self.scale * 0.5 * (hi - lo)  # dpsi / dp_roll
        self.m = self.base._fbt.m.to(torch.float32)
        self.n_g = list(n_g)
        if len(self.n_g) != self.n_p - 1:
            raise ValueError(f"n_g needs {self.n_p - 1} entries")
        self.mli = TorchMultilinearInterpolator(self.n_p - 1, self.n_g, self.device)
        data = None
        if file is not None and Path(file).exists():
            data = torch.load(file)
            if data["n_g"] != self.n_g:
                data = None
        if data is None:
            t0 = time.perf_counter()
            data = self._grid_data(batch)
            print(f"roll-aware grid data {self.n_g} computed in "
                  f"{time.perf_counter() - t0:.0f} s")  # fmt: skip
            if file is not None:
                torch.save(data, file)
        self.F = data["F"].to(self.device)  # (nodes, 2 M): Re, Im
        self.dF = data["dF"].to(self.device)  # (nodes, 2 M, n_p - 1)
        self.xd = None if data["xd"] is None else data["xd"].to(self.device)
        self.dxd = None if data["dxd"] is None else data["dxd"].to(self.device)
        self.sigma = data["sigma"]

    def coeffs(self, p):
        """(f complex (n, M), x_D (n,) or None) of the views p (n, n_p) (rendered)."""
        q = self.st.full(p) if isinstance(self.st, ReducedStencil) else p
        return self.base.fb_coeffs(q.to(self.device, torch.float64))

    def _grid_data(self, batch):
        nodes = torch.as_tensor(grid_points(self.n_g))  # (N, n_p - 1)
        N, k_o = len(nodes), len(self.others)
        h = torch.tensor([self.base.dp[self._full(k)].item() / self.scale
                          for k in self.others], dtype=torch.float64)  # fmt: skip
        M = len(self.m)
        F = torch.empty((N, 2 * M))
        dF = torch.empty((N, 2 * M, k_o))
        xd = None if self.base.fb["depth"] is None else torch.empty(N)
        dxd = None if xd is None else torch.empty((N, k_o))
        for i in range(0, N, batch):
            q = nodes[i : i + batch]
            p = torch.zeros((len(q), self.n_p), dtype=torch.float64)
            p[:, self.others] = q
            f, x = self.coeffs(p)
            F[i : i + len(q)] = torch.cat([f.real, f.imag], dim=1).cpu()
            if xd is not None:
                xd[i : i + len(q)] = x.cpu()
            for j, k in enumerate(self.others):
                e = torch.zeros(self.n_p, dtype=torch.float64)
                e[k] = h[j]
                fp, xp = self.coeffs(p + e)
                fm, xm = self.coeffs(p - e)
                d = (fp - fm) / (2 * h[j])
                dF[i : i + len(q), :, j] = torch.cat([d.real, d.imag], dim=1).cpu()
                if xd is not None:
                    dxd[i : i + len(q), j] = ((xp - xm) / (2 * h[j])).cpu()
        sigma = self._sign(nodes)
        return {"n_g": self.n_g, "F": F, "dF": dF, "xd": xd, "dxd": dxd,
                "sigma": sigma}  # fmt: skip

    def _full(self, k):
        return self.st.dims[k] if isinstance(self.st, ReducedStencil) else k

    def _sign(self, nodes, n=64, roll=0.5):
        """sigma: the rotation of f_nm with the roll, from rendered views."""
        g = torch.Generator().manual_seed(0)
        q = nodes[torch.randint(len(nodes), (n,), generator=g)]
        p = torch.zeros((n, self.n_p), dtype=torch.float64)
        p[:, self.others] = q
        f0, _ = self.coeffs(p)
        p[:, self.roll] = roll
        f1, _ = self.coeffs(p)
        phase = self.m.to(torch.float64) * self.dpsi * roll
        errs = {}
        for sigma in (1, -1):
            rot = f0 * torch.exp(-1j * sigma * phase.to(f0.device))
            errs[sigma] = ((rot - f1).abs().sum() / f1.abs().sum()).item()
        sigma = min(errs, key=errs.get)
        print(f"roll phase sign {sigma:+d} (relative errors {errs})")
        return sigma

    def compute(self, p, y, dy_dp):
        """y (n, n_y) and dy/dp (n, n_y, n_p) for p (n, n_p)."""
        p = p.to(self.device, torch.float32)
        n, M, k_o = len(p), len(self.m), len(self.others)
        q = p[:, self.others].contiguous()
        F = torch.empty((n, 2 * M), device=self.device)
        dF = torch.empty((n, 2 * M, k_o), device=self.device)
        self.mli.interpolate(q, F, self.F)
        self.mli.interpolate(q, dF, self.dF)
        phase = self.sigma * self.dpsi * p[:, self.roll : self.roll + 1] * self.m
        c, s = phase.cos(), phase.sin()
        Fr, Fi = F[:, :M], F[:, M:]
        fr, fi = Fr * c + Fi * s, Fi * c - Fr * s  # f exp(-i sigma phase)
        dr = dF[:, :M] * c[..., None] + dF[:, M:] * s[..., None]
        di = dF[:, M:] * c[..., None] - dF[:, :M] * s[..., None]
        w = self.sigma * self.dpsi * self.m  # d/dp_roll: -i w f
        b = self.base
        mode, imag, factor = b._fb_mode, b._fb_imag, b._fb_factor.to(torch.float32)
        x = torch.where(imag, fi[:, mode], fr[:, mode]) * factor
        dx = torch.empty((n, len(mode), self.n_p), device=self.device)
        dx[:, :, self.others] = (
            torch.where(imag[:, None], di[:, mode], dr[:, mode]) * factor[:, None]
        )
        roll_r, roll_i = w * fi, -w * fr  # (-i w f): Re, Im
        dx[:, :, self.roll] = (
            torch.where(imag, roll_i[:, mode], roll_r[:, mode]) * factor
        )
        if self.xd is not None:
            xd = torch.empty((n,), device=self.device)
            dxd = torch.empty((n, k_o), device=self.device)
            self.mli.interpolate(q, xd, self.xd)
            self.mli.interpolate(q, dxd, self.dxd)
            x = torch.cat([xd[:, None], x], dim=1)
            dxd_full = torch.zeros((n, 1, self.n_p), device=self.device)
            dxd_full[:, 0, self.others] = dxd
            dx = torch.cat([dxd_full, dx], dim=1)
        y.copy_(x / torch.sqrt(1 + x * x))
        dy_dp.copy_(dx * (1 + x * x).pow(-1.5)[..., None])
