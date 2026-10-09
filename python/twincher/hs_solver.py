# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

from pathlib import Path
import numpy as np
from .shuttle import Shuttle, load_metadata
from .tw_util import validate_parameters, copy_to, select_device
import torch

class HSSolver: # solver via Gauss-Newton descent in r space for hs architecture
    def __init__(self,
        file_name,
        n_c, # number of cases to be processed in parallel
        device="auto",
        dp_cap = 1.0, # maximal step size in p space
        epsilon = 1e-8, # regularizing constant (used during inversion)
        damper = {}, # controller for reducing/increasing steps based on loss changes
    ):
        device = select_device(device)
        if device == "cuda":
            data_type = "torch"
            dtype = torch.float32
        else:
            data_type = "torch_cpu"
            dtype = torch.float64
        
        self.file_name = file_name
        self.n_c = n_c
        self.tw_meta = load_metadata(file_name)
        if self.tw_meta["twincher_type"] != "hs":
            raise TypeError(f"{file_name} is not of type 'hs'")
        self.n_p = self.tw_meta["n_p"]
        self.n_y = self.tw_meta["n_y"]
        self.SH = Shuttle(data_type=data_type)
        self.SH.load(file_name=file_name, n_c=n_c, n_p=self.n_p, n_m=1)
        self.n_s = self.SH.n_s
        n_p, n_y, n_s = self.n_p, self.n_y, self.n_s
        self.dp_cap = dp_cap
        self.epsilon = epsilon

        self.damper, _ = validate_parameters(damper, {
            "down_factor": (0.0, 1.0, 0.5), 
            "up_factor": (1.0, 2.0, 1.1), 
            "dp_min": (0, None, dp_cap*1e-3), 
        }, raise_err=True)

        self.p = torch.empty((n_c, n_p), dtype=dtype, device=device)

        self.dy_dp = torch.empty((n_c, n_y, n_p), dtype=dtype, device=device)
        self.dp = torch.empty((n_c, n_p), dtype=dtype, device=device)
        self.R = torch.empty((n_c, n_p), dtype=dtype, device=device) # residual r_true - r_current
        self.J = torch.empty((n_c, n_p, n_p), dtype=dtype, device=device)
        self.D = torch.empty((n_c, n_p, n_p), dtype=dtype, device=device)
        self.D_reg = torch.empty((n_c, n_p, n_p), dtype=dtype, device=device)
        self.D_reg.fill_(0)
        self.D_reg[:, :, :].diagonal(dim1=1, dim2=2).fill_(epsilon) # regularization for D inversion
        self.inv_D = torch.empty((n_c, n_p, n_p), dtype=dtype, device=device)
        self.y = torch.empty((n_c, n_y), dtype=dtype, device=device)
        self.y_true = torch.empty((n_c, n_y), dtype=dtype, device=device)
        self.r_true = torch.empty((n_c, n_p), dtype=dtype, device=device)
        self.dp_size = torch.empty((n_c,), dtype=dtype, device=device)
        self.p_prev = torch.empty((n_c, n_p), dtype=dtype, device=device)

        if damper is not None:
            self.R_norm = torch.empty((n_c,), dtype=dtype, device=device)
            self.R_norm_prev = torch.empty((n_c,), dtype=dtype, device=device)

        self.state = "initiated"

    def start_solution(self, y_true, p_start = None):
        n_p, n_y, n_c = self.n_p, self.n_y, self.n_c
        copy_to(self.y_true[:, :], y_true[:, :])

        if p_start is None:
            self.p[:, :].fill_(0)
        else:
            copy_to(self.p, p_start)

        self.SH.s[:n_y, :].copy_(self.y_true[:, :].permute(1, 0))
        self.SH.s[n_y:, :].fill_(0)
        self.SH.forward_inference()
        self.r_true[:, :].copy_(self.SH.s[:n_p, :].permute(1, 0))

        self.dp_size[:] = self.dp_cap

        self.state = "start"

    def step(self, y, dy_dp, i_detail = None):
        if self.state == "initiated":
            raise RuntimeError("step is called before start_solution()")

        n_p, n_y, n_c = self.n_p, self.n_y, self.n_c

        if i_detail is not None: print("\n")

        copy_to(self.y[:, :], y[:, :])
        copy_to(self.dy_dp[:, :, :], dy_dp)
        self.R[:, :].copy_(self.r_true[:, :])

        if i_detail is not None:
            print(f"dp_size={self.dp_size[i_detail]}, r_goal=", self.R[i_detail, :].cpu().numpy(), end = "")

        self.SH.s[:n_y, :].permute(1, 0).copy_(self.y[:, :])
        self.SH.s[n_y:, :].fill_(0)
        self.SH.forward_inference()

        if i_detail is not None:
            print(", r_track=", self.SH.s[:n_p, i_detail].cpu().numpy(), end = "")

        self.R[:, :] -= self.SH.s[:n_p, :].permute(1, 0)

        if self.damper is not None: 
            torch.linalg.vector_norm(self.R[:, :], out=self.R_norm[:], dim=1)
            if self.state != "start":
                self.dp_size[:] = torch.where(
                    self.R_norm[:] > 0.9*self.R_norm_prev[:],
                    self.dp_size[:] * self.damper["down_factor"],
                    self.dp_size[:] * self.damper["up_factor"],
                )
                self.dp_size.clamp_min_(self.damper["dp_min"])
                self.dp_size.clamp_max_(self.dp_cap)

        if i_detail is not None:
            print(", diff=", torch.linalg.vector_norm(self.R[i_detail, :], dim=0))

        self.SH.s[:n_y, :].permute(1, 0).copy_(self.y[:, :])
        self.SH.s[n_y:, :].fill_(0)
        self.SH.compute_V()

        # compute D
        torch.matmul(
            self.SH.V[:, :n_y, :].permute(2, 0, 1), 
            self.dy_dp[:, :, :],
            out=self.J[:, :, :]
        )

        if i_detail is not None:
            J = self.J[i_detail, :, :].cpu().numpy()
            eigenvalues, eigenvectors = np.linalg.eig(J.transpose(1, 0) @ J)
            print("Eigenvalues:", eigenvalues, f", det = {np.linalg.det(J)}")
            print("Eigenvectors:\n", eigenvectors)
            print("R=", self.R[i_detail, :])

        torch.matmul(self.J.permute(0, 2, 1), self.J, out=self.D)

        # compute inv_D
        self.inv_D.copy_(self.D)
        self.inv_D[...] += self.D_reg[...]
        if n_p > 1: # in-place computations may not work for 1x1 matrices
            torch.linalg.inv(self.inv_D, out=self.inv_D)
        else:
            self.inv_D.copy_(torch.linalg.inv(self.inv_D))

        # compute dp
        torch.matmul(
            self.J.permute(0, 2, 1),
            self.R[..., None],
            out=self.dp[..., None]
        )
        self.R.copy_(self.dp)
        if i_detail is not None:
            print("J@R=", self.R[i_detail, :])
            inv_D = self.inv_D[i_detail, :, :].cpu().numpy()
            eigenvalues, eigenvectors = np.linalg.eig(inv_D)
            print("inv_D=", inv_D)
            print("Eigenvalues:", eigenvalues)
            print("Eigenvectors:\n", eigenvectors)
            print("R=", self.R[i_detail, :])
            print("self.dp_size[:, None]=", self.dp_size[i_detail, None])

        torch.matmul(
            self.inv_D,
            self.R[..., None],
            out=self.dp[..., None]
        )

        self.dp[:, :] *= self.dp_size[:, None]

        if i_detail is not None:
            print("dp=", self.dp[i_detail, :])

        if self.damper is not None: 
            if self.state != "start":
                if i_detail is not None:
                    print(f"R_norm_prev={self.R_norm_prev[i_detail]}, R_norm={self.R_norm[i_detail]}")

                self.dp[:, :] = torch.where(
                    self.R_norm[:, None] > self.R_norm_prev[:, None],
                    (self.p_prev[:, :] - self.p[:, :]) * (1 - self.damper["down_factor"]), # rejection
                    self.dp[:, :]
                )
                self.p_prev[:, :] = torch.where(
                    self.R_norm[:, None] > self.R_norm_prev[:, None],
                    self.p_prev[:, :],
                    self.p[:, :]
                )
                self.R_norm_prev[:] = torch.where(
                    self.R_norm[:] > self.R_norm_prev[:],
                    self.R_norm_prev[:],
                    self.R_norm[:]
                )
            else:
                self.p_prev[:, :] = self.p
                self.R_norm_prev[:] = self.R_norm[:]
            

        # apply step, clip result
        self.p[:] += self.dp
        self.p.clamp_min_(-1)
        self.p.clamp_max_(1)

        self.state = "solve"