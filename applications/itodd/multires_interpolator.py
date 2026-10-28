"""Multilinear interpolation on a grid with a separate resolution per dimension.

Generalization of twincher.interpolator.TorchMultilinearInterpolator: n_g is either one
number of grid points for all dimensions (then identical to the package's class) or a
sequence with one number per dimension, e.g. (16, 16, 32, 16). The grid spans [-1, 1] in
every dimension; the flattened order is that of grid_points(n_g) (itertools.product,
the last dimension fastest), as for Learner.get_p_request.

With a Learner: set_data with the grid data, then after learner.init() replace the
learner's interpolator: learner.MLI = TorchMultilinearInterpolator(n_p, n_g, device).
"""

from itertools import product

import numpy as np
import torch


def grid_points(n_g, n_p=None):
    """p (n_d, n_p) of the grid in the flattened order of the interpolator."""
    n_g = [n_g] * n_p if np.isscalar(n_g) else list(n_g)
    axes = [np.linspace(-1, 1, n) for n in n_g]
    return np.array(list(product(*axes)), dtype=np.float64)


class TorchMultilinearInterpolator:
    def __init__(self, n_p, n_g, device="cuda", dtype=torch.float32):
        if n_p < 1:
            raise ValueError("n_p must be >= 1")
        n_g = [int(n_g)] * n_p if np.isscalar(n_g) else [int(n) for n in n_g]
        if len(n_g) != n_p:
            raise ValueError(f"n_g needs {n_p} entries, got {len(n_g)}")
        if min(n_g) < 2:
            raise ValueError("every n_g must be >= 2")
        self.n_p, self.n_g = n_p, n_g
        self.n_d = int(np.prod(n_g))
        self.device = torch.device(device)
        self.dtype = dtype
        self.h = torch.tensor(
            [2.0 / (n - 1) for n in n_g], dtype=dtype, device=self.device
        )
        self.i_max = torch.tensor(
            [n - 2 for n in n_g], dtype=torch.long, device=self.device
        )
        strides = [int(np.prod(n_g[d + 1 :])) for d in range(n_p)]
        self.strides = torch.tensor(strides, dtype=torch.long, device=self.device)
        self.n_corners = 2**n_p
        bits = [
            [(k >> (n_p - 1 - d)) & 1 for d in range(n_p)]
            for k in range(self.n_corners)
        ]
        self.corner_bits = torch.tensor(bits, dtype=torch.long, device=self.device)
        offsets = [sum(s for b, s in zip(row, strides) if b) for row in bits]
        self.corner_offsets = torch.tensor(
            offsets, dtype=torch.long, device=self.device
        )

    def interpolate(self, p, y, data_y):
        """y (batch, ...) = multilinear interpolation at p (batch, n_p) of data_y
        (n_d, ...) on the grid; y is filled in place and returned."""
        if p.ndim == 1:
            p = p.unsqueeze(0)
        if p.shape[1] != self.n_p:
            raise ValueError(f"p must have shape (batch_size, {self.n_p})")
        if data_y.shape[0] != self.n_d:
            raise ValueError(
                f"data_y.shape[0] must be {self.n_d}, got {data_y.shape[0]}"
            )
        if y.shape[0] != p.shape[0] or y.shape[1:] != data_y.shape[1:]:
            raise ValueError(
                f"shapes of y {tuple(y.shape)} and data_y {tuple(data_y.shape)}"
            )
        x = (p.to(self.dtype) + 1.0) / self.h  # in [0, n_g - 1]
        i0 = torch.minimum(x.long().clamp(min=0), self.i_max)  # x >= 0: long() floors
        t = x - i0
        base = (i0 * self.strides).sum(dim=1)
        lower, upper = 1.0 - t, t
        weights = torch.where(
            self.corner_bits[None] == 0, lower[:, None], upper[:, None]
        )
        weights = weights.prod(dim=2)  # (batch, n_corners)
        shape = (len(p),) + (1,) * (y.ndim - 1)
        y.zero_()
        for k in range(self.n_corners):
            y += weights[:, k].reshape(shape) * data_y[base + self.corner_offsets[k]]
        return y
