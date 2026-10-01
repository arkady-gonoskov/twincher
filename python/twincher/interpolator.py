# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

import torch

class TorchMultilinearInterpolator:
    """
    Multilinear interpolation on a uniform tensor-product grid
    spanning [-1, 1] in every parameter dimension.

    Parameters
    ----------
    n_p : int
        Number of parameter dimensions.

    n_g : int
        Number of grid points per dimension.

    device : torch.device or str, optional
        Device on which the interpolation is performed.

    dtype : torch.dtype, optional
        Floating-point dtype used for interpolation.

    Notes
    -----
    The flattened grid ordering must correspond to

        itertools.product(np.linspace(-1, 1, n_g), repeat=n_p)

    The first dimension of y_data is the flattened grid dimension.
    All remaining dimensions are arbitrary.

    Examples
    --------
    y_data.shape = (n_d,)
    y.shape      = (batch_size,)

    y_data.shape = (n_d, n_y)
    y.shape      = (batch_size, n_y)

    y_data.shape = (n_d, n1, n2)
    y.shape      = (batch_size, n1, n2)
    """

    def __init__(
        self,
        n_p,
        n_g,
        device="cuda",
        dtype=torch.float32,
    ):
        self.n_p = n_p
        self.n_g = n_g
        self.n_d = n_g ** n_p

        if n_p < 1:
            raise ValueError("n_p must be >= 1")

        if n_g < 2:
            raise ValueError("n_g must be >= 2")

        self.device = torch.device(device)
        self.dtype = dtype

        # Grid spacing
        self.h = 2.0 / (n_g - 1)

        # ---------------------------------------------------------
        # Flattened-grid strides
        #
        # These are CPU-calculated and then stored on the desired
        # device. This avoids integer matrix multiplication on CUDA.
        # ---------------------------------------------------------

        strides = [
            n_g ** (n_p - 1 - d)
            for d in range(n_p)
        ]

        self.strides = torch.tensor(
            strides,
            dtype=torch.long,
            device=self.device,
        )

        # ---------------------------------------------------------
        # Corner information
        # ---------------------------------------------------------

        self.n_corners = 2 ** n_p

        corner_bits = [
            [
                (k >> (n_p - 1 - d)) & 1
                for d in range(n_p)
            ]
            for k in range(self.n_corners)
        ]

        self.corner_bits = torch.tensor(
            corner_bits,
            dtype=torch.long,
            device=self.device,
        )

        # Flattened offset from lower cell corner to each corner.
        #
        # Calculated on CPU to avoid Long CUDA matmul.
        #
        corner_offsets = []

        for k in range(self.n_corners):
            offset = 0

            for d in range(n_p):
                if corner_bits[k][d]:
                    offset += strides[d]

            corner_offsets.append(offset)

        self.corner_offsets = torch.tensor(
            corner_offsets,
            dtype=torch.long,
            device=self.device,
        )

        # ---------------------------------------------------------
        # Reusable workspace
        # ---------------------------------------------------------

        self._batch_capacity = 0

        self._x = None
        self._i0 = None
        self._t = None
        self._base_index = None
        self._weights = None

    def _allocate_workspace(self, batch_size):

        if batch_size <= self._batch_capacity:
            return

        self._batch_capacity = batch_size

        self._x = torch.empty(
            (batch_size, self.n_p),
            dtype=self.dtype,
            device=self.device,
        )

        self._i0 = torch.empty(
            (batch_size, self.n_p),
            dtype=torch.long,
            device=self.device,
        )

        self._t = torch.empty(
            (batch_size, self.n_p),
            dtype=self.dtype,
            device=self.device,
        )

        self._base_index = torch.empty(
            batch_size,
            dtype=torch.long,
            device=self.device,
        )

        self._weights = torch.empty(
            (batch_size, self.n_corners),
            dtype=self.dtype,
            device=self.device,
        )

    def interpolate(self, p, y, data_y):
        """
        Batch multilinear interpolation.

        Parameters
        ----------
        p : torch.Tensor
            Shape (batch_size, n_p).

        y : torch.Tensor
            Preallocated output, shape (batch_size, ...).

        y_data : torch.Tensor
            Grid data, shape (n_g**n_p, ...).

        Returns
        -------
        y : torch.Tensor
            The same output tensor supplied to the function.
        """

        if p.ndim == 1:
            p = p.unsqueeze(0)

        batch_size = p.shape[0]

        # ---------------------------------------------------------
        # Validate dimensions
        # ---------------------------------------------------------

        if p.shape[1] != self.n_p:
            raise ValueError(
                f"p must have shape (batch_size, {self.n_p})"
            )

        if data_y.shape[0] != self.n_d:
            raise ValueError(
                f"y_data.shape[0] must be {self.n_d}, "
                f"got {data_y.shape[0]}"
            )

        if y.shape[0] != batch_size:
            raise ValueError(
                "y.shape[0] must equal p.shape[0]"
            )

        if y.shape[1:] != data_y.shape[1:]:
            raise ValueError(
                f"Trailing dimensions do not match: "
                f"{y.shape[1:]} != {data_y.shape[1:]}"
            )

        if p.device.type != self.device.type:
            raise ValueError(
                f"p is on {p.device}, "
                f"but interpolator is on {self.device}"
            )

        if y.device.type != self.device.type:
            raise ValueError(
                f"y is on {y.device}, "
                f"but interpolator is on {self.device}"
            )

        if data_y.device.type != self.device.type:
            raise ValueError(
                f"y_data is on {data_y.device}, "
                f"but interpolator is on {self.device}"
            )

        # ---------------------------------------------------------
        # Allocate/reuse workspace
        # ---------------------------------------------------------

        self._allocate_workspace(batch_size)

        x = self._x[:batch_size]
        i0 = self._i0[:batch_size]
        t = self._t[:batch_size]
        base_index = self._base_index[:batch_size]
        weights = self._weights[:batch_size]

        # ---------------------------------------------------------
        # Map [-1, 1] -> [0, n_g - 1]
        # ---------------------------------------------------------

        torch.sub(p, -1.0, out=x)
        x.div_(self.h)

        # Since x >= 0, integer conversion is equivalent to floor.
        i0[:] = x.to(torch.long)

        # p == +1 belongs to the final cell.
        i0.clamp_(max=self.n_g - 2)

        # Fractional position within cell
        torch.sub(x, i0, out=t)

        # ---------------------------------------------------------
        # Calculate flattened lower-corner index.
        # ---------------------------------------------------------

        base_index.zero_()

        for d in range(self.n_p):
            base_index += i0[:, d] * self.strides[d]

        # ---------------------------------------------------------
        # Calculate interpolation weights
        # ---------------------------------------------------------

        weights.fill_(1.0)

        for d in range(self.n_p):

            td = t[:, d]

            lower = 1.0 - td
            upper = td

            bits = self.corner_bits[:, d]

            weights *= torch.where(
                bits[None, :] == 0,
                lower[:, None],
                upper[:, None],
            )

        # ---------------------------------------------------------
        # Accumulate corner contributions
        # ---------------------------------------------------------

        y.zero_()

        # Shape for broadcasting weights over arbitrary
        # trailing dimensions.
        weight_shape = (
            batch_size,
        ) + (1,) * (y.ndim - 1)

        for k in range(self.n_corners):

            indices = (
                base_index
                + self.corner_offsets[k]
            )

            y += (
                weights[:, k].reshape(weight_shape)
                * data_y[indices]
            )

        return y
