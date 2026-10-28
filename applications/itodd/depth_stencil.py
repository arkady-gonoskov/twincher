"""Depth-map stencil: pyramid-kernel probes of a mesh model seen from a camera.

Stencil for twincher (`compute(p, y, dy_dp)`). The object is at the origin of its model
frame (BOP convention, mm). The camera looks at the origin from the distance D and the
direction given by the elevation e above the model's XY plane (e = pi/2 - polar angle)
and, for n_p = 4, the azimuth phi; it is rotated by the roll psi about its optical axis:

    p0 -> D in dist_range           p1 -> e in elevation_range
    p2 -> psi in roll_range         p3 -> phi in azimuth_range (optional)

(linear maps from [-1, 1]). For psi = 0, the projection of the model's +Z axis points up
in the image; positive psi rotates it counterclockwise. Without azimuth, the camera is in
the XZ plane (phi = 0), which suffices for objects with a continuous symmetry about Z.
The default ranges, +-(pi/2 - 0.3) for e and psi, form one chart for the cylinder: p = 0
is a side view with the cylinder vertical in the image; because of the flip symmetry,
(e, psi) ~ (-e, psi + pi), |psi| < pi/2 covers all roll angles.

Forward map y(p):
1. depth map Z, resolution x resolution pixels over a fixed square of normalized image
   coordinates (x/z, y/z) centred on the optical axis that contains the bounding sphere
   of the model (about the origin) at the smallest distance, times `margin`; Z is the
   depth along the optical axis (like BOP depth maps), 0 where a ray misses the model;
   everything outside the square is background;
2. centre of mass (x_S, y_S) of the mask Z > 0;
3. n_grid x n_grid probes centred at (x_S, y_S) with the spacing side/n_grid; the
   component y_j is the probe at grid position grid_index[j] = row*n_grid + col (rows
   from the top of the image); probe_order "centre" (default) sorts the probes by their
   distance from the centre of the grid (then by angle), so that the first components,
   which form the representation r of an untrained (identity) twincher, see the object;
   "rows" keeps the row-major order;
4. each probe is a kernel of half-size K pixels (default: the probe spacing), centred
   at the (sub-pixel) probe position c, with Z = Z_bg for misses; Z_bg = D_max + r
   (r: bounding radius) is the largest possible depth:
   - kernel "max" (default): the largest weighted height above the background,
         Z_probe = Z_bg + min_d [(Z(c + d) - Z_bg) (1 - |d|_inf / K)],   |d|_inf < K,
     continuous when the kernel slides off an edge and ignoring pixels without depth;
     on a smooth surface it is in effect the depth at the probe centre (one pixel);
   - kernel "tent": the weighted mean height above the background,
         Z_probe = Z_bg + sum_d w(d) (Z(c + d) - Z_bg) / K^2,
         w(d) = (1 - |d_x| / K)(1 - |d_y| / K),   |d_x|, |d_y| < K,
     whose weights sum to one over the grid for K = probe spacing (pixels without
     depth count as background);
5. y = s_amp*(2*(Z_probe - Z_lo)/(Z_bg - Z_lo) - 1) with Z_lo = D_min - r: all possible
   depths map to [-s_amp, s_amp], the background to +s_amp.

dy/dp by central finite differences with the step dp (scalar or one per component).
Ray casting on a BVH and the kernels by NVIDIA Warp (GPU or CPU), the rest by PyTorch.
"""

import math

import numpy as np
import torch
import warp as wp

wp.config.log_level = wp.LOG_WARNING  # no banner

# options of the Fourier-Bessel stencil (DepthStencil(fb={...})): n components, damping
# exp(-gamma (n_r / n_cr)^2) of the radial order n_r, scale C of y = ISRU(C f), disk
# radius (fraction of the half side of the square, about the centre of mass), background
# (None: heights above Z_bg; a number (mm): heights above the mean depth of the mask plus
# this offset, deeper surfaces count as background), polar sampling n_rho x n_theta;
# depth: None, or the scale C_D of a first component ISRU(C_D (Z_mean - Z_mid) / (Z_bg -
# Z_lo)) with the mean depth Z_mean of the mask and Z_mid = (Z_lo + Z_bg) / 2 (the
# distance, otherwise only in the scale of the image with a near background)
FB_DEFAULTS = {"n": 64, "gamma": 1.0, "n_cr": 3.0, "scale": 1.0, "radius": 1.0,
               "background": None, "n_rho": 128, "n_theta": 512, "depth": None}  # fmt: skip


def read_ply(file_name):
    """Vertices (n_v, 3) and triangles (n_f, 3) of a PLY file (ASCII or binary)."""
    with open(file_name, "rb") as f:
        header = []
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            header.append(line.split())
        body = f.read()
    fmt = next(words[1] for words in header if words[0] == "format")
    types = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
             "int": "i4", "uint": "u4", "float": "f4", "double": "f8"}  # fmt: skip
    elements = []  # [name, count, properties]
    for words in header:
        if words[0] == "element":
            elements.append([words[1], int(words[2]), []])
        elif words[0] == "property":
            elements[-1][2].append(words[1:])
    if fmt == "ascii":
        tokens = iter(body.split())
        data = {}
        for name, count, props in elements:
            rows = []
            for _ in range(count):
                row = []
                for prop in props:
                    if prop[0] == "list":
                        n = int(next(tokens))
                        row.append([float(next(tokens)) for _ in range(n)])
                    else:
                        row.append(float(next(tokens)))
                rows.append(row)
            data[name] = (props, rows)
        props, rows = data["vertex"]
        names = [prop[-1] for prop in props]
        vertices = np.array([[row[names.index(c)] for c in "xyz"] for row in rows])
        faces = [row[0] for row in data["face"][1]]
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        offset = 0
        for name, count, props in elements:
            if all(prop[0] != "list" for prop in props):
                dtype = np.dtype([(p[1], endian + types[p[0]]) for p in props])
                array = np.frombuffer(body, dtype, count, offset)
                offset += count * dtype.itemsize
                if name == "vertex":
                    vertices = np.stack([array[c] for c in "xyz"], axis=-1)
            elif name == "face" and len(props) == 1:  # triangles only
                _, n_type, i_type, _ = props[0]
                dtype = np.dtype(
                    [("n", endian + types[n_type]), ("i", endian + types[i_type], 3)]
                )
                array = np.frombuffer(body, dtype, count, offset)
                if (array["n"] != 3).any():
                    raise ValueError(f"{file_name}: binary faces must be triangles")
                faces = array["i"]
                offset += count * dtype.itemsize
            else:
                raise ValueError(f"{file_name}: unsupported element {name}")
    # fan triangulation of polygons
    triangles = [(f[0], f[i], f[i + 1]) for f in faces for i in range(1, len(f) - 1)]
    return np.asarray(vertices, np.float64), np.asarray(triangles, np.int32)


@wp.kernel
def _render(
    mesh: wp.uint64,
    C: wp.array(dtype=wp.vec3),  # camera centre per image
    E: wp.array(dtype=wp.mat33),  # columns: camera axes x, y, z per image
    x0: float,  # normalized coordinate of the edge of the square
    pix: float,  # pixel size in normalized coordinates
    z: wp.array3d(dtype=float),  # out: Z depth, 0 for a miss
):
    c, i, j = wp.tid()
    ray = wp.vec3(x0 + (float(j) + 0.5) * pix, x0 + (float(i) + 0.5) * pix, 1.0)
    q = wp.mesh_query_ray(mesh, C[c], E[c] @ ray, 1.0e7)  # hit at C + t*(E ray)
    z[c, i, j] = wp.where(q.result, q.t, 0.0)


@wp.kernel
def _render_pinhole(
    mesh: wp.uint64,
    C: wp.vec3,  # camera centre (model frame)
    E: wp.mat33,  # columns: camera axes x, y, z (model frame)
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z: wp.array2d(dtype=float),  # out: Z depth, 0 for a miss
):
    i, j = wp.tid()  # pixel (row, col); OpenCV convention: centre at (u, v) = (j, i)
    ray = wp.vec3((float(j) - cx) / fx, (float(i) - cy) / fy, 1.0)
    q = wp.mesh_query_ray(mesh, C, E @ ray, 1.0e7)
    z[i, j] = wp.where(q.result, q.t, 0.0)


@wp.kernel
def _probes(
    z: wp.array3d(dtype=float),  # depth maps
    centre: wp.array(dtype=wp.vec2),  # grid centre per image (pixel units)
    grid_index: wp.array(dtype=int),  # grid position (row-major) of component j
    n_grid: int,
    spacing: float,  # probe spacing (pixels)
    k: float,  # kernel half-size K (pixels)
    z_bg: float,
    h_probe: wp.array2d(dtype=float),  # out: Z_bg - Z_probe
):
    c, j = wp.tid()
    res = z.shape[1]
    half = 0.5 * float(n_grid - 1)
    g = grid_index[j]
    px = centre[c][0] + (float(g % n_grid) - half) * spacing
    py = centre[c][1] + (float(g // n_grid) - half) * spacing
    i_lo = wp.max(int(wp.ceil(py - k - 0.5)), 0)
    i_hi = wp.min(int(wp.floor(py + k - 0.5)), res - 1)
    j_lo = wp.max(int(wp.ceil(px - k - 0.5)), 0)
    j_hi = wp.min(int(wp.floor(px + k - 0.5)), res - 1)
    # largest weighted height above the background (Warp needs the explicit type of a
    # variable that changes in a loop)
    h = float(0.0)  # noqa: UP018
    for i in range(i_lo, i_hi + 1):
        dy = wp.abs(float(i) + 0.5 - py)
        for jj in range(j_lo, j_hi + 1):
            zz = z[c, i, jj]
            if zz > 0.0:
                w = 1.0 - wp.max(dy, wp.abs(float(jj) + 0.5 - px)) / k
                h = wp.max(h, (z_bg - zz) * w)
    h_probe[c, j] = h


def view_from_pose(R, t, fold_cylinder=True):
    """(D, e, psi, phi) of a model-to-camera pose (BOP: R = cam_R_m2c, t = cam_t_m2c).

    The view is that of a virtual camera looking at the model origin, obtained from the
    real camera by the smallest rotation. With fold_cylinder, (e, psi) is mapped to the
    equivalent view with |psi| <= pi/2 by the flip symmetry (e, psi) ~ (-e, psi + pi).
    """
    R, t = np.asarray(R, float).reshape(3, 3), np.asarray(t, float)
    D = np.linalg.norm(t)
    tz = t / D
    s = -R.T @ tz  # model frame: origin -> camera
    e, phi = math.asin(s[2]), math.atan2(s[1], s[0])
    # rotation Q of the real camera axes that turns the optical axis onto tz
    a = np.cross([0.0, 0.0, 1.0], tz)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    sin2 = a @ a
    Q = np.eye(3) + K + (K @ K * (1 - tz[2]) / sin2 if sin2 > 0 else 0)
    z_axis = R[:, 2]  # model +Z in the camera frame
    psi = math.atan2(-(z_axis @ Q[:, 0]), -(z_axis @ Q[:, 1]))
    if fold_cylinder and abs(psi) > math.pi / 2:
        psi, e = psi - math.copysign(math.pi, psi), -e
    return D, e, psi, phi


# charts of view directions: up axis U, azimuth 0 along A, azimuth pi/2 along B
CHART_AXES = {
    "Z": [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    "Y": [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
}


def chart_view_from_pose(R, t, directions):
    """(D, e, phi, psi) of a model-to-camera pose in a chart of directions ("Z", "Y"),
    for the virtual camera that looks at the model origin (obtained from the real
    camera by the smallest rotation); psi in (-pi, pi]."""
    R, t = np.asarray(R, float).reshape(3, 3), np.asarray(t, float)
    U, A, B = (np.asarray(a) for a in CHART_AXES[directions])
    D = np.linalg.norm(t)
    tz = t / D
    s = -R.T @ tz  # model frame: origin -> camera
    e, phi = math.asin(np.clip(s @ U, -1, 1)), math.atan2(s @ B, s @ A)
    a = np.cross([0.0, 0.0, 1.0], tz)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    sin2 = a @ a
    Q = np.eye(3) + K + (K @ K * (1 - tz[2]) / sin2 if sin2 > 0 else 0)
    u = R @ U  # up axis in the camera frame
    psi = math.atan2(-(u @ Q[:, 0]), -(u @ Q[:, 1]))
    return D, e, phi, psi


def spectral_order(n_grid, n, device):
    """The first n real components of the 2D Fourier transform of a real n_grid^2 field:
    frequencies (row, column) on square rings max(|k|, |l|) = 0, 1, ... (cyclic
    indexing), each ring from (r, 0) on by the angle of (k, -l), one frequency of each
    conjugate pair (the first), as Re and Im (Im omitted where zero); factor sqrt(2) for
    the pairs, so that the transform stays orthonormal. Returns (rows, cols, imag,
    factor) tensors of length n."""
    G = n_grid
    half = G // 2
    freqs = [
        (k, l) for k in range(-half + 1, half + 1) for l in range(-half + 1, half + 1)
    ]
    ring = lambda kl: max(abs(kl[0]), abs(kl[1]))
    angle = lambda kl: math.atan2(-kl[1], kl[0]) % (2 * math.pi)
    freqs.sort(key=lambda kl: (ring(kl), angle(kl)))
    seen, comps = set(), []
    for k, l in freqs:
        key = (k % G, l % G)
        if key in seen:
            continue
        seen.add(key)
        conj = ((-k) % G, (-l) % G)
        seen.add(conj)
        if conj == key:  # real component
            comps.append((key[0], key[1], False, 1.0))
        else:
            comps += [
                (key[0], key[1], False, math.sqrt(2)),
                (key[0], key[1], True, math.sqrt(2)),
            ]
    if n > len(comps):
        raise ValueError(f"at most {len(comps)} spectral components for n_grid {G}")
    rows, cols, imag, factor = zip(*comps[:n])
    return (torch.tensor(rows, device=device), torch.tensor(cols, device=device),
            torch.tensor(imag, device=device), torch.tensor(factor, dtype=torch.float64,
                                                            device=device))  # fmt: skip


def fb_components(n):
    """The first n real components of the Fourier-Bessel expansion of a real field on a
    disk: modes (n_r, m) with radial index n_r >= 1 and angular index m >= 0, ordered by
    the shell index si = m + 2 (n_r - 1), then by m; m = 0 gives one component (real),
    m > 0 two (Re, Im). Returns a list of (n_r, m, imag)."""
    comps, si = [], 0
    while len(comps) < n:
        for m in range(si % 2, si + 1, 2):
            n_r = (si - m) // 2 + 1
            comps.append((n_r, m, False))
            if m > 0:
                comps.append((n_r, m, True))
        si += 1
    return comps[:n]


class FourierBessel:
    """Coefficients of the orthonormal Fourier-Bessel basis of the unit disk,
    B_nm(rho, theta) = N_nm J_m(j_mn rho) exp(i m theta), N_nm = 1 / (sqrt(pi)
    |J_{m+1}(j_mn)|) (j_mn: n-th zero of J_m, Dirichlet at rho = 1), of fields sampled on
    a polar grid: rho_k = (k + 1/2) / n_rho, theta_j = 2 pi j / n_theta.
    __call__(S (n, n_rho, n_theta)) -> f (n, n_modes) complex, f_nm = integral of
    S conj(B_nm) dA (midpoint rule in rho, FFT in theta). A rotation of the field by
    alpha multiplies f_nm by exp(-i m alpha)."""

    def __init__(self, modes, n_rho, n_theta, device):
        import mpmath

        self.modes, self.n_theta = list(modes), n_theta
        rho = (np.arange(n_rho) + 0.5) / n_rho
        W = np.zeros((n_rho, len(self.modes)))
        for i, (n_r, m) in enumerate(self.modes):
            j = float(mpmath.besseljzero(m, n_r))
            norm = 1 / (math.sqrt(math.pi) * abs(float(mpmath.besselj(m + 1, j))))
            J = np.array([float(mpmath.besselj(m, j * r)) for r in rho])
            W[:, i] = 2 * math.pi * norm * J * rho / n_rho
        self.W = torch.tensor(W, dtype=torch.float32, device=device)
        self.m = torch.tensor([m for _, m in self.modes], device=device)

    def __call__(self, S):
        C = torch.fft.rfft(S, dim=2) / self.n_theta  # (n, n_rho, m): angular means
        return (C[:, :, self.m] * self.W).sum(dim=1)


class DepthStencil:
    def __init__(
        self,
        model_file,
        n_grid=16,
        probe_order="centre",
        resolution=512,
        kernel="max",
        kernel_size=None,
        spectral=None,
        spectral_scale=1.0,
        fb=None,
        dist_range=(638.38, 775.97),
        elevation_range=(-0.5 * math.pi + 0.3, 0.5 * math.pi - 0.3),
        roll_range=(-0.5 * math.pi + 0.3, 0.5 * math.pi - 0.3),
        azimuth_range=None,
        directions=None,
        margin=1.1,
        s_amp=0.8,
        dp=0.05,
        max_pixels=2**27,
        device="cuda",
        name="depth_stencil",
    ):
        self.name = name
        self.device = torch.device(device)
        self.wp_device = "cuda:0" if self.device.type == "cuda" else "cpu"
        self.directions = directions
        if directions is None:  # p: D, e, psi[, phi]
            self.ranges = [dist_range, elevation_range, roll_range]
            if azimuth_range is not None:
                self.ranges.append(azimuth_range)
        else:  # chart of view directions; p: D, e, phi, psi
            if directions not in CHART_AXES:
                raise ValueError(f"unknown directions {directions}")
            if azimuth_range is None:
                raise ValueError("a chart of directions needs azimuth_range")
            self.ranges = [dist_range, elevation_range, azimuth_range, roll_range]
            self._axes = torch.tensor(CHART_AXES[directions], dtype=torch.float64)
        self.n_p = len(self.ranges)
        lo, hi = torch.tensor(self.ranges, dtype=torch.float64).T
        self._lo, self._hi = lo.to(self.device), hi.to(self.device)
        self.n_grid, self.n_y = n_grid, n_grid**2
        row, col = np.divmod(np.arange(self.n_y), n_grid)
        if probe_order == "centre":
            dr, dc = row - 0.5 * (n_grid - 1), col - 0.5 * (n_grid - 1)
            radius = np.round(np.hypot(dr, dc), 6)  # rounded: ties sorted by angle
            self.grid_index = np.lexsort((np.arctan2(dr, dc), radius))
        elif probe_order == "rows":
            self.grid_index = np.arange(self.n_y)
        else:
            raise ValueError(f"unknown probe_order {probe_order}")
        self._grid_index = torch.tensor(
            self.grid_index, dtype=torch.int32, device=self.device
        )
        self.res = resolution
        self.spacing = resolution / n_grid  # pixels
        if kernel not in ("max", "tent", "flat"):
            raise ValueError(f"unknown kernel {kernel}")
        self.kernel = kernel
        self.kernel_size = self.spacing if kernel_size is None else kernel_size
        self.spectral, self.spectral_scale = spectral, spectral_scale
        if spectral is not None:  # y: the lowest Fourier components of the cells
            if kernel == "max":
                raise ValueError("spectral needs the kernel tent or flat")
            self.n_y = spectral
            self._spec = spectral_order(n_grid, spectral, self.device)
        self.fb = None if fb is None else {**FB_DEFAULTS, **fb}
        if self.fb is not None:  # y: Fourier-Bessel components of the heights
            if spectral is not None:
                raise ValueError("fb and spectral exclude each other")
            self._init_fb()
        self.s_amp = s_amp
        self.dp = (
            torch.as_tensor(dp, dtype=torch.float64).expand(self.n_p).to(self.device)
        )
        self.chunk = max(1, max_pixels // resolution**2)  # images per launch
        if self.fb is not None:  # several full-size maps per image
            self.chunk = max(1, self.chunk // 4)

        vertices, triangles = read_ply(model_file)
        self.radius = np.linalg.norm(vertices, axis=1).max()  # bounding sphere
        self.mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=self.wp_device),
            indices=wp.array(triangles.ravel(), dtype=wp.int32, device=self.wp_device),
        )
        d_min, d_max = dist_range
        # half-side of the square in normalized image coordinates
        self.half_width = margin * self.radius / math.sqrt(d_min**2 - self.radius**2)
        self.pix = 2 * self.half_width / resolution
        self.z_lo, self.z_bg = d_min - self.radius, d_max + self.radius

    def _init_fb(self):
        o = self.fb
        self.n_y = o["n"]
        comps = fb_components(o["n"] - (o["depth"] is not None))
        modes = sorted({(n_r, m) for n_r, m, _ in comps}, key=lambda nm: (nm[1], nm[0]))
        self._fbt = FourierBessel(modes, o["n_rho"], o["n_theta"], self.device)
        index = {nm: i for i, nm in enumerate(modes)}
        self._fb_mode = torch.tensor([index[(n_r, m)] for n_r, m, _ in comps],
                                     device=self.device)  # fmt: skip
        self._fb_imag = torch.tensor([im for _, _, im in comps], device=self.device)
        factor = [(math.sqrt(2) if m > 0 else 1.0)
                  * math.exp(-o["gamma"] * (n_r / o["n_cr"]) ** 2) * o["scale"]
                  for n_r, m, _ in comps]  # fmt: skip
        self._fb_factor = torch.tensor(factor, device=self.device)
        radius = o["radius"] * 0.5 * self.res  # pixels
        rho = (torch.arange(o["n_rho"], device=self.device) + 0.5) / o["n_rho"]
        theta = torch.arange(o["n_theta"], device=self.device) * (2 * math.pi
                                                                  / o["n_theta"])  # fmt: skip
        self._fb_offsets = radius * torch.stack(
            [rho[:, None] * theta.cos(), rho[:, None] * theta.sin()], dim=-1
        )  # (n_rho, n_theta, 2): x (columns), y (rows) in pixels

    def _fb_coeffs(self, z, centre):
        """Fourier-Bessel coefficients f (n, n_modes) (complex, before damping and ISRU)
        of the heights of depth maps z (n, res, res) about centre (n, 2) (pixel units),
        and the argument x_D (n,) of the depth component (None without it)."""
        o, scale = self.fb, self.z_bg - self.z_lo
        mask = z > 0
        zm = torch.where(mask, z, 0.0).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(
            min=1
        )
        if o["background"] is None:  # heights above Z_bg
            h = torch.where(mask, (self.z_bg - z) / scale, 0.0)
        else:  # background brought to the mean depth of the mask + background (mm)
            ref = (zm + o["background"])[:, None, None]
            h = torch.where(mask, (ref - z).clamp(min=0) / scale, 0.0)
        pts = centre[:, None, None, :].to(h.dtype) + self._fb_offsets
        grid = pts / self.res * 2 - 1
        S = torch.nn.functional.grid_sample(h[:, None], grid, mode="bilinear",
                                            padding_mode="zeros", align_corners=False)  # fmt: skip
        xd = None
        if o["depth"] is not None:
            xd = o["depth"] * (zm - 0.5 * (self.z_lo + self.z_bg)) / scale
        return self._fbt(S[:, 0]), xd

    def fb_x(self, f, xd=None):
        """Arguments x (n, n_y) of the ISRU for coefficients f (n, n_modes) (and x_D)."""
        v = torch.where(
            self._fb_imag, f[:, self._fb_mode].imag, f[:, self._fb_mode].real
        )
        x = v * self._fb_factor
        if xd is not None:
            x = torch.cat([xd[:, None].to(x.dtype), x], dim=1)
        return x

    def _fb_y(self, z, centre):
        """y of depth maps z (n, res, res) for the Fourier-Bessel stencil, centred at
        centre (n, 2) in pixel units."""
        x = self.fb_x(*self._fb_coeffs(z, centre))
        return x / torch.sqrt(1 + x * x)  # ISRU

    def fb_coeffs(self, p):
        """(f (n, n_modes) complex, x_D (n,) or None) of the views p (n, n_p)."""
        out = []
        for q in p.split(self.chunk):
            z = self.depth_image(q)
            mask = (z > 0).float()
            count = mask.sum(dim=(1, 2))
            x = torch.arange(self.res, device=self.device) + 0.5
            cx = (mask.sum(dim=1) * x).sum(dim=1)
            cy = (mask.sum(dim=2) * x).sum(dim=1)
            centre = torch.stack([cx, cy], dim=-1) / count.clamp(min=1)[:, None]
            centre[count == 0] = 0.5 * self.res
            out.append(self._fb_coeffs(z, centre))
        f = torch.cat([o[0] for o in out])
        xd = None if out[0][1] is None else torch.cat([o[1] for o in out])
        return f, xd

    def view(self, p):
        """D (mm) and the angles e, psi[, phi] (rad) for p (n_c, n_p)."""
        lo, hi = self._lo.to(p.device, p.dtype), self._hi.to(p.device, p.dtype)
        return lo + 0.5 * (p + 1) * (hi - lo)

    def p_from_view(self, view):
        """Inverse of view()."""
        lo, hi = (
            self._lo.to(view.device, view.dtype),
            self._hi.to(view.device, view.dtype),
        )
        return 2 * (view - lo) / (hi - lo) - 1

    def _frames(self, p):
        """Camera centres (n, 3) and axes (n, 3, 3; columns x, y, z), model frame."""
        if self.directions is not None:
            return self._chart_frames(p)
        v = self.view(p.to(self.device, torch.float64))
        D, e, psi = v[:, 0], v[:, 1], v[:, 2]
        phi = v[:, 3] if self.n_p == 4 else torch.zeros_like(D)
        ce, se, cf, sf = e.cos(), e.sin(), phi.cos(), phi.sin()
        s = torch.stack(
            [ce * cf, ce * sf, se], dim=-1
        )  # unit vector towards the camera
        ex0 = torch.stack([-sf, cf, torch.zeros_like(sf)], dim=-1)
        ey0 = torch.stack([se * cf, se * sf, -ce], dim=-1)  # image down for psi = 0
        cp, sp = psi.cos()[:, None], psi.sin()[:, None]
        E = torch.stack([cp * ex0 + sp * ey0, -sp * ex0 + cp * ey0, -s], dim=-1)
        return (D[:, None] * s).float().contiguous(), E.float().contiguous()

    def _chart_frames(self, p):
        """_frames for a chart of directions: s = cos e (cos phi A + sin phi B) + sin e U
        (origin -> camera); image up for psi = 0: U projected on the image plane."""
        v = self.view(p.to(self.device, torch.float64))
        D, e, phi, psi = v.unbind(dim=1)
        U, A, B = self._axes.to(v.device)
        s = e.cos()[:, None] * (phi.cos()[:, None] * A + phi.sin()[:, None] * B)
        s = s + e.sin()[:, None] * U
        up = U - (s @ U)[:, None] * s
        ey0 = -up / up.norm(dim=1, keepdim=True)  # image down for psi = 0
        ez = -s
        ex0 = torch.linalg.cross(ey0, ez)
        cp, sp = psi.cos()[:, None], psi.sin()[:, None]
        E = torch.stack([cp * ex0 + sp * ey0, -sp * ex0 + cp * ey0, ez], dim=-1)
        return (D[:, None] * s).float().contiguous(), E.float().contiguous()

    def render_pinhole(self, R, t, K, width, height):
        """Z depth (height, width) of the model with the pose R, t (model to camera, mm)
        seen by a pinhole camera with the intrinsic matrix K (OpenCV), 0 for a miss."""
        R = torch.as_tensor(R, dtype=torch.float64).reshape(3, 3)
        t = torch.as_tensor(t, dtype=torch.float64).reshape(3)
        K = torch.as_tensor(K, dtype=torch.float64).reshape(3, 3)
        C = (-R.T @ t).float().tolist()  # camera centre in the model frame
        E = R.T.float().tolist()  # columns: camera axes in the model frame
        z = torch.empty((height, width), device=self.device)
        wp.launch(
            _render_pinhole,
            dim=z.shape,
            inputs=[
                self.mesh.id,
                wp.vec3(*C),
                wp.mat33(*[v for row in E for v in row]),
                float(K[0, 0]),
                float(K[1, 1]),
                float(K[0, 2]),
                float(K[1, 2]),
                wp.from_torch(z),
            ],
            device=self.wp_device,
        )
        return z

    def depth_image(self, p):
        """Z depth (n, res, res) over the square, 0 for a miss."""
        C, E = self._frames(p)
        z = torch.empty((p.shape[0], self.res, self.res), device=self.device)
        wp.launch(
            _render,
            dim=z.shape,
            inputs=[
                self.mesh.id,
                wp.from_torch(C, dtype=wp.vec3),
                wp.from_torch(E, dtype=wp.mat33),
                -self.half_width,
                self.pix,
                wp.from_torch(z),
            ],
            device=self.wp_device,
        )
        return z

    def _forward(self, p):
        """y (n, n_y) and the centre of mass (n, 2) in pixel units, for one chunk."""
        return self._y_of_depth(self.depth_image(p))

    def y_from_depth(self, z, centre=None):
        """y (n, n_y) and the centre (x_S, y_S) (n, 2) of the probe grid in normalized
        image coordinates, for depth maps z (n, res, res) over the square of the stencil:
        Z in mm where z > 0 (the mask), 0 for background; NaN (unknown) is treated as
        background, i.e. ignored by the kernels. The grid is centred at the centre of
        mass of the mask, or at centre (n, 2; normalized image coordinates) if given."""
        if centre is None:
            parts = [self._y_of_depth(c.contiguous()) for c in z.split(self.chunk)]
        else:
            px = ((centre + self.half_width) / self.pix).float().split(self.chunk)
            parts = [
                self._y_of_depth(c.contiguous(), q)
                for c, q in zip(z.split(self.chunk), px)
            ]
        ys, centres = zip(*parts)
        return torch.cat(ys), torch.cat(centres) * self.pix - self.half_width

    def _y_of_depth(self, z, centre=None):
        """y (n, n_y) and the centre of the probe grid (n, 2) in pixel units (default:
        the centre of mass of the mask), for one chunk."""
        if centre is None:
            mask = (z > 0).float()
            count = mask.sum(dim=(1, 2))
            x = torch.arange(self.res, device=self.device) + 0.5
            cx = (mask.sum(dim=1) * x).sum(dim=1)
            cy = (mask.sum(dim=2) * x).sum(dim=1)
            centre = torch.stack([cx, cy], dim=-1) / count.clamp(min=1)[:, None]
            centre[count == 0] = 0.5 * self.res
        if self.fb is not None:
            return self._fb_y(z, centre), centre
        if self.kernel in ("tent", "flat"):
            h = self._cell_heights(z, centre)
            if self.spectral is not None:
                return self._spectral_y(h.double()).float(), centre
            h = h.reshape(len(z), -1)[:, self._grid_index.long()]
            return self.s_amp * (1 - 2 * h / (self.z_bg - self.z_lo)), centre
        h = torch.empty((z.shape[0], self.n_y), device=self.device)
        wp.launch(
            _probes,
            dim=h.shape,
            inputs=[
                wp.from_torch(z),
                wp.from_torch(centre.contiguous(), dtype=wp.vec2),
                wp.from_torch(self._grid_index),
                self.n_grid,
                self.spacing,
                self.kernel_size,
                self.z_bg,
                wp.from_torch(h),
            ],
            device=self.wp_device,
        )
        y = self.s_amp * (1 - 2 * h / (self.z_bg - self.z_lo))
        return y, centre

    def _cell_heights(self, z, centre):
        """Heights above the background (0 for background and NaN) at the probes,
        (n, row, column), for the kernels with separable weights: "tent": weighted with
        (1 - |dx|/K)(1 - |dy|/K), |dx|, |dy| < K, around each probe (with K = probe
        spacing, the weights of a pixel sum to one over the grid); "flat": the mean over
        the cell of side K centred at the probe (pixels cut by its border with their
        overlap, so that the mean changes continuously when the grid moves)."""
        a = torch.where(z > 0, self.z_bg - z, 0.0)
        x = torch.arange(self.res, device=self.device, dtype=a.dtype)  # pixel edges
        off = torch.arange(self.n_grid, device=self.device) - 0.5 * (self.n_grid - 1)
        k = self.kernel_size

        def weights(probes):  # (n, n_grid, res) for probe positions (n, n_grid)
            if self.kernel == "tent":
                return (1 - (x + 0.5 - probes[:, :, None]).abs() / k).clamp(min=0)
            lo, hi = probes[:, :, None] - 0.5 * k, probes[:, :, None] + 0.5 * k
            return (torch.minimum(x + 1, hi) - torch.maximum(x, lo)).clamp(min=0)

        w_x = weights(centre[:, :1] + off * self.spacing)
        w_y = weights(centre[:, 1:] + off * self.spacing)
        return w_y @ a @ w_x.transpose(1, 2) / k**2

    def _spectral_y(self, h):
        """y of cell heights h (n, G, G) for spectral: -2 s_amp spectral_scale times the
        orthonormal 2D FFT of h / (Z_bg - Z_lo), components in the order of
        spectral_order (one of each conjugate pair, as sqrt(2) Re and sqrt(2) Im); linear
        in h. (With spectral_scale 1, the lowest components vary 2-6 times faster with p
        than the central probes of the flat kernel; 0.2 matches them.)"""
        F = torch.fft.fft2(h / (self.z_bg - self.z_lo), norm="ortho")
        rows, cols, imag, factor = self._spec
        v = F[:, rows, cols]
        v = torch.where(imag, v.imag, v.real) * factor
        return -2 * self.s_amp * self.spectral_scale * v

    def _heights(self, p):
        """Cell heights (n, G, G) of the renderings at p (one chunk), centred at the
        centre of mass of each mask."""
        z = self.depth_image(p)
        mask = (z > 0).float()
        count = mask.sum(dim=(1, 2))
        x = torch.arange(self.res, device=self.device) + 0.5
        cx, cy = (mask.sum(dim=1) * x).sum(dim=1), (mask.sum(dim=2) * x).sum(dim=1)
        centre = torch.stack([cx, cy], dim=-1) / count.clamp(min=1)[:, None]
        centre[count == 0] = 0.5 * self.res
        return self._cell_heights(z, centre)

    def forward(self, p):
        """y (n_c, n_y) and the centre of mass (x_S, y_S) (n_c, 2) of the mask in
        normalized image coordinates, for p (n_c, n_p)."""
        ys, centres = zip(*(self._forward(q) for q in p.split(self.chunk)))
        centre = torch.cat(centres) * self.pix - self.half_width
        return torch.cat(ys), centre

    def compute(self, p, y, dy_dp):
        """y (n_c, n_y) and dy/dp (n_c, n_y, n_p) for p (n_c, n_p), in place."""
        p = p.to(self.device, torch.float64)
        n_c = p.shape[0]
        steps = torch.diag(self.dp)  # (n_p, n_p)
        # cases: p, then p + dp_k e_k and p - dp_k e_k for each k
        q = torch.cat([p[None], p[None] + steps[:, None], p[None] - steps[:, None]])
        if self.spectral is not None:  # differences of the cell heights, then the FFT
            h = torch.cat(
                [self._heights(c) for c in q.reshape(-1, self.n_p).split(self.chunk)]
            )
            h = h.double().view(1 + 2 * self.n_p, n_c, self.n_grid, self.n_grid)
            dh = (h[1 : 1 + self.n_p] - h[1 + self.n_p :]) / (
                2 * self.dp[:, None, None, None]
            )
            y.copy_(self._spectral_y(h[0]))
            dy = self._spectral_y(dh.reshape(-1, self.n_grid, self.n_grid))
            dy_dp.copy_(dy.view(self.n_p, n_c, self.n_y).permute(1, 2, 0))
            return
        s, _ = self.forward(q.reshape(-1, self.n_p))
        s = s.view(1 + 2 * self.n_p, n_c, self.n_y)
        y.copy_(s[0])
        diff = (s[1 : 1 + self.n_p] - s[1 + self.n_p :]) / (2 * self.dp[:, None, None])
        dy_dp.copy_(diff.permute(1, 2, 0))
