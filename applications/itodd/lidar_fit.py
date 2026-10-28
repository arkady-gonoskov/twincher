"""LidarFit: 6D pose of an object instance in a depth image of a BOP dataset (stage 5).

One object model; one or more charts (twinchers, each with its stencil, covering
overlapping ranges of view directions); one instance at a time. The configuration (JSON,
e.g. lidarfit_obj12_m1.json) gives the model files, the charts and the solver settings.

process_instance(path, im_id, inst_id) for a BOP scene directory path:
1. reads depth/<im>.tif, mask/<im>_<inst>.png (silhouette including occluded parts),
   mask_visib/<im>_<inst>.png and the camera (cam_K, depth_scale) of scene_camera.json;
   a smaller visible frame can be given (frame) to simulate the image border; masks that
   touch the border are dismissed unless edge_cases is set;
2. initial virtual camera V: the real camera turned by the smallest rotation Q whose
   optical axis passes through the centre of mass of the mask (observation "mask") or
   of mask_visib (observation "visib"), in the frame;
3. stencil centre c (2 normalized coordinates of V): direction of the model origin;
   the stencil camera S(c) is V turned by the smallest rotation that moves its axis to c;
   the observation is resampled into the square of the stencil seen by S(c) (nearest
   pixel; depth along the optical axis of S(c)) and fused with renderings z (fuse()):
   - "mask" (map_avail): the observed depth on mask_visib, background outside the mask,
     unknown (NaN, filled from z) where the object is occluded (mask but not
     mask_visib), has no depth, or (edge_cases) lies outside the frame;
   - "visib" (no use of the mask): the observed depth on mask_visib; outside it, free
     space (background) where z shows the object but the observed surface lies more
     than carve_mm behind it; z elsewhere (observed surface in front: occlusion; no
     depth; outside the frame);
4. p by the fused solver (FusedHSSolver, Newton or plain step, from p_start or 0), for
   each chart and each attempt; c by the centre mode:
   - "com": c = 0 (S = V); the pose then matches the centres of mass of the
     silhouettes of the rendering and of the fused map (as in the first version);
   - "dist_y": before each step, c is moved towards the minimum of a quadratic fit of
     dist(c) = |y_fused(c) - y_curr|^2 on a 3 x 3 grid of step h (initially
     centre_h * side of the square), at most 2h, if dist decreases (otherwise h / 2);
     the probes of y_fused are centred at its own centre of mass (dist_grid "own") or
     at the grid of y_curr ("curr"), which makes dist sensitive to a plain shift;
   - "com_track": before each step, c is moved so that the centre of mass of the
     observed silhouette coincides with that of the rendering (the shift divided by the
     observed fraction of the silhouette, i.e. as if the rendering were right); the
     observed silhouette (centre_silhouette): "fused" (the fused map) or "mask" (the
     mask in the frame, completed by the rendering outside the frame only; pixels of the
     mask without depth then count as object, which removes the bias of one-sided
     structured-light shadows);
   the solution of the chart and attempt with the smallest misfit (RMS of the y-tone
   difference between the rendering and the fused map on its known pixels) is kept;
5. pose: "com": R_m2c = Q R_SV R_m2s, t_m2c = Q R_SV (0, 0, D), with R_SV turning the
   centre of mass of the rendering (c_S) onto that of the fused map (c_V); other modes:
   R_m2c = Q_S(c) R_m2s, t_m2c = Q_S(c) (0, 0, D);
6. errors against scene_gt.json if present: translation, angle of the symmetry axis,
   MSSD (BOP: maximum vertex distance, minimized over the symmetries of models_info.json);
7. depth map of the solution rendered with the real camera (full frame; saved as TIFF,
   like the depth maps of BOP), compared with the observed depth on the available pixels
   (mask_visib with depth, in the frame): signed Z_solution - Z_observed, its plane fit
   (offset, tilt) and the scatter around the plane; pixels not covered by the solution.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import twincher
from PIL import Image

from depth_stencil import DepthStencil, chart_view_from_pose, read_ply, view_from_pose
from fused_solver import FusedHSSolver
from itodd_data import resolve

EVAL_BATCH = 16  # cases of the shuttle that evaluates r for dist(c)


def smallest_rotation(a, b):
    """Rotation matrix (3, 3) that turns the direction a onto the direction b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), a @ b
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if v @ v < 1e-24:
        if c > 0:
            return np.eye(3)
        raise ValueError("opposite directions")
    return np.eye(3) + K + K @ K / (1 + c)


def symmetry_transforms(info, n_continuous=360):
    """Rotations (n, 3, 3) and translations (n, 3) of the symmetries of a model (BOP
    models_info: discrete ones combined with continuous ones discretized in n steps)."""
    discrete = [(np.eye(3), np.zeros(3))]
    for m in info.get("symmetries_discrete", []):
        m = np.asarray(m, float).reshape(4, 4)
        discrete.append((m[:3, :3], m[:3, 3]))
    continuous = [(np.eye(3), np.zeros(3))]
    for sym in info.get("symmetries_continuous", []):
        axis = np.asarray(sym["axis"], float) / np.linalg.norm(sym["axis"])
        offset = np.asarray(sym["offset"], float)
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        continuous = []
        for phi in np.linspace(0, 2 * math.pi, n_continuous, endpoint=False):
            R = np.eye(3) + math.sin(phi) * K + (1 - math.cos(phi)) * K @ K
            continuous.append((R, offset - R @ offset))  # rotation about the axis line
    Rs, ts = [], []
    for Rc, tc in continuous:
        for Rd, td in discrete:
            Rs.append(Rc @ Rd)
            ts.append(Rc @ td + tc)
    return np.array(Rs), np.array(ts)


def read_image(file_name, dtype=np.float64):
    return np.asarray(Image.open(file_name), dtype=dtype)


def finite_differences(y, dp):
    """dy/dp (n, n_y, n_p) from y at p, p + dp_k e_k (k < n_p), p - dp_k e_k."""
    n_p = len(dp)
    diff = [(y[1 + k] - y[1 + n_p + k]).double() / (2 * dp[k]) for k in range(n_p)]
    return torch.stack(diff, dim=-1).float()


def fill_nearest(values, known, region, max_iter=1000):
    """values (2D tensor) in region from the nearest known pixels, by iterative
    dilation (each step: the mean of the known pixels among the 3 x 3 neighbours);
    0 outside region and where no known pixel is reached."""
    import torch.nn.functional as F

    z = torch.where(known, values, 0.0)[None, None]
    k = known.to(values.dtype)[None, None]
    w = torch.ones((1, 1, 3, 3), dtype=values.dtype, device=values.device)
    todo = region & ~known
    for _ in range(max_iter):
        count = F.conv2d(k, w, padding=1)[0, 0]
        new = todo & (count > 0)
        if not new.any():
            break
        z[0, 0] = torch.where(new, F.conv2d(z, w, padding=1)[0, 0] / count, z[0, 0])
        k[0, 0] = torch.where(new, 1.0, k[0, 0])
        todo &= ~new
    return torch.where(region & (k[0, 0] > 0), z[0, 0], 0.0)


class Chart:
    """A twincher with its stencil, its solver and a shuttle that evaluates r(y).

    Without stencil_cfg (the charts of train_chart.py), the stencil is that of the
    metadata of the twincher; otherwise (the cylinder of milestone 1) it is built from
    stencil_cfg (updated by the chart's own "stencil" entries) and checked against the
    metadata."""

    def __init__(self, base, cfg, stencil_cfg, device):
        self.twc = base / cfg["twincher"]
        self.name = cfg.get("name", self.twc.stem)
        meta = twincher.load_metadata(str(self.twc))["stencil"]
        if stencil_cfg is None:
            from train_chart import stencil_from_metadata

            self.stencil = stencil_from_metadata(meta)
            self._init_solver()
            return
        sc = {**stencil_cfg, **cfg.get("stencil", {})}
        model = resolve(base, cfg["model"])
        self.stencil = DepthStencil(model, device=device, **sc)
        meta.setdefault("kernel", "max")  # files written before the kernel option
        sc.setdefault("kernel", "max")
        for key in ("n_grid", "resolution", "dp", "s_amp", "kernel"):
            if meta[key] != sc[key]:
                raise ValueError(f"{self.name}: {key} of the twincher {meta[key]}, "
                                 f"of the configuration {sc[key]}")  # fmt: skip
        ranges = [sc["dist_range"], sc["elevation_range"], sc["roll_range"]]
        if not np.allclose(meta["ranges"], ranges):
            raise ValueError(
                f"{self.name}: ranges of twincher and configuration differ"
            )
        self._init_solver()

    def _init_solver(self):
        self.solver = FusedHSSolver(str(self.twc), n_c=1)
        self.shuttle = twincher.Shuttle()
        self.shuttle.load(str(self.twc), n_c=EVAL_BATCH, n_p=self.stencil.n_p, n_m=1)

    def r_of(self, y):
        """r (n, n_p) for y (n, n_y), n <= EVAL_BATCH."""
        n, n_y, n_p = len(y), self.stencil.n_y, self.stencil.n_p
        s = self.shuttle.s
        s.fill_(0)
        s[:n_y, :n].copy_(y.T)
        self.shuttle.forward_inference()
        return s[:n_p, :n].T.clone()

    def p_of_pose(self, R, t):
        """Parameters p (n_p,) of the view of a model-to-camera pose in this chart (charts
        of view directions only; the roll wrapped into the turn about the chart's roll
        centre)."""
        st = self.stencil
        D, e, phi, psi = chart_view_from_pose(R, t, st.directions)
        lo, hi = st._lo.double(), st._hi.double()
        centre = 0.5 * (lo[3] + hi[3]).item()
        psi = centre + (psi - centre + math.pi) % (2 * math.pi) - math.pi
        view = torch.tensor([[D, e, phi, psi]], dtype=torch.float64)
        return st.p_from_view(view)[0].numpy()


class LidarFit:
    def __init__(self, config_file, device="cuda"):
        self.config_file = Path(config_file).resolve()
        base = self.config_file.parent
        self.cfg = cfg = json.loads(self.config_file.read_text())
        self.obj_id = cfg["obj_id"]
        charts = cfg.get("charts", [{"twincher": cfg.get("twincher")}])
        self.charts = [Chart(base, {"model": cfg["model"], **c}, cfg.get("stencil"),
                             device) for c in charts]  # fmt: skip
        sol = cfg["solver"]
        self.n_iter = sol["n_iter"]
        self.attempts = sol.get("attempts", [{"newton": sol.get("newton", True)}])
        self.centre = sol.get("centre", "com_track")
        # p kept in |p_k| <= clamp after each step (the box on which the twinchers are
        # embeddings; None: no clamp, as in milestone 1)
        self.clamp = sol.get("clamp")
        self.centre_h = sol.get("centre_h", 0.1)
        self.edge_cases = sol.get("edge_cases", True)
        # com_track: observed silhouette for the centre of mass: "fused" (map_avail with
        # its unknown pixels from the rendering) or "mask" (the mask in the frame, and the
        # rendering only outside the frame)
        self.centre_silhouette = sol.get("centre_silhouette", "fused")
        # observation: "mask" (depth, mask and mask_visib) or "visib" (depth and
        # mask_visib only; the mask is then used only to evaluate the solution)
        self.observation = sol.get("observation", "mask")
        self.carve_mm = sol.get("carve_mm", 5.0)
        # dist_y: probes of y_fused at their own centre of mass ("own") or at the grid of
        # y_curr ("curr")
        self.dist_grid = sol.get("dist_grid", "own")
        # pixels of mask_visib without depth (sensor holes) count as object (known), with
        # the depth of the rendering, or of the nearest pixels of mask_visib with depth
        # where the rendering misses them: the visible silhouette is enforced (both
        # observations; "mask" without it: unknown, i.e. the rendering, possibly empty)
        self.fill_visib = sol.get("fill_visib", False)
        # the filled pixels also in the target of the solver (else only in the centre
        # refinement and the misfit: the filled depth, copied from the nearest lit
        # pixels, is too near at the grazing edges and biases the depth of p)
        self.fill_target = sol.get("fill_target", False)
        # "visib", dist_y: penalty for observed object outside the stencil square
        self.dist_guard = sol.get("dist_guard", True)
        vertices, _ = read_ply(resolve(base, cfg["model_eval"]))
        self.eval_points = vertices
        info = json.loads(resolve(base, cfg["models_info"]).read_text())
        info = info[str(self.obj_id)]
        self.diameter = info["diameter"]
        self.sym_R, self.sym_t = symmetry_transforms(info)
        self.axis_symmetric = bool(info.get("symmetries_continuous"))
        self.device = device

    # --- observation ---------------------------------------------------------------
    def sample(self, obs, st, Q):
        """The observation in the stencil square of the camera Q (real <- camera),
        nearest pixel: depth along the optical axis of Q (0 where none), whether the
        pixel lies in the frame, mask (None for the observation "visib"), mask_visib
        (both in the frame) and the filled depth of fill_visib (or None)."""
        dev = st.device
        height, width = obs["depth"].shape
        x0, x1, y0, y1 = obs["frame"]
        x = (
            -st.half_width
            + (torch.arange(st.res, dtype=torch.float64, device=dev) + 0.5) * st.pix
        )
        yy, xx = torch.meshgrid(x, x, indexing="ij")
        rays = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)  # camera frame
        rays = rays @ torch.as_tensor(Q, device=dev).T  # real camera frame
        K = obs["K"]
        u = K[0, 0] * rays[..., 0] / rays[..., 2] + K[0, 2]
        v = K[1, 1] * rays[..., 1] / rays[..., 2] + K[1, 2]
        col, row = torch.round(u).long(), torch.round(v).long()
        inside = (col >= x0) & (col < x1) & (row >= y0) & (row < y1)
        col, row = col.clamp(0, width - 1), row.clamp(0, height - 1)

        def along_axis(z):  # depth along the optical axis of Q, 0 where none
            z = z[row, col]
            return torch.where((z > 0) & inside, z / rays[..., 2], 0.0)

        mask = None if obs["t_mask"] is None else obs["t_mask"][row, col] & inside
        visib = obs["t_mask_visib"][row, col] & inside
        fill = None if obs.get("t_fill") is None else along_axis(obs["t_fill"])
        return along_axis(obs["t_depth"]), inside, mask, visib, fill

    def resample(self, obs, st, Q, with_mask=False):
        """map_avail (res, res) in the stencil square of the camera Q (real <- camera);
        with_mask: also the mask in the frame (bool, res x res)."""
        return self._avail(st, *self.sample(obs, st, Q)[:4], with_mask)

    def _avail(self, st, depth, inside, m, visib, with_mask=False):
        available = m & visib & (depth > 0)
        out = torch.zeros((st.res, st.res), dtype=torch.float64, device=st.device)
        out[m] = torch.nan
        if self.edge_cases:
            out[~inside] = torch.nan
        out[available] = depth[available]
        return (out.float(), m) if with_mask else out.float()

    def observe(self, obs, st, Q):
        """The observation in the stencil square of the camera Q, for fuse(): "mask":
        {"avail": map_avail}; "visib": {"depth": depth along the axis, NaN where none
        or outside the frame; "seen": pixels of mask_visib with depth}; with fill_visib
        also "fill": the filled depth on mask_visib without depth, NaN elsewhere."""
        if self.observation == "mask":
            depth, inside, m, visib, fill = self.sample(obs, st, Q)
            ob = {"avail": self._avail(st, depth, inside, m, visib)}
            if fill is not None:
                hole = m & visib & ~(depth > 0) & (fill > 0)
                ob["fill"] = torch.where(hole, fill, torch.nan).float()
            return ob
        depth, _, _, visib, fill = self.sample(obs, st, Q)
        depth = torch.where(depth > 0, depth, torch.nan).float()
        ob = {"depth": depth, "seen": visib & ~torch.isnan(depth)}
        if fill is not None:
            hole = visib & torch.isnan(depth) & (fill > 0)
            ob["fill"] = torch.where(hole, fill, torch.nan).float()
        return ob

    def fuse(self, z, ob):
        """map_fused for renderings z (n, res, res) and the observation ob, and the known
        pixels (whose value comes from the observation).

        "mask": the available depth, background outside the mask, z elsewhere (unknown).
        "visib": the observed depth on mask_visib; outside it, a pixel where z is the
        object but the observed surface lies more than carve_mm behind it is free space
        (background); elsewhere (observed surface in front: occlusion, or no depth) z.
        With fill_visib (both), pixels of mask_visib without depth are object: z where it
        is the object, the filled depth where z misses it (known in both cases)."""
        if self.observation == "mask":
            unknown = torch.isnan(ob["avail"])
            fused = torch.where(unknown, z, ob["avail"])
            known = ~unknown.expand_as(fused)
        else:
            depth, seen = ob["depth"], ob["seen"]
            carved = ~seen & (z > 0) & (depth > z + self.carve_mm)  # NaN: not carved
            fused = torch.where(seen, depth, torch.where(carved, 0.0, z))
            known = seen | carved
        if "fill" in ob:
            hole = ~torch.isnan(ob["fill"])
            fused = torch.where(hole & (z <= 0), ob["fill"], fused)
            known = known | hole
        return fused, known

    @staticmethod
    def camera_of(Q, c):
        """Rotation (real <- stencil camera) of the stencil camera with centre c."""
        return Q @ smallest_rotation([0.0, 0.0, 1.0], [c[0], c[1], 1.0])

    # --- solver ----------------------------------------------------------------------
    def solve(self, chart, obs, Q, attempt):
        """p, c and the observation in the stencil square (for the final c) of one
        chart and attempt."""
        st, solver = chart.stencil, chart.solver
        newton = attempt["newton"]
        c = np.zeros(2)
        h = self.centre_h * 2 * st.half_width
        ob = self.observe(obs, st, self.camera_of(Q, c))
        p_start = attempt.get("p_start")
        if p_start is None:
            solver.p.fill_(0)
        else:
            solver.p.copy_(torch.tensor([p_start], device=solver.p.device))
        steps = torch.diag(st.dp).to(st.device, torch.float64)
        for it in range(self.n_iter):
            p = solver.p.to(st.device, torch.float64)
            # p, p + dp_k e_k, p - dp_k e_k, rendered and evaluated in one batch
            points = torch.cat([p, p + steps, p - steps])
            maps = st.depth_image(points)
            y_c = st.y_from_depth(maps)[0][:, None]  # (1 + 2 n_p, 1, n_y)
            if self.centre == "dist_y":
                c, h, ob = self.refine_dist(chart, obs, Q, c, h, maps[:1], y_c[0], ob)
            elif self.centre == "com_track":
                c, ob = self.refine_com(obs, st, Q, c, maps[:1], ob)
            fused = maps if newton else maps[:1]
            target = ob if self.fill_target else {k: ob[k] for k in ob if k != "fill"}
            y_f = st.y_from_depth(self.fuse(fused, target)[0])[0][:, None]
            if it == 0:
                solver.start_solution(y_f[0], p_start=solver.p.clone())
            else:
                solver.set_target(y_f[0])
            dy_c = finite_differences(y_c, st.dp)
            if newton:
                dy = solver.newton_dy_dp(
                    y_c[0], dy_c, y_f[0], finite_differences(y_f, st.dp)
                )
            else:
                dy = dy_c
            solver.step(y=y_c[0], dy_dp=dy)
            if self.clamp is not None:
                solver.p.clamp_(-self.clamp, self.clamp)
        return solver.p.double().clone(), c, ob

    def refine_dist(self, chart, obs, Q, c, h, map_curr, y_curr, ob):
        """One update of the stencil centre by the quadratic model of dist(c)."""
        st = chart.stencil
        grid = None
        if self.dist_grid == "curr":  # probes of y_fused at the grid of y_curr
            y_curr, grid = st.y_from_depth(map_curr)

        def dist(centres):
            obs_c = [self.observe(obs, st, self.camera_of(Q, cc)) for cc in centres]
            fused = torch.cat([self.fuse(map_curr, o)[0] for o in obs_c])
            g = None if grid is None else grid.expand(len(centres), 2)
            y = st.y_from_depth(fused, centre=g)[0]
            d = ((y - y_curr) ** 2).sum(dim=1).cpu().numpy()
            if self.observation == "visib" and self.dist_guard:
                # observed object outside the square is unexplained: the fraction of the
                # pixels of mask_visib with depth there, times the largest possible dist
                d_max = st.n_y * (2 * st.s_amp) ** 2
                d += d_max * np.array(
                    [self.seen_outside(obs, st, Q, cc) for cc in centres]
                )
            return d, obs_c

        offsets = np.array([(i, j) for j in (-1, 0, 1) for i in (-1, 0, 1)], float)
        d, obs_c = dist([c + h * o for o in offsets])
        # least-squares fit d = a + g.x + x.H.x / 2 on the 3 x 3 grid (x = h * offset)
        x = h * offsets
        A = np.stack([np.ones(9), x[:, 0], x[:, 1], 0.5 * x[:, 0] ** 2,
                      x[:, 0] * x[:, 1], 0.5 * x[:, 1] ** 2], axis=1)  # fmt: skip
        _, gx, gy, hxx, hxy, hyy = np.linalg.lstsq(A, d, rcond=None)[0]
        g, H = np.array([gx, gy]), np.array([[hxx, hxy], [hxy, hyy]])
        if np.all(np.linalg.eigvalsh(H) > 0):
            step = -np.linalg.solve(H, g)
        else:  # no minimum: steepest descent to the trust-region boundary
            step = -2 * h * g / max(np.linalg.norm(g), 1e-30)
        if np.linalg.norm(step) > 2 * h:
            step *= 2 * h / np.linalg.norm(step)
        d_new, ob_new = dist([c + step])
        if d_new[0] < d[4]:  # d[4]: offset (0, 0)
            return c + step, h, ob_new[0]
        return c, max(0.5 * h, st.pix), obs_c[4]

    def seen_outside(self, obs, st, Q, c):
        """Fraction of the pixels of mask_visib with depth (in the frame) that lie
        outside the stencil square of the stencil camera with centre c."""
        rays = obs["seen_rays"] @ torch.as_tensor(
            self.camera_of(Q, c), device=st.device
        )
        xy = rays[:, :2] / rays[:, 2:]
        return float((xy.abs().amax(dim=1) >= st.half_width).double().mean())

    def refine_com(self, obs, st, Q, c, map_curr, ob):
        """One update of the stencil centre that aligns the centres of mass."""
        if self.centre_silhouette == "mask":
            unknown = torch.isnan(ob["avail"])
            _, mask = self.resample(obs, st, self.camera_of(Q, c), with_mask=True)
            outside = unknown & ~mask  # outside the frame
            fused = (mask | (outside & (map_curr[0] > 0))).float()
            known = mask
        else:
            fused, known = (x[0] for x in self.fuse(map_curr, ob))
        _, com = st.y_from_depth(torch.stack([fused, map_curr[0]]))
        obj = fused > 0
        observed = (obj & known).sum().item() / max(obj.sum().item(), 1)
        shift = (com[0] - com[1]).cpu().numpy() / max(observed, 0.2)
        if np.linalg.norm(shift) > st.half_width:
            shift *= st.half_width / np.linalg.norm(shift)
        c = c + shift
        return c, self.observe(obs, st, self.camera_of(Q, c))

    def misfit(self, st, p, ob):
        """RMS of the y-tone difference between the rendering at p and the fused map on
        its known pixels (background included); "mask": mean over the known pixels of
        map_avail; "visib": sum over the known pixels (mask_visib with depth and the
        free space carved from the rendering) divided by the pixels of mask_visib with
        depth, which do not depend on p."""
        z = st.depth_image(p.to(st.device, torch.float64))[0]

        def tone(m):
            y = st.s_amp * (2 * (m - st.z_lo) / (st.z_bg - st.z_lo) - 1)
            return torch.where(m > 0, y, st.s_amp)

        fused, known = self.fuse(z, ob)
        d = (tone(z) - tone(fused))[known]
        if self.observation == "mask":
            return float(d.square().mean().sqrt())
        return float((d.square().sum() / ob["seen"].sum().clamp(min=1)).sqrt())

    def pose(self, st, p, c, ob, Q):
        """R_m2c, t_m2c of the solution p with the stencil centre c."""
        _, E = st._frames(p)
        R_m2s = E[0].double().cpu().numpy().T  # model -> stencil camera S
        D = st.view(p.to(st.device))[0, 0].item()
        if self.centre == "com":
            _, c_S = st.forward(p)
            _, c_V = st.y_from_depth(self.fuse(st.depth_image(p), ob)[0])
            c_S, c_V = c_S[0].cpu().numpy(), c_V[0].cpu().numpy()
            Q_S = Q @ smallest_rotation([c_S[0], c_S[1], 1.0], [c_V[0], c_V[1], 1.0])
        else:
            Q_S = self.camera_of(Q, c)
        return Q_S @ R_m2s, Q_S @ np.array([0.0, 0.0, D])

    # --- evaluation ------------------------------------------------------------------
    def pose_errors(self, R, t, R_gt, t_gt):
        """Translation error (mm), rotation error (deg; angle of R_gt^T R, for the
        symmetry of the model that minimizes the MSSD), MSSD (mm); for a model with a
        continuous symmetry (the cylinder) also the angle of its axis (z; flip
        symmetric)."""
        x = self.eval_points
        est = x @ R.T + t
        # ground truth composed with each symmetry: R_gt (S x + t_s) + t_gt
        gt = np.einsum("ij,sxj->sxi", R_gt, x[None] @ self.sym_R.transpose(0, 2, 1)
                       + self.sym_t[:, None]) + t_gt  # fmt: skip
        dist = np.linalg.norm(gt - est[None], axis=-1).max(axis=1)
        k = int(dist.argmin())
        trace = np.trace((R_gt @ self.sym_R[k]).T @ R)
        out = {
            "t_error_mm": float(np.linalg.norm(t - t_gt)),
            "rot_error_deg": float(math.degrees(math.acos(np.clip((trace - 1) / 2, -1, 1)))),
            "mssd_mm": float(dist[k]),
            "mssd_rel_diameter": float(dist[k] / self.diameter),
        }
        if self.axis_symmetric:
            cos = abs(R[:, 2] @ R_gt[:, 2])  # z axis = symmetry axis; flip symmetric
            out["axis_error_deg"] = float(math.degrees(math.acos(min(1.0, cos))))
        return out

    def view_dict(self, st, p):
        """The view of p in degrees (and D in mm)."""
        v = st.view(p.to(st.device))[0].cpu().numpy()
        out = {"D_mm": float(v[0]), "elevation_deg": math.degrees(v[1])}
        if st.directions is not None:  # (D, e, phi, psi)
            out.update({"azimuth_deg": math.degrees(v[2]), "roll_deg": math.degrees(v[3])})
        else:  # (D, e, psi[, phi])
            out["roll_deg"] = math.degrees(v[2])
        return out

    @staticmethod
    def depth_comparison(z_sol, depth, mask, available, erode=3):
        """Statistics of dZ = Z_solution - Z_observed on the available pixels: mean |dZ|,
        median dZ; and on the interior pixels (available, eroded by `erode` pixels, so
        that pixels mixing object and background at the silhouette are excluded): a
        robust plane fit dZ = offset + tilt . (u, v) (least squares, refitted without
        the residuals beyond 3 robust standard deviations; offset at the centre of the
        pixels, tilt in mm per 100 pixels), the robust scatter around the plane
        (1.4826 * median absolute deviation) and the fraction of outliers."""
        hit = z_sol > 0
        both = available & hit
        dz_all = z_sol[both] - depth[both]
        out = {
            "available_px": int(available.sum()),
            "available_not_covered": float((available & ~hit).sum() / available.sum()),
            "silhouette_iou": float((hit & mask).sum() / (hit | mask).sum()),
        }
        if dz_all.size < 10:
            return out
        out["depth_diff_mean_mm"] = float(np.abs(dz_all).mean())
        out["depth_bias_mm"] = float(np.median(dz_all))
        interior = both.copy()
        for _ in range(erode):  # erosion with a 3 x 3 square
            m = np.pad(interior, 1)
            interior = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2]
                        & m[1:-1, 2:] & m[:-2, :-2] & m[:-2, 2:] & m[2:, :-2] & m[2:, 2:])  # fmt: skip
        rows, cols = np.nonzero(interior)
        if rows.size < 10:
            return out
        dz = z_sol[interior] - depth[interior]
        A = np.stack(
            [np.ones(dz.size), (cols - cols.mean()) / 100, (rows - rows.mean()) / 100],
            1,
        )
        keep = np.ones(dz.size, bool)
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(A[keep], dz[keep], rcond=None)
            res = dz - A @ coef
            sigma = 1.4826 * np.median(np.abs(res - np.median(res)))
            keep = np.abs(res) <= 3 * sigma
        out.update({
            "interior_px": int(dz.size),
            "plane_offset_mm": float(coef[0]),
            "plane_tilt_mm_per_100px": float(np.hypot(coef[1], coef[2])),
            "scatter_robust_mm": float(sigma),
            "outlier_fraction": float(1 - keep.mean()),
        })  # fmt: skip
        return out

    # --- main entry ------------------------------------------------------------------
    def load_instance(self, path, im_id, inst_id):
        """The data of one instance in a BOP scene directory: depth (mm), K, mask,
        mask_visib and the ground-truth pose (R, t) (None if scene_gt.json is missing or
        lists another object)."""
        path = Path(path)
        name = f"{im_id:06d}_{inst_id:06d}"
        cam = json.loads((path / "scene_camera.json").read_text())[str(im_id)]
        K = np.asarray(cam["cam_K"], float).reshape(3, 3)
        depth = read_image(path / f"depth/{im_id:06d}.tif") * cam.get(
            "depth_scale", 1.0
        )
        gt = None
        gt_file = path / "scene_gt.json"
        if gt_file.exists():
            g = json.loads(gt_file.read_text())[str(im_id)][inst_id]
            if g["obj_id"] == self.obj_id:
                gt = (np.asarray(g["cam_R_m2c"], float).reshape(3, 3),
                      np.asarray(g["cam_t_m2c"], float))  # fmt: skip
        return {
            "name": name,
            "info": {"path": str(path), "im_id": im_id, "inst_id": inst_id},
            "depth": depth,
            "K": K,
            "mask": read_image(path / f"mask/{name}.png", np.uint8) > 0,
            "mask_visib": read_image(path / f"mask_visib/{name}.png", np.uint8) > 0,
            "gt": gt,
        }

    def process_instance(self, path, im_id, inst_id, out_dir=None, frame=None, tag=""):
        """frame = (x0, x1, y0, y1): visible part of the image (default: all of it)."""
        data = self.load_instance(path, im_id, inst_id)
        return self.process(data, out_dir=out_dir, frame=frame, tag=tag)

    def process(self, data, out_dir=None, frame=None, tag=""):
        """The pose of the instance of data (load_instance, or synthetic data with the
        same keys: name, info, depth, K, mask, mask_visib, gt)."""
        t0 = time.perf_counter()
        name, K, depth = data["name"], data["K"], data["depth"]
        mask_full, mask_visib = data["mask"], data["mask_visib"]
        height, width = depth.shape
        frame = frame or (0, width, 0, height)
        in_frame = np.zeros_like(mask_full)
        in_frame[frame[2] : frame[3], frame[0] : frame[1]] = True
        mask = mask_full & in_frame
        result = {**data["info"], "frame": frame}
        x0, x1, y0, y1 = frame

        def touches(m):
            return bool(m[y0].any() or m[y1 - 1].any() or m[:, x0].any()
                        or m[:, x1 - 1].any())  # fmt: skip

        # the mask, which the observation "visib" does not use, only for the evaluation
        result["touches_border"] = touches(mask)
        result["mask_in_frame"] = float(mask.sum() / mask_full.sum())
        silhouette = mask if self.observation == "mask" else mask_visib & in_frame
        if touches(silhouette) and not self.edge_cases:
            result["status"] = "dismissed: the mask touches the image border"
            return result
        dev = self.device
        obs = {
            "depth": depth,
            "K": K,
            "frame": frame,
            "t_depth": torch.as_tensor(depth, device=dev),
            "t_mask": torch.as_tensor(mask, device=dev)
            if self.observation == "mask"
            else None,
            "t_mask_visib": torch.as_tensor(mask_visib & in_frame, device=dev),
        }
        if self.observation == "visib":  # rays (real camera) of mask_visib with depth
            seen = obs["t_mask_visib"] & (obs["t_depth"] > 0)
            v, u = (x.double() for x in torch.nonzero(seen, as_tuple=True))
            x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
            obs["seen_rays"] = torch.stack([x, y, torch.ones_like(x)], dim=1)
        if self.fill_visib:
            visib = obs["t_mask_visib"]
            obs["t_fill"] = fill_nearest(obs["t_depth"], visib & (obs["t_depth"] > 0),
                                         visib)  # fmt: skip
        rows, cols = np.nonzero(silhouette)
        d_M = [
            (cols.mean() - K[0, 2]) / K[0, 0],
            (rows.mean() - K[1, 2]) / K[1, 1],
            1.0,
        ]
        Q = smallest_rotation([0.0, 0.0, 1.0], d_M)

        trials = []
        for ci, chart in enumerate(self.charts):
            for ai, attempt in enumerate(self.attempts):
                p, c, ob = self.solve(chart, obs, Q, attempt)
                trials.append({"chart": ci, "attempt": ai, "p": p, "c": c, "ob": ob,
                               "misfit": self.misfit(chart.stencil, p, ob),
                               "residual": float(chart.solver.R_norm[0])})  # fmt: skip
        best = min(trials, key=lambda tr: tr["misfit"])
        chart = self.charts[best["chart"]]
        st, p, c, ob = chart.stencil, best["p"], best["c"], best["ob"]
        R, t = self.pose(st, p, c, ob, Q)
        if self.observation == "mask":  # object: the mask
            known = ~torch.isnan(ob["avail"])
            obj = torch.isnan(ob["avail"]) | (ob["avail"] > 0)
        else:  # object: the fused map at the solution
            fused, known = (x[0] for x in self.fuse(st.depth_image(p), ob))
            obj = fused > 0
        result.update({
            "status": "solved",
            "chart": chart.name,
            "attempt": best["attempt"],
            "trials": [{"chart": self.charts[tr["chart"]].name, "attempt": tr["attempt"],
                        "misfit": tr["misfit"], "residual": tr["residual"],
                        "p": tr["p"][0].cpu().tolist()} for tr in trials],
            "unknown_fraction": float((~known & obj).sum() / obj.sum()),
            "p": p[0].cpu().tolist(),
            "centre": c.tolist(),
            "view": self.view_dict(st, p),
            "misfit": best["misfit"],
            "R_m2c": R.ravel().tolist(),
            "t_m2c_mm": t.tolist(),
        })  # fmt: skip

        z_sol = st.render_pinhole(R, t, K, width, height).cpu().numpy()
        available = mask_visib & (depth > 0) & in_frame
        result["depth"] = self.depth_comparison(z_sol, depth, mask, available)
        if data["gt"] is not None:
            R_gt, t_gt = data["gt"]
            result["errors"] = self.pose_errors(R, t, R_gt, t_gt)
            if st.directions is None:  # the cylinder: (D, e, psi) with the flip folded
                D, e, psi, _ = view_from_pose(R_gt, t_gt)
                result["view_gt"] = {"D_mm": D, "elevation_deg": math.degrees(e),
                                     "roll_deg": math.degrees(psi)}  # fmt: skip
            else:  # p of the ground truth in each chart, and in the chosen one
                p_gt = {ch.name: ch.p_of_pose(R_gt, t_gt) for ch in self.charts}
                result["p_gt"] = p_gt[chart.name].tolist()
                result["charts_gt"] = {k: float(np.abs(v).max()) for k, v in p_gt.items()}
            z_gt = st.render_pinhole(R_gt, t_gt, K, width, height).cpu().numpy()
            result["depth_gt"] = self.depth_comparison(z_gt, depth, mask, available)
        result["time_s"] = time.perf_counter() - t0

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(z_sol.astype(np.float32), mode="F").save(
                out_dir / f"{name}{tag}_depth.tif"
            )
            self.plot(out_dir / f"{name}{tag}.png", result, st, ob, p, z_sol, depth,
                      mask_full, available, frame)  # fmt: skip
        return result

    def plot(self, file_name, result, st, ob, p, z_sol, depth, mask, available, frame):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def tones(z):
            y = st.s_amp * (2 * (z - st.z_lo) / (st.z_bg - st.z_lo) - 1)
            return np.where(z > 0, 0.5 - 0.5 * y / st.s_amp, 0.0)

        a = (ob["avail"] if self.observation == "mask" else ob["depth"]).cpu().numpy()
        rgb = np.stack([tones(np.nan_to_num(a))] * 3, -1)
        rgb[np.isnan(a)] = (0.85, 0.2, 0.75)  # unknown (no depth): magenta
        fused, known = (x[0] for x in self.fuse(st.depth_image(p.to(st.device)), ob))
        sel = mask | (z_sol > 0)
        rows, cols = np.nonzero(sel)
        r0, r1 = max(rows.min() - 20, 0), min(rows.max() + 21, mask.shape[0])
        c0, c1 = max(cols.min() - 20, 0), min(cols.max() + 21, mask.shape[1])
        diff = np.where(available & (z_sol > 0), z_sol - depth, np.nan)[r0:r1, c0:c1]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained",
                                 facecolor="#fcfcfb")  # fmt: skip
        axes[0].imshow(rgb)
        if self.observation == "mask":
            axes[0].set_title(
                "map_avail (stencil camera; magenta: unknown)", fontsize=9
            )
        else:
            axes[0].contour(ob["seen"].cpu().numpy(), levels=[0.5], colors=["#eb6834"],
                            linewidths=0.8)  # fmt: skip
            axes[0].set_title("observed depth (stencil camera; magenta: none; orange "
                              "outline: mask_visib with depth)", fontsize=9)  # fmt: skip
        axes[1].imshow(tones(fused.cpu().numpy()), cmap="gray", vmin=0, vmax=1)
        axes[1].contour((~known).cpu().numpy(), levels=[0.5], colors=["#d93cbf"],
                        linewidths=0.8)  # fmt: skip
        axes[1].set_title("fused map at the solution (outline: unknown)", fontsize=9)
        # signed difference: diverging scale, neutral (white) at 0
        im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-2, vmax=2)
        axes[2].contour((z_sol > 0)[r0:r1, c0:c1], levels=[0.5], colors=["#eb6834"],
                        linewidths=0.8)  # fmt: skip
        axes[2].contour(mask[r0:r1, c0:c1], levels=[0.5], colors=["#52514e"],
                        linewidths=0.8, linestyles="dashed")  # fmt: skip
        x0, x1, y0, y1 = frame
        if frame != (0, mask.shape[1], 0, mask.shape[0]):
            axes[2].add_patch(plt.Rectangle((x0 - c0 - 0.5, y0 - r0 - 0.5), x1 - x0, y1 - y0,
                                            fill=False, ec="#0b0b0b", lw=1.2, ls=":"))  # fmt: skip
        fig.colorbar(im, ax=axes[2], label="Z solution - Z sensor (mm)", shrink=0.8)
        d = result["depth"]
        if "depth_bias_mm" in d:
            title = (f"camera image (orange: solution, dashed: mask)\nplane offset "
                     f"{d.get('plane_offset_mm', float('nan')):+.2f} mm, scatter "
                     f"{d.get('scatter_robust_mm', float('nan')):.2f} mm")  # fmt: skip
            axes[2].set_title(title, fontsize=9)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        v = result["view"]
        label = (f"image {result['im_id']}, instance {result['inst_id']}"
                 if "im_id" in result else result.get("label", ""))  # fmt: skip
        head = (f"{label} ({result['chart']}): unknown {result['unknown_fraction']:.0%}; "
                f"D {v['D_mm']:.1f} mm, elevation {v['elevation_deg']:+.1f} deg"
                + (f", azimuth {v['azimuth_deg']:+.1f} deg" if "azimuth_deg" in v else "")
                + f", roll {v['roll_deg']:+.1f} deg")  # fmt: skip
        if "errors" in result:
            e = result["errors"]
            angle = (f"axis {e['axis_error_deg']:.2f} deg" if "axis_error_deg" in e
                     else f"rotation {e['rot_error_deg']:.2f} deg")  # fmt: skip
            head += (f"; errors: t {e['t_error_mm']:.2f} mm, {angle}, "
                     f"MSSD {e['mssd_mm']:.2f} mm")  # fmt: skip
        fig.suptitle(head, x=0.01, ha="left", fontsize=10)
        fig.savefig(file_name, dpi=100, facecolor="#fcfcfb")
        plt.close(fig)
