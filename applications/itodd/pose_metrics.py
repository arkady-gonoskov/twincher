"""Per-instance metrics of README Part III (milestone 3): data availability, agreement of
the solution with the depth and the masks, and the pose-error metrics of BOP (ITODD).

Availability (measured in the camera image, independent of the method): fractions of the
pixels of the amodal mask (BOP `mask`, the full silhouette including occluded parts) that
are available (in mask_visib and with depth), holes (in mask_visib without depth: glossy
or dark surfaces, shadows of the structured-light sensor) and occluded (in mask but not
in mask_visib). unavailable = holes + occluded = 1 - available. Parts outside the image
are not in the BOP masks and do not count.

Agreement with the observation (no ground truth needed), for the depth map z_sol of the
solution rendered with the real camera:
- depth_mismatch_mm: mean |Z_solution - Z_sensor| over the pixels where both show the
  object (mask_visib with depth and covered by the solution);
- silhouette_mismatch: 1 - IoU of the silhouette of the solution and the amodal mask;
  silhouette_uncovered / silhouette_excess: the fractions of the mask not covered by the
  solution, and of the solution outside the mask (relative to the mask);
- the robust plane fit of LidarFit.depth_comparison (offset, tilt, scatter).

Pose errors (ground truth required: validation images, synthetic data), as defined by BOP
(Hodan et al., BOP challenge 2019/2020; bop_toolkit, not validated against it):
- VSD with delta = 15 mm, tau = 0.05..0.5 x diameter, visibility "bop19" (distances
  along the rays, the test depth decides visibility; missing test depth counts as
  visible); correct if e_VSD < theta, theta = 0.05..0.5;
- MSSD: maximum vertex distance, minimized over the symmetries; correct below
  0.05..0.5 x diameter;
- MSPD: maximum projection distance of the vertices (pixels), minimized over the
  symmetries; correct below 5r..50r pixels, r = image width / 640;
- per instance, score_X = the fraction of the thresholds (and taus) at which it is
  correct; the mean of score_X over instances is the average recall AR_X of BOP, and
  AR = (AR_VSD + AR_MSSD + AR_MSPD) / 3.
"""

import numpy as np

VSD_DELTA = 15.0  # mm (BOP 2019, ITODD)
VSD_TAUS = np.arange(0.05, 0.51, 0.05)  # x diameter
THETAS = np.arange(0.05, 0.51, 0.05)  # VSD error; MSSD x diameter
MSPD_STEPS = np.arange(5, 51, 5)  # x r pixels, r = width / 640


def availability(depth, mask, mask_visib):
    n = max(int(mask.sum()), 1)
    available = mask_visib & (depth > 0)
    return {
        "mask_px": int(mask.sum()),
        "available": float((available & mask).sum() / n),
        "holes": float((mask_visib & ~(depth > 0) & mask).sum() / n),
        "occluded": float((mask & ~mask_visib).sum() / n),
        "unavailable": float(1 - (available & mask).sum() / n),
    }


def agreement(z_sol, depth, mask, mask_visib):
    sol = z_sol > 0
    both = sol & mask_visib & (depth > 0)
    n = max(int(mask.sum()), 1)
    out = {
        "silhouette_iou": float((sol & mask).sum() / max((sol | mask).sum(), 1)),
        "silhouette_uncovered": float((mask & ~sol).sum() / n),
        "silhouette_excess": float((sol & ~mask).sum() / n),
        "depth_px": int(both.sum()),
    }
    out["silhouette_mismatch"] = 1 - out["silhouette_iou"]
    if both.any():
        out["depth_mismatch_mm"] = float(np.abs(z_sol[both] - depth[both]).mean())
    return out


def dist_image(depth, K):
    """Distances from the camera centre along the rays (BOP: depth_im_to_dist_im)."""
    h, w = depth.shape
    x = (np.arange(w) - K[0, 2]) / K[0, 0]
    y = (np.arange(h) - K[1, 2]) / K[1, 1]
    return depth * np.sqrt(x[None, :] ** 2 + y[:, None] ** 2 + 1)


def vsd_errors(z_est, z_gt, depth_test, K, diameter):
    """e_VSD for each tau of VSD_TAUS (visibility "bop19")."""
    d_est, d_gt = dist_image(z_est, K), dist_image(z_gt, K)
    d_test = dist_image(depth_test, K)
    visib_gt = ((d_gt - d_test <= VSD_DELTA) | (d_test == 0)) & (d_gt > 0)
    visib_est = ((d_est - d_test <= VSD_DELTA) | (d_test == 0)) & (d_est > 0)
    visib_est |= visib_gt & (d_est > 0)
    inter, union = visib_gt & visib_est, visib_gt | visib_est
    n_union = union.sum()
    if n_union == 0:
        return np.ones(len(VSD_TAUS))
    dists = np.abs(d_gt[inter] - d_est[inter]) / diameter
    n_comp = n_union - inter.sum()
    return np.array([((dists >= tau).sum() + n_comp) / n_union for tau in VSD_TAUS])


def mspd(R, t, R_gt, t_gt, K, points, sym_R, sym_t):
    """Maximum projection distance (pixels), minimized over the symmetries."""

    def project(x):
        x = x @ K.T
        return x[..., :2] / x[..., 2:]

    est = project(points @ R.T + t)
    gt = np.einsum("ij,sxj->sxi", R_gt, points[None] @ sym_R.transpose(0, 2, 1)
                   + sym_t[:, None]) + t_gt  # fmt: skip
    return float(np.linalg.norm(project(gt) - est[None], axis=-1).max(axis=1).min())


def bop_scores(e_vsd, mssd_mm, mspd_px, diameter, width):
    s_vsd = float(np.mean(e_vsd[:, None] < THETAS[None, :]))
    s_mssd = float(np.mean(mssd_mm < THETAS * diameter))
    s_mspd = float(np.mean(mspd_px < MSPD_STEPS * width / 640))
    return {"score_vsd": s_vsd, "score_mssd": s_mssd, "score_mspd": s_mspd,
            "score_ar": (s_vsd + s_mssd + s_mspd) / 3}  # fmt: skip


def instance_metrics(lf, st, data, R, t):
    """All metrics of one solved instance (lf: LidarFit, st: a stencil of its model,
    data: LidarFit.load_instance or synthetic, R, t: the solution)."""
    depth, K = data["depth"], data["K"]
    mask, visib = data["mask"], data["mask_visib"]
    h, w = depth.shape
    z_sol = st.render_pinhole(R, t, K, w, h).cpu().numpy()
    out = {**availability(depth, mask, visib), **agreement(z_sol, depth, mask, visib)}
    if data.get("gt") is not None:
        R_gt, t_gt = data["gt"]
        z_gt = st.render_pinhole(R_gt, t_gt, K, w, h).cpu().numpy()
        e = lf.pose_errors(R, t, R_gt, t_gt)
        e_vsd = vsd_errors(z_sol, z_gt, depth, K, lf.diameter)
        e_mspd = mspd(R, t, R_gt, t_gt, K, lf.eval_points, lf.sym_R, lf.sym_t)
        out.update({"vsd": e_vsd.tolist(), "mssd_mm": e["mssd_mm"], "mspd_px": e_mspd,
                    **bop_scores(e_vsd, e["mssd_mm"], e_mspd, lf.diameter, w)})  # fmt: skip
    return out
