"""Whole-scene demonstration of the solutions (README Part III): one row of five panels
per scene image, with all solved instances of the objects in it.

Panels (labels above):
- [Source]: the measured depth map (viridis, fixed range DEPTH_RANGE; no depth: black),
  outlines of mask_visib (magenta), or of the amodal mask (green) for --mode mask;
- [Solution]: the depth map of the solution: all solved instances rendered with the real
  camera (z-buffer) in front of the background plane; outlines of mask_visib (magenta)
  and, on top, of the whole silhouette of each solved instance (black, dashed: like the
  amodal mask, including the parts hidden by other instances);
- [Mismatch]: Z_solution - Z_sensor where the sensor has depth (RdBu, -5..+5 mm; no
  depth: light gray), outlines of the amodal mask (green) and, on top, of the solution
  (black, dashed);
- [Reconstruction]: the solution (instances, albedo 0.9, and plane, 0.45) shaded with
  ambient light (0.25) and a light from above (along the plane normal, tilted 40
  degrees towards the top of the image), seen from the real camera;
- [Ground Truth]: the gray image of the scene (gray/).
The background plane: 1 / Z = a x + b y + c in normalized image coordinates (a plane in
3D), fitted to 3 random pixels with depth outside all masks, for 400 triples; the median
of each coefficient over the triples (robust against points on other objects).

The solutions come from the results of run_m3.py (RESULTS: the runs with and without
the amodal mask for each object). The colour bars are written once (colorbars.png).

Usage:
    python scene_demo.py --mode mask [--images 643 233 655] [--out-dir DIR]
    python scene_demo.py --mode visib          (all images of both objects)
"""

import argparse
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patheffects
from PIL import Image

from lidar_fit import LidarFit, read_image
from run_m3 import CASES

HERE = Path(__file__).resolve().parent
DEPTH_RANGE = (640.0, 770.0)  # mm, all scenes of objects 12 and 26
MISMATCH_RANGE = (-5.0, 5.0)  # mm
RESULTS = {  # object: {mode: (configuration, run directory)}
    12: {"mask": ("lidarfit_obj12_m3_tent_mask.json", "output_m3/obj12_tent_mask"),
         "visib": ("lidarfit_obj12_m3_tent_visib.json", "output_m3/obj12_tent_visib")},
    26: {"mask": ("lidarfit_obj26_m3_mask.json", "output_m3/obj26_mask"),
         "visib": ("lidarfit_obj26_m3_visib.json", "output_m3/obj26_visib")},
}  # fmt: skip
SURFACE = "#fcfcfb"
MAGENTA, GREEN = "#d93cbf", "#13a538"


def background_plane(depth, K, exclude, rng, n_triples=400):
    """Coefficients (a, b, c) of 1 / Z = a x + b y + c, x, y normalized image
    coordinates: the median over random triples of pixels with depth outside exclude."""
    v, u = np.nonzero((depth > 0) & ~exclude)
    x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    inv = 1.0 / depth[v, u]
    coef = []
    for _ in range(n_triples):
        k = rng.choice(len(u), 3, replace=False)
        A = np.stack([x[k], y[k], np.ones(3)], 1)
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        coef.append(np.linalg.solve(A, inv[k]))
    return np.median(np.array(coef), axis=0)


def plane_depth(coef, K, shape):
    h, w = shape
    x = (np.arange(w) - K[0, 2]) / K[0, 0]
    y = (np.arange(h) - K[1, 2]) / K[1, 1]
    return 1.0 / (coef[0] * x[None, :] + coef[1] * y[:, None] + coef[2])


def shade(z, K, up, albedo, ambient=0.25, tilt_deg=40.0):
    """Gray shading of a depth map: albedo x (ambient light plus a light from above:
    the plane normal `up`, camera frame, tilted towards the top of the image)."""
    h, w = z.shape
    x = (np.arange(w) - K[0, 2]) / K[0, 0]
    y = (np.arange(h) - K[1, 2]) / K[1, 1]
    P = np.stack([x[None, :] * z, y[:, None] * z, z], -1)  # 3D points
    n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)
    n *= -np.sign(n[..., 2:])  # towards the camera
    t = np.radians(tilt_deg)
    light = np.cos(t) * up + np.sin(t) * np.array([0.0, -1.0, 0.0])
    light /= np.linalg.norm(light)
    return albedo * (ambient + (1 - ambient) * np.clip(n @ light, 0, 1))


def contour(ax, m, color, lw=1.0, ls="solid", zorder=2):
    """An outline with a thin white halo (visible on any background)."""
    if m.any():
        cs = ax.contour(m, levels=[0.5], colors=[color], linewidths=lw,
                        linestyles=[ls], zorder=zorder)  # fmt: skip
        cs.set_path_effects([patheffects.withStroke(linewidth=lw + 0.9,
                                                    foreground="white", alpha=0.7)])  # fmt: skip


def solution_contour(ax, m):
    """The silhouette of a solution: black, dashed, above everything else."""
    contour(ax, m, "black", lw=0.9, ls=(0, (3, 2)), zorder=10)


def scene(fits, split, path, im, mode, rng):
    """The five panels of one scene image (dict of arrays)."""
    cam = json.loads((path / "scene_camera.json").read_text())[str(im)]
    K = np.asarray(cam["cam_K"], float).reshape(3, 3)
    depth = read_image(path / f"depth/{im:06d}.tif") * cam.get("depth_scale", 1.0)
    h, w = depth.shape
    z = np.full((h, w), np.inf)
    owner = np.full((h, w), -1)
    silhouettes = []
    masks, visibs = [], []
    for k, (lf, r) in enumerate(fits):
        name = f"{im:06d}_{r['inst_id']:06d}"
        masks.append(read_image(path / f"mask/{name}.png", np.uint8) > 0)
        visibs.append(read_image(path / f"mask_visib/{name}.png", np.uint8) > 0)
        R, t = np.array(r["R_m2c"]).reshape(3, 3), np.array(r["t_m2c_mm"])
        zk = lf.charts[0].stencil.render_pinhole(R, t, K, w, h).cpu().numpy()
        front = (zk > 0) & (zk < z)
        z[front], owner[front] = zk[front], k
        silhouettes.append(zk > 0)
    any_mask = np.any(masks, axis=0) if masks else np.zeros((h, w), bool)
    coef = background_plane(depth, K, any_mask, rng)
    plane = plane_depth(coef, K, (h, w))
    solution = np.where(np.isfinite(z), z, plane)
    up = -coef / np.linalg.norm(coef)  # plane normal (camera frame) towards the camera
    up *= -np.sign(up[2]) if up[2] != 0 else 1.0
    gray = np.asarray(Image.open(path / f"gray/{im:06d}.tif"), float)
    return {"depth": depth, "masks": masks, "visibs": visibs, "owner": owner,
            "silhouettes": silhouettes,
            "solution": solution, "gray": gray,
            "shaded": shade(solution, K, up, np.where(owner >= 0, 0.9, 0.45)),
            "n": len(fits)}  # fmt: skip


def plot_row(file_name, d, title, mode):
    viridis = plt.get_cmap("viridis").copy()
    viridis.set_bad("black")
    rdbu = plt.get_cmap("RdBu").copy()
    rdbu.set_bad("#d9d9d6")
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.4), facecolor=SURFACE,
                             layout="constrained")  # fmt: skip
    depth = np.where(d["depth"] > 0, d["depth"], np.nan)
    diff = np.where(d["depth"] > 0, d["solution"] - d["depth"], np.nan)
    labels = [
        "[Source]",
        "[Solution]",
        "[Mismatch]",
        "[Reconstruction]",
        "[Ground Truth]",
    ]
    axes[0].imshow(depth, cmap=viridis, vmin=DEPTH_RANGE[0], vmax=DEPTH_RANGE[1])
    for m in d["masks"] if mode == "mask" else d["visibs"]:
        contour(axes[0], m, GREEN if mode == "mask" else MAGENTA)
    axes[1].imshow(
        d["solution"], cmap=viridis, vmin=DEPTH_RANGE[0], vmax=DEPTH_RANGE[1]
    )
    for m in d["visibs"]:
        contour(axes[1], m, MAGENTA)
    for m in d["silhouettes"]:
        solution_contour(axes[1], m)
    axes[2].imshow(diff, cmap=rdbu, vmin=MISMATCH_RANGE[0], vmax=MISMATCH_RANGE[1])
    for m in d["masks"]:
        contour(axes[2], m, GREEN)
    for m in d["silhouettes"]:
        solution_contour(axes[2], m)
    axes[3].imshow(d["shaded"], cmap="gray", vmin=0, vmax=1)
    axes[4].imshow(d["gray"], cmap="gray", vmin=0, vmax=255)
    for ax, label in zip(axes, labels):
        ax.set_title(label, fontsize=12)
        ax.set_axis_off()
    fig.suptitle(title, x=0.005, ha="left", fontsize=11)
    fig.savefig(file_name, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def colorbars(file_name):
    fig, axes = plt.subplots(2, 1, figsize=(7, 1.9), facecolor=SURFACE,
                             layout="constrained")  # fmt: skip
    for ax, cmap, rng, label in (
        (axes[0], "viridis", DEPTH_RANGE, "depth Z (mm) [Source], [Solution]"),
        (axes[1], "RdBu", MISMATCH_RANGE, "Z solution - Z sensor (mm) [Mismatch]"),
    ):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(*rng))
        cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
        cb.set_label(label)
    fig.savefig(file_name, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["mask", "visib"], required=True)
    parser.add_argument("--images", type=int, nargs="+")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out = args.out_dir or HERE / f"output_m3/scenes_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)
    colorbars(out / "colorbars.png")
    for obj, modes in RESULTS.items():
        cfg, run = modes[args.mode]
        if not (HERE / run / "results.json").exists():
            print(f"object {obj}: no results in {run}")
            continue
        results = json.loads((HERE / run / "results.json").read_text())
        lf = LidarFit(HERE / cfg)
        images = sorted({(r["split"], r["im_id"]) for r in results})
        for split, im in images:
            if args.images and im not in args.images:
                continue
            path = next(p for s, p, i, _ in CASES[obj] if i == im)
            fits = [(lf, r) for r in results if r["im_id"] == im]
            d = scene(
                fits, split, path, im, args.mode, np.random.default_rng(args.seed)
            )
            title = (f"{split} image {im}: {len(fits)} instance(s) of object {obj}, "
                     + ("with" if args.mode == "mask" else "without") + " the amodal mask")  # fmt: skip
            plot_row(out / f"{split}_{im:06d}.png", d, title, args.mode)
            del d
            gc.collect()  # matplotlib otherwise accumulates memory (killed after ~10 scenes)
            print(f"{split}_{im:06d}.png", flush=True)
        del lf
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
