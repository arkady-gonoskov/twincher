"""Convert the depth maps of a BOP scene into 8-bit PNG images for viewing.

BOP depth maps are single-channel images (ITODD: float32 TIFF; most other datasets:
uint16 PNG) whose values times `depth_scale` (from scene_camera.json) are Z in mm;
0 marks pixels without a measurement. Image viewers show them almost uniformly dark
or white. This script maps Z to tones with one Z range for all images of the
directory, so that equal Z gives equal tones in all images:

- gray:  near = white (255), far = dark gray (32);
- color: colormap turbo, near = red, far = blue;
- shade: shaded relief (Lambertian, light from the upper left), which makes noise,
         steps and artifacts of the surfaces visible; needs scene_camera.json.

Pixels without a measurement are black in all modes.

Usage:
    python -I depth_to_png.py DEPTH_DIR OUT_DIR [--mode gray|color|shade]
                              [--range ZMIN ZMAX]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# unit vector towards the light, camera frame (x right, y down, z forward)
LIGHT = np.array([-1.0, -1.0, -2.0]) / np.sqrt(6.0)


def load_cameras(depth_dir):
    """Camera data per image id from scene_camera.json of the scene, or None."""
    path = depth_dir.parent / "scene_camera.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def read_depth(path, cameras):
    """Z in mm (in raw units without cameras) and the mask of measured pixels."""
    z = np.asarray(Image.open(path), dtype=np.float64)
    if cameras is not None:
        z *= cameras[int(path.stem)]["depth_scale"]
    return z, np.isfinite(z) & (z > 0)


def tones(x, valid):
    """Map x in [0, 1] to 32..255; pixels without a measurement get 0."""
    return np.where(valid, np.rint(32 + 223 * np.clip(x, 0, 1)), 0).astype(np.uint8)


def relief(z, valid, cam_K):
    """Lambertian shading of the surface given by Z, and the mask of shaded pixels."""
    fx, _, cx, _, fy, cy = cam_K[:6]
    v, u = np.indices(z.shape, dtype=np.float64)
    p = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z])  # 3D points in mm
    p_v, p_u = np.gradient(p, axis=(1, 2))
    n = np.cross(p_v, p_u, axis=0)  # normal pointing towards the camera
    with np.errstate(invalid="ignore", divide="ignore"):
        n /= np.linalg.norm(n, axis=0)
    # the central differences need both neighbours
    m = np.pad(valid, 1)
    valid = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return np.tensordot(LIGHT, n, axes=1), valid


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("depth_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--mode", choices=["gray", "color", "shade"], default="gray")
    parser.add_argument(
        "--range",
        nargs=2,
        type=float,
        metavar=("ZMIN", "ZMAX"),
        help="Z range in mm mapped to the tones "
        "(default: from the 0.1 and 99.9 percentiles of Z of all images)",
    )
    args = parser.parse_args()

    files = sorted(
        p for p in args.depth_dir.iterdir() if p.suffix in (".tif", ".tiff", ".png")
    )
    if not files:
        parser.error(f"no depth maps (.tif, .png) in {args.depth_dir}")
    if args.out_dir.resolve() == args.depth_dir.resolve():
        parser.error("OUT_DIR must differ from DEPTH_DIR")
    cameras = load_cameras(args.depth_dir)
    if cameras is None:
        if args.mode == "shade":
            parser.error("--mode shade needs scene_camera.json next to DEPTH_DIR")
        print("No scene_camera.json next to DEPTH_DIR: Z in raw units")

    if args.mode != "shade":
        if args.range:
            z_min, z_max = args.range
        else:
            q = [
                np.percentile(z[valid], [0.1, 99.9])
                for z, valid in (read_depth(path, cameras) for path in files)
            ]
            z_min, z_max = min(r[0] for r in q), max(r[1] for r in q)
        print(f"Z range: {z_min:.1f} (near) to {z_max:.1f} (far)")
    if args.mode == "color":
        import matplotlib

        cmap = matplotlib.colormaps["turbo_r"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        z, valid = read_depth(path, cameras)
        if args.mode == "shade":
            s, valid = relief(z, valid, cameras[int(path.stem)]["cam_K"])
            image = tones(s, valid)
        else:
            x = (z - z_min) / (z_max - z_min)
            if args.mode == "gray":
                image = tones(1 - x, valid)
            else:
                image = np.rint(255 * cmap(np.clip(x, 0, 1))[..., :3]).astype(np.uint8)
                image[~valid] = 0
        Image.fromarray(image).save(args.out_dir / f"{path.stem}.png")
    print(f"{len(files)} images written to {args.out_dir}")


if __name__ == "__main__":
    main()
