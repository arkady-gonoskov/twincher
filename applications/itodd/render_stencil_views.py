"""Depth maps and probe values of the stencil views, for visual review.

For n values of each of p0, p1, p2 in [-1, 1], writes OUT_DIR/views/<i0><i1><i2>_...png:
left, the depth map (tones as y: near = white, far = dark gray; no hit = black) with the
probe grid centred at the centre of mass of the mask (+), the probe centres as dots
(orange: the probe sees the object, blue: background); right, the probe values y as
the twincher gets them (same tones, background black). OUT_DIR/overview/p0_<i0>.png
shows the views of one p0 value (rows p1, columns p2).

Usage:
    python render_stencil_views.py [OUT_DIR] [--n 5]
"""

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from depth_stencil import DepthStencil
from itodd_data import model

MODEL = model(12)
SEEN, BACKGROUND, CENTRE = (255, 140, 0), (70, 130, 255), (255, 60, 60)


def tones(y, s_amp):
    """Gray tones of y values: -s_amp -> 255, below s_amp -> down to 32, s_amp -> 0."""
    t = 32 + 223 * (s_amp - y) / (2 * s_amp)
    return np.where(y < s_amp, np.rint(t), 0).astype(np.uint8)


def view_image(st, z, y, centre, view, p, font):
    """RGB image of one view: depth map with probes | probe values, and a caption."""
    s_amp, res, n = st.s_amp, st.res, st.n_grid
    y_pix = s_amp * (2 * (z - st.z_lo) / (st.z_bg - st.z_lo) - 1)
    left = Image.fromarray(tones(np.where(z > 0, y_pix, s_amp), s_amp)).convert("RGB")
    draw = ImageDraw.Draw(left)
    cx, cy = (centre + st.half_width) / st.pix  # pixel units
    grid = np.empty(st.n_y)
    grid[st.grid_index] = y  # probe values in row-major grid order
    for k in range(st.n_y):
        u = cx + (k % n - 0.5 * (n - 1)) * st.spacing
        v = cy + (k // n - 0.5 * (n - 1)) * st.spacing
        colour = SEEN if grid[k] < s_amp else BACKGROUND
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=colour)
    draw.line((cx - 8, cy, cx + 8, cy), fill=CENTRE, width=2)
    draw.line((cx, cy - 8, cx, cy + 8), fill=CENTRE, width=2)
    right = Image.fromarray(tones(grid.reshape(n, n), s_amp)).convert("RGB")
    right = right.resize((res, res), Image.NEAREST)
    image = Image.new("RGB", (2 * res + 8, res), (255, 255, 255))
    image.paste(left, (0, 0))
    image.paste(right, (res + 8, 0))
    D, e, psi = view
    text = (
        f"p = ({p[0]:+.1f}, {p[1]:+.1f}, {p[2]:+.1f})\n"
        f"D = {D:.0f} mm, elevation = {math.degrees(e):+.1f}°, "
        f"roll = {math.degrees(psi):+.1f}°"
    )
    ImageDraw.Draw(image).multiline_text((6, 4), text, fill=(255, 255, 255), font=font)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "out_dir", nargs="?", type=Path,
        default=Path(__file__).parent / "output_stencil_views",
    )  # fmt: skip
    parser.add_argument("--n", type=int, default=5, help="values per parameter")
    args = parser.parse_args()

    st = DepthStencil(MODEL)
    values = np.linspace(-1, 1, args.n)
    index = list(itertools.product(range(args.n), repeat=3))
    p = torch.tensor([[values[i] for i in ijk] for ijk in index], dtype=torch.float64)
    views = st.view(p).numpy()
    z = st.depth_image(p).cpu().numpy()
    y, centre = (a.cpu().numpy() for a in st.forward(p))

    # the probe grid must contain the object: its outer ring sees only background
    ring = np.ones((st.n_grid, st.n_grid), bool)
    ring[1:-1, 1:-1] = False
    seen = y < st.s_amp
    print(
        f"probes seeing the object per view: {seen.sum(1).min()}..{seen.sum(1).max()}"
    )
    on_ring = ring.ravel()[st.grid_index]
    print(f"outer-ring probes seeing the object: {seen[:, on_ring].sum()}")
    off = np.abs(centre).max() / st.pix
    print(f"largest distance of the centre of mass from the axis: {off:.2f} pixels")

    font = ImageFont.load_default(size=14)
    views_dir, overview_dir = args.out_dir / "views", args.out_dir / "overview"
    for d in (views_dir, overview_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()
    images = {}
    for c, ijk in enumerate(index):
        image = view_image(st, z[c], y[c], centre[c], views[c], p[c], font)
        images[ijk] = image
        name = "".join(map(str, ijk)) + "_p0={:+.1f}_p1={:+.1f}_p2={:+.1f}.png"
        image.save(views_dir / name.format(*p[c].tolist()))

    tile_w, tile_h, head = 416, 200, 40
    for i0 in range(args.n):
        sheet = Image.new(
            "RGB", (args.n * tile_w, head + args.n * tile_h), (255, 255, 255)
        )
        for i1, i2 in itertools.product(range(args.n), repeat=2):
            thumb = images[(i0, i1, i2)].resize((tile_w - 8, tile_h - 4), Image.LANCZOS)
            sheet.paste(thumb, (i2 * tile_w, head + i1 * tile_h))
        D = views[index.index((i0, 0, 0))][0]
        ImageDraw.Draw(sheet).text(
            (6, 6),
            f"p0 = {values[i0]:+.1f} (D = {D:.0f} mm); rows: p1 = -1..+1 (elevation), "
            f"columns: p2 = -1..+1 (roll); each: depth map | probe values",
            fill=(0, 0, 0),
            font=ImageFont.load_default(size=16),
        )
        sheet.save(overview_dir / f"p0_{i0}.png")
    print(f"{len(index)} views in {views_dir}, {args.n} overviews in {overview_dir}")


if __name__ == "__main__":
    main()
