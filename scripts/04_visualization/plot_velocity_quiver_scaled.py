#!/usr/bin/env python3
"""
Create clearer static quiver plots from a binned velocity field.

This script is meant for report figures. It enlarges arrows only visually.
The numeric values in the CSV remain unchanged.

Example:
python scripts/04_visualization/plot_velocity_quiver_scaled.py \
  --csv Tribolium_Daten/velocity_fields_540_570/multi_existing_labels_540_570/binned_velocity_field.csv \
  --image-dir Tribolium_Daten/cylinder3_projections \
  --start-frame 540 \
  --times 0,10,20,29 \
  --downscale 2 \
  --vector-scale 10 \
  --max-speed 30 \
  --out-dir plots/final_analysis/quiver_scaled/multi_existing_labels_540_570
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def find_images(image_dir: Path) -> list[Path]:
    paths = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(sorted(image_dir.glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return sorted(paths)


def read_image(path: Path) -> np.ndarray:
    try:
        from skimage import io
        img = io.imread(path)
    except Exception:
        from imageio.v3 import imread
        img = imread(path)
    if img.ndim > 2:
        img = img[..., 0]
    return img


def robust_limits(img: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(img, [1, 99])
    if lo == hi:
        lo, hi = float(img.min()), float(img.max())
    return float(lo), float(hi)


def parse_times(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--times", default="0,10,20,29")
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--vector-scale", type=float, default=10.0)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-speed", type=float, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title-prefix", default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {"t", "y", "x", "dy", "dx", "speed", "count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    df = df[df["count"] >= args.min_count].copy()
    if args.max_speed is not None:
        df = df[df["speed"] <= args.max_speed].copy()

    image_paths = None
    if args.image_dir:
        image_paths = find_images(Path(args.image_dir).expanduser().resolve())

    times = parse_times(args.times)
    title_prefix = args.title_prefix or csv_path.parent.name

    for t in times:
        frame = df[df["t"].astype(int) == int(t)].copy()
        if frame.empty:
            print(f"Skipping t={t}: no vectors")
            continue

        fig, ax = plt.subplots(figsize=(8, 10))

        if image_paths is not None:
            original_frame = args.start_frame + t
            if original_frame >= len(image_paths):
                raise IndexError(
                    f"Original frame {original_frame} is outside image list of length {len(image_paths)}"
                )
            img = read_image(image_paths[original_frame])
            if args.downscale > 1:
                img_show = img[::args.downscale, ::args.downscale]
            else:
                img_show = img

            lo, hi = robust_limits(img_show)
            ax.imshow(
                img_show,
                cmap="gray",
                vmin=lo,
                vmax=hi,
                origin="upper",
                extent=[0, img.shape[1], img.shape[0], 0],
                alpha=0.75,
            )

        # Important: dx/dy are multiplied directly. scale=1 means that the
        # displayed arrow length is dx*vector_scale in data coordinates.
        q = ax.quiver(
            frame["x"],
            frame["y"],
            frame["dx"] * args.vector_scale,
            frame["dy"] * args.vector_scale,
            frame["speed"],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0045,
            headwidth=4.5,
            headlength=6.0,
            headaxislength=5.0,
            minlength=0.15,
        )
        cbar = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("speed [px/frame]")

        ax.set_title(
            f"{title_prefix}: t={t} "
            f"(original {args.start_frame + t}→{args.start_frame + t + 1}, arrows ×{args.vector_scale:g})"
        )
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        ax.set_xlim(0, 1061)
        ax.set_ylim(1612, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.20)

        out_path = out_dir / f"{title_prefix}_t{t:03d}_vectors_x{args.vector_scale:g}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
