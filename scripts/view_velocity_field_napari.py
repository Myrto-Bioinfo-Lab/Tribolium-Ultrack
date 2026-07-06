#!/usr/bin/env python3
"""
Open a binned velocity field in napari.

The numeric data remain in px/frame. The option --vector-scale only enlarges
arrows for visualization, because biological movements of 1-5 px/frame can be
almost invisible on a 1600 px image.

Examples
--------
540-570 window with raw images:
python python_scripts/view_velocity_field_napari.py \
  --csv Tribolium_Daten/velocity_fields_540_570/multi_existing_labels_540_570/binned_velocity_field.csv \
  --image-dir Tribolium_Daten/cylinder3_projections \
  --start-frame 540 \
  --end-frame 570 \
  --downscale 2 \
  --vector-scale 10

Full-frame selected run with downscaled raw images:
python python_scripts/view_velocity_field_napari.py \
  --csv Tribolium_Daten/velocity_fields_full/multi_cellpose_variants_no_baseline_all_frames/binned_velocity_field.csv \
  --image-dir Tribolium_Daten/cylinder3_projections \
  --start-frame 0 \
  --end-frame 570 \
  --downscale 4 \
  --vector-scale 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def read_image(path: Path) -> np.ndarray:
    try:
        from skimage import io
        img = io.imread(path)
    except Exception:
        from imageio.v3 import imread
        img = imread(path)

    if img.ndim > 2:
        # Keep the first channel if RGB/RGBA is encountered.
        img = img[..., 0]
    return img


def find_images(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(sorted(image_dir.glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return sorted(paths)


def load_image_stack(
    image_dir: Path,
    start_frame: int,
    end_frame: int,
    downscale: int,
) -> np.ndarray:
    paths = find_images(image_dir)
    if end_frame >= len(paths):
        raise IndexError(
            f"Requested end frame {end_frame}, but only {len(paths)} images were found."
        )

    selected = paths[start_frame : end_frame + 1]
    frames = []
    for p in selected:
        img = read_image(p)
        if downscale > 1:
            img = img[::downscale, ::downscale]
        frames.append(img)

    stack = np.stack(frames, axis=0)
    return stack


def make_napari_vectors(
    binned: pd.DataFrame,
    vector_scale: float,
    min_count: int,
    max_speed: float | None,
) -> np.ndarray:
    required = {"t", "y", "x", "dy", "dx", "count", "speed"}
    missing = required - set(binned.columns)
    if missing:
        raise ValueError(f"Missing columns in velocity CSV: {sorted(missing)}")

    df = binned.copy()
    df = df[df["count"] >= min_count]
    if max_speed is not None:
        df = df[df["speed"] <= max_speed]

    # Napari vector format: (N, 2, D)
    # D=3 because dimensions are (t, y, x).
    data = np.zeros((len(df), 2, 3), dtype=float)
    data[:, 0, 0] = df["t"].to_numpy()
    data[:, 0, 1] = df["y"].to_numpy()
    data[:, 0, 2] = df["x"].to_numpy()

    # No movement along time axis in the displayed arrow.
    data[:, 1, 0] = 0.0
    data[:, 1, 1] = df["dy"].to_numpy() * vector_scale
    data[:, 1, 2] = df["dx"].to_numpy() * vector_scale
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to binned_velocity_field.csv")
    parser.add_argument("--image-dir", default=None, help="Optional raw image directory")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--vector-scale", type=float, default=10.0)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-speed", type=float, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    import napari

    csv_path = Path(args.csv).expanduser().resolve()
    binned = pd.read_csv(csv_path)

    vectors = make_napari_vectors(
        binned=binned,
        vector_scale=args.vector_scale,
        min_count=args.min_count,
        max_speed=args.max_speed,
    )

    viewer = napari.Viewer(title=args.title or csv_path.parent.name)

    if args.image_dir is not None:
        image_dir = Path(args.image_dir).expanduser().resolve()
        end_frame = args.end_frame
        if end_frame is None:
            # Match the vector time range if no end frame is specified.
            end_frame = args.start_frame + int(binned["t"].max())

        print(f"Loading images {args.start_frame}-{end_frame} from {image_dir}")
        stack = load_image_stack(
            image_dir=image_dir,
            start_frame=args.start_frame,
            end_frame=end_frame,
            downscale=args.downscale,
        )
        viewer.add_image(
            stack,
            name="raw_images",
            scale=(1, args.downscale, args.downscale),
            contrast_limits=[float(np.percentile(stack, 1)), float(np.percentile(stack, 99))],
        )

    viewer.add_vectors(
        vectors,
        name=f"velocity_vectors_x{args.vector_scale:g}",
        edge_width=1.5,
        length=1.0,
    )

    viewer.dims.current_step = (0, 0, 0)
    print("Napari opened.")
    print("Use the time slider to move through frames.")
    print("Increase/decrease --vector-scale if arrows are too small or too large.")
    napari.run()


if __name__ == "__main__":
    main()
