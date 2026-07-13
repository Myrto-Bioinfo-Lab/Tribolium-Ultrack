#!/usr/bin/env python3
"""Create gamma-corrected raw image input folders.

This script reads raw image files, normalizes each frame using robust
percentiles, applies one or more gamma corrections, and saves the resulting
uint16 images to separate output folders.

It can be used for all frames or for a selected frame range. The generated
folders can be used as preprocessing variants for subsequent segmentation
runs.

The script does not run Cellpose, CPSAM or Ultrack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def normalize_frame(img: np.ndarray) -> np.ndarray:
    """Normalize one raw frame to [0, 1] using robust percentiles."""
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99))
    img_norm = (img - p1) / (p99 - p1 + 1e-8)
    return np.clip(img_norm, 0, 1)


def to_uint16(img_norm: np.ndarray) -> np.ndarray:
    """Convert a normalized [0, 1] image to uint16."""
    return np.round(np.clip(img_norm, 0, 1) * 65535).astype(np.uint16)


def parse_gammas(text: str) -> list[float]:
    """Parse comma-separated gamma values."""
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw image files")
    parser.add_argument("--out-base", required=True, help="Output base directory")
    parser.add_argument("--pattern", default="extr_memb_cyl_2_MIP_tp_*.tif")
    parser.add_argument("--gammas", default="0.5,0.7,0.9")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--suffix", default="", help="Optional suffix for output folder names")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_base = Path(args.out_base).expanduser().resolve()
    files = sorted(raw_dir.glob(args.pattern))

    if not files:
        raise FileNotFoundError(f"No raw files found in {raw_dir} with pattern {args.pattern}")

    start = args.start_frame if args.start_frame is not None else 0
    end = args.end_frame if args.end_frame is not None else len(files) - 1

    if start < 0 or end >= len(files) or start > end:
        raise ValueError(f"Invalid frame range {start}-{end} for {len(files)} files")

    selected_files = files[start:end + 1]
    gammas = parse_gammas(args.gammas)

    print("Raw directory:", raw_dir)
    print("Output base:", out_base)
    print("Frames:", start, "to", end)
    print("Selected files:", len(selected_files))
    print("Gammas:", gammas)

    for gamma in gammas:
        gamma_name = str(gamma).replace(".", "p")
        folder_name = f"gamma_{gamma_name}{args.suffix}"
        out_dir = out_base / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print()
        print("Creating:", out_dir)

        for i, file_path in enumerate(selected_files):
            if i % 50 == 0:
                print(f"  {i}/{len(selected_files) - 1}: {file_path.name}")

            raw = imageio.imread(file_path)
            img_norm = normalize_frame(raw)
            img_gamma = img_norm ** gamma
            imageio.imwrite(out_dir / file_path.name, to_uint16(img_gamma))

        print("Done:", out_dir)


if __name__ == "__main__":
    main()
