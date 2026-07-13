"""Create gamma-preprocessed raw image inputs for frames 540-570.

This script loads the raw Tribolium image sequence, normalizes original frames
540-570, applies selected gamma corrections, and saves the resulting images as
uint16 TIFF-compatible input images.

The generated image folders can be used as preprocessing variants for subsequent
segmentation runs. The script does not run Cellpose, CPSAM or Ultrack.
"""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


RAW_DIR = Path("<PROJECT_ROOT>/Tribolium_Daten/cylinder3_projections")
OUTPUT_BASE = Path("<PROJECT_ROOT>/Tribolium_Daten/independent_inputs")

START_FRAME = 540
END_FRAME = 570

GAMMAS = [0.5, 0.7, 0.9]


def normalize_frame(img):
    """Normalize one raw frame to [0, 1] using robust percentiles."""
    img = img.astype(np.float32)

    p1, p99 = np.percentile(img, (1, 99))
    img_norm = (img - p1) / (p99 - p1 + 1e-8)
    img_norm = np.clip(img_norm, 0, 1)

    return img_norm


def to_uint16(img_norm):
    """Convert a normalized [0, 1] image to uint16."""
    return np.round(img_norm * 65535).astype(np.uint16)


def main():
    raw_files = sorted(RAW_DIR.glob("extr_memb_cyl_2_MIP_tp_*.tif"))
    selected_files = raw_files[START_FRAME:END_FRAME + 1]

    print("Selected files:", len(selected_files))
    print("First:", selected_files[0])
    print("Last:", selected_files[-1])

    for gamma in GAMMAS:
        out_dir = OUTPUT_BASE / f"gamma_{gamma}_540_570"
        out_dir.mkdir(parents=True, exist_ok=True)

        print()
        print("Creating gamma input:", gamma)
        print("Output:", out_dir)

        for file_path in selected_files:
            raw = imageio.imread(file_path)
            raw_norm = normalize_frame(raw)
            img_gamma = np.clip(raw_norm, 0, 1) ** gamma
            img_out = to_uint16(img_gamma)

            out_path = out_dir / file_path.name
            imageio.imwrite(out_path, img_out)

        print("Done:", out_dir)


if __name__ == "__main__":
    main()
