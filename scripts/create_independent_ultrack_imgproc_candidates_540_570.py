from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology
from skimage.feature import canny


RAW_DIR = Path("<PROJECT_ROOT>/Tribolium_Daten/cylinder3_projections")
OUTPUT_BASE = Path("<PROJECT_ROOT>/Tribolium_Daten/independent_ultrack_imgproc_540_570")

START_FRAME = 540
END_FRAME = 570

# Candidate settings.
FOREGROUND_THRESHOLDS = [0.05, 0.10]
ROBUST_INVERT_SIGMAS = [1.0, 2.0, 3.0]
CANNY_SIGMAS = [1.0, 2.0]

MIN_OBJECT_SIZE = 300
HOLE_AREA_THRESHOLD = 5000


def normalize_frame(img):
    """Normalize one raw frame to [0, 1] using robust percentiles."""
    img = img.astype(np.float32)

    p1, p99 = np.percentile(img, (1, 99))
    img_norm = (img - p1) / (p99 - p1 + 1e-8)
    img_norm = np.clip(img_norm, 0, 1)

    return img_norm


def robust_invert_like_signal(img_norm, sigma):
    """
    Create a simple boundary-enhancing inverted signal.

    This is not meant to exactly reproduce ultrack.imgproc.robust_invert.
    It is an independent preprocessing candidate inspired by the same idea:
    bright structures become dark and boundary-like contrast is enhanced.
    """
    smooth = ndi.gaussian_filter(img_norm, sigma=sigma)
    inverted = 1.0 - smooth
    inverted = inverted - np.percentile(inverted, 1)
    inverted = inverted / (np.percentile(inverted, 99) + 1e-8)
    return np.clip(inverted, 0, 1).astype(np.float32)


def threshold_foreground(img_norm, threshold):
    """Create a cleaned foreground mask from a normalized image and threshold."""
    foreground = img_norm > threshold
    foreground = morphology.remove_small_objects(
        foreground.astype(bool),
        min_size=MIN_OBJECT_SIZE,
    )
    foreground = morphology.remove_small_holes(
        foreground,
        area_threshold=HOLE_AREA_THRESHOLD,
    )
    return foreground.astype(np.uint8)


def canny_boundary_signal(img_norm, sigma):
    """Create a binary edge signal using Canny filtering."""
    edges = canny(img_norm, sigma=sigma)
    return edges.astype(np.uint8)


def save_uint16_image(img, out_path):
    """Save a normalized image as uint16."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_uint16 = np.round(np.clip(img, 0, 1) * 65535).astype(np.uint16)
    imageio.imwrite(out_path, img_uint16)


def save_uint8_mask(mask, out_path):
    """Save a binary mask as uint8 PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, (mask > 0).astype(np.uint8) * 255)


def main():
    raw_files = sorted(RAW_DIR.glob("extr_memb_cyl_2_MIP_tp_*.tif"))
    selected_files = raw_files[START_FRAME:END_FRAME + 1]

    print("Selected files:", len(selected_files))
    print("First:", selected_files[0])
    print("Last:", selected_files[-1])
    print("Output base:", OUTPUT_BASE)

    # Output folders.
    raw_norm_dir = OUTPUT_BASE / "raw_norm"

    foreground_dirs = {
        threshold: OUTPUT_BASE / f"foreground_threshold_{threshold:.2f}".replace(".", "p")
        for threshold in FOREGROUND_THRESHOLDS
    }

    robust_dirs = {
        sigma: OUTPUT_BASE / f"contours_robust_invert_sigma_{sigma:.1f}".replace(".", "p")
        for sigma in ROBUST_INVERT_SIGMAS
    }

    canny_dirs = {
        sigma: OUTPUT_BASE / f"contours_canny_sigma_{sigma:.1f}".replace(".", "p")
        for sigma in CANNY_SIGMAS
    }

    for frame_index, file_path in enumerate(selected_files):
        original_frame = START_FRAME + frame_index

        if frame_index % 5 == 0:
            print(f"Processing original frame {original_frame}")

        raw = imageio.imread(file_path)
        img_norm = normalize_frame(raw)

        raw_name = f"extr_memb_cyl_2_MIP_tp_{original_frame:03d}_raw_norm.png"
        save_uint16_image(img_norm, raw_norm_dir / raw_name)

        for threshold, out_dir in foreground_dirs.items():
            foreground = threshold_foreground(img_norm, threshold)
            out_name = f"extr_memb_cyl_2_MIP_tp_{original_frame:03d}_foreground.png"
            save_uint8_mask(foreground, out_dir / out_name)

        for sigma, out_dir in robust_dirs.items():
            signal = robust_invert_like_signal(img_norm, sigma=sigma)
            out_name = f"extr_memb_cyl_2_MIP_tp_{original_frame:03d}_contours.png"
            save_uint16_image(signal, out_dir / out_name)

        for sigma, out_dir in canny_dirs.items():
            edges = canny_boundary_signal(img_norm, sigma=sigma)
            out_name = f"extr_memb_cyl_2_MIP_tp_{original_frame:03d}_edges.png"
            save_uint8_mask(edges, out_dir / out_name)

    print()
    print("Done.")
    for path in sorted(OUTPUT_BASE.glob("*")):
        if path.is_dir():
            count = len(list(path.glob("*.png")))
            print(path.name, ":", count, "files")


if __name__ == "__main__":
    main()
