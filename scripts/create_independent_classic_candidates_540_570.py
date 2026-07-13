"""Create classical segmentation candidate masks for frames 540-570.

This script loads the raw Tribolium image sequence, normalizes original frames
540-570, and generates independent classical label-mask candidates using
thresholding and watershed-based segmentation.

The generated PNG label images use the *_cp_masks.png suffix for compatibility
with the existing label-mask loading workflow. They are not Cellpose outputs.

The script is intended to create additional comparison candidates for Ultrack.
It does not run Ultrack tracking.
"""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation, feature


RAW_DIR = Path("<PROJECT_ROOT>/Tribolium_Daten/cylinder3_projections")
OUTPUT_BASE = Path("<PROJECT_ROOT>/Tribolium_Daten")

START_FRAME = 540
END_FRAME = 570

MIN_OBJECT_SIZE = 300
HOLE_AREA_THRESHOLD = 5000

WATERSHED_MIN_DISTANCE = 12


CANDIDATE_DIRS = {
    "classic_threshold_otsu_540_570": OUTPUT_BASE / "independent_classic_threshold_otsu_540_570",
    "classic_threshold_yen_540_570": OUTPUT_BASE / "independent_classic_threshold_yen_540_570",
    "classic_threshold_li_540_570": OUTPUT_BASE / "independent_classic_threshold_li_540_570",
    "classic_watershed_otsu_540_570": OUTPUT_BASE / "independent_classic_watershed_otsu_540_570",
    "classic_watershed_yen_540_570": OUTPUT_BASE / "independent_classic_watershed_yen_540_570",
}


def normalize_frame(img):
    """Normalize one raw frame to [0, 1] using robust percentiles."""
    img = img.astype(np.float32)

    p1, p99 = np.percentile(img, (1, 99))
    img_norm = (img - p1) / (p99 - p1 + 1e-8)
    img_norm = np.clip(img_norm, 0, 1)

    return img_norm


def clean_foreground(mask):
    """Remove small objects and fill holes in a binary foreground mask."""
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=MIN_OBJECT_SIZE)
    mask = morphology.remove_small_holes(mask, area_threshold=HOLE_AREA_THRESHOLD)
    return mask


def labels_from_threshold(img_norm, method):
    """Create connected-component labels from a threshold method."""
    if method == "otsu":
        threshold = filters.threshold_otsu(img_norm)
    elif method == "yen":
        threshold = filters.threshold_yen(img_norm)
    elif method == "li":
        threshold = filters.threshold_li(img_norm)
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    foreground = img_norm > threshold
    foreground = clean_foreground(foreground)

    labels = measure.label(foreground).astype(np.uint16)
    return labels


def labels_from_watershed(img_norm, method):
    """Create labels using threshold foreground plus distance-transform watershed."""
    if method == "otsu":
        threshold = filters.threshold_otsu(img_norm)
    elif method == "yen":
        threshold = filters.threshold_yen(img_norm)
    else:
        raise ValueError(f"Unknown watershed method: {method}")

    foreground = img_norm > threshold
    foreground = clean_foreground(foreground)

    distance = ndi.distance_transform_edt(foreground)

    coords = feature.peak_local_max(
        distance,
        min_distance=WATERSHED_MIN_DISTANCE,
        labels=foreground,
    )

    markers = np.zeros(distance.shape, dtype=np.int32)

    for marker_id, (y, x) in enumerate(coords, start=1):
        markers[y, x] = marker_id

    markers = ndi.label(markers > 0)[0]

    labels = segmentation.watershed(
        -distance,
        markers,
        mask=foreground,
    ).astype(np.uint16)

    return labels


def save_label_image(labels, out_path):
    """Save a label image as uint16 PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, labels.astype(np.uint16))


def main():
    raw_files = sorted(RAW_DIR.glob("extr_memb_cyl_2_MIP_tp_*.tif"))
    selected_files = raw_files[START_FRAME:END_FRAME + 1]

    print("Selected files:", len(selected_files))
    print("First:", selected_files[0])
    print("Last:", selected_files[-1])

    for out_dir in CANDIDATE_DIRS.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, file_path in enumerate(selected_files):
        original_frame = START_FRAME + frame_index

        if frame_index % 5 == 0:
            print(f"Processing original frame {original_frame}")

        raw = imageio.imread(file_path)
        img_norm = normalize_frame(raw)

        candidates = {
            "classic_threshold_otsu_540_570": labels_from_threshold(img_norm, "otsu"),
            "classic_threshold_yen_540_570": labels_from_threshold(img_norm, "yen"),
            "classic_threshold_li_540_570": labels_from_threshold(img_norm, "li"),
            "classic_watershed_otsu_540_570": labels_from_watershed(img_norm, "otsu"),
            "classic_watershed_yen_540_570": labels_from_watershed(img_norm, "yen"),
        }

        for candidate_name, labels in candidates.items():
            out_dir = CANDIDATE_DIRS[candidate_name]
            out_name = f"extr_memb_cyl_2_MIP_tp_{original_frame:03d}_cp_masks.png"
            out_path = out_dir / out_name

            save_label_image(labels, out_path)

    print()
    print("Done.")
    for candidate_name, out_dir in CANDIDATE_DIRS.items():
        count = len(list(out_dir.glob("*_cp_masks.png")))
        print(candidate_name, ":", count, "files")


if __name__ == "__main__":
    main()
