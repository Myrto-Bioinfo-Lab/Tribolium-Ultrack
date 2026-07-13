"""Open multiple track-based velocity fields for frames 540-570 in napari.

This script loads the normalized raw image sequence for original frames 540-570
and adds several precomputed binned velocity fields as Napari vector layers.

The viewer is intended for visual comparison of different Ultrack runs in the
late serosa-motion window. Napari frame 0 corresponds to original frame 540.

The script does not compute new tracking or velocity data. It only displays
existing CSV outputs.
"""

from pathlib import Path

import imageio.v2 as imageio
import napari
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Paths and settings
# ------------------------------------------------------------

BASE = Path("<PROJECT_ROOT>/Tribolium_Daten")
RAW_DIR = BASE / "cylinder3_projections"
VELOCITY_BASE = BASE / "velocity_fields_540_570"

START_FRAME = 540
END_FRAME = 570
N_FRAMES = END_FRAME - START_FRAME + 1

RUNS = [
    "single_cpsam_d55",
    "independent_multi_label_candidates",
    "independent_multi_cpsam_candidates",
    "multi_existing_labels_540_570",
    "u3a_threshold005_sigma10",
    "u3b_threshold010_sigma10",
    "u3c_threshold005_sigma20",
    "u3d_threshold005_sigma30",
    "u4a_tribolium_params_sigma20",
    "u4b_tribolium_params_sigma30",
]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_for_display(raw_stack):
    """Normalize raw images frame-wise to [0, 1] for display."""
    raw_norm = np.zeros(raw_stack.shape, dtype=np.float32)

    for t in range(raw_stack.shape[0]):
        img = raw_stack[t].astype(np.float32)
        p1, p99 = np.percentile(img, (1, 99))
        img_norm = (img - p1) / (p99 - p1 + 1e-8)
        raw_norm[t] = np.clip(img_norm, 0, 1)

    return raw_norm


def load_stack(files):
    """Load image files into a stack."""
    return np.stack([imageio.imread(file_path) for file_path in files])


def load_raw():
    """Load raw frames 540-570."""
    raw_files_all = sorted(RAW_DIR.glob("extr_memb_cyl_2_MIP_tp_*.tif"))
    raw_files = raw_files_all[START_FRAME:END_FRAME + 1]

    if len(raw_files) != N_FRAMES:
        raise RuntimeError(f"Expected {N_FRAMES} frames, found {len(raw_files)}")

    raw = load_stack(raw_files)
    return normalize_for_display(raw)


def binned_to_napari_vectors(binned_df, scale=4.0):
    """
    Convert binned velocity field to Napari vectors.

    Napari vector format:
    start point: [t, y, x]
    direction:   [0, dy * scale, dx * scale]
    """
    vectors = []

    for _, row in binned_df.iterrows():
        start = np.array(
            [float(row["t"]), float(row["y"]), float(row["x"])],
            dtype=float,
        )
        direction = np.array(
            [0.0, float(row["dy"]) * scale, float(row["dx"]) * scale],
            dtype=float,
        )
        vectors.append([start, direction])

    if not vectors:
        return None

    return np.asarray(vectors, dtype=float)


def add_velocity_run(viewer, run_name):
    """Add one binned velocity field to Napari if it exists."""
    csv_path = VELOCITY_BASE / run_name / "binned_velocity_field.csv"

    if not csv_path.exists():
        print("Missing:", csv_path)
        return

    binned = pd.read_csv(csv_path)

    if binned.empty:
        print("Empty:", csv_path)
        return

    vectors = binned_to_napari_vectors(binned, scale=4.0)

    if vectors is None:
        print("No vectors:", run_name)
        return

    print()
    print("Run:", run_name)
    print("Binned rows:", len(binned))
    print("Vectors:", vectors.shape)
    print("Mean speed:", binned["speed"].mean())

    viewer.add_vectors(
        vectors,
        name=f"binned_velocity_{run_name}",
        visible=False,
        edge_width=2,
        length=1.0,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    raw_norm = load_raw()

    viewer = napari.Viewer()

    viewer.add_image(
        raw_norm,
        name="raw_norm_540_570",
        colormap="gray",
        contrast_limits=(0, 1),
        visible=True,
    )

    for run_name in RUNS:
        add_velocity_run(viewer, run_name)

    print()
    print("Napari frame 0 corresponds to original frame", START_FRAME)
    print("Napari frame", END_FRAME - START_FRAME, "corresponds to original frame", END_FRAME)
    print()
    print("Turn on one binned_velocity_* layer at a time.")
    print("These are spatially averaged track-based velocity fields.")
    print("Units are pixels per frame before scaling for display.")

    napari.run()


if __name__ == "__main__":
    main()
