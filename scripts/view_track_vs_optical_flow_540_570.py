from pathlib import Path

import imageio.v2 as imageio
import napari
import numpy as np
import pandas as pd


BASE = Path("<PROJECT_ROOT>/Tribolium_Daten")
RAW_DIR = BASE / "cylinder3_projections"

TRACK_VELOCITY_BASE = BASE / "velocity_fields_540_570"
OPTICAL_FLOW_DIR = BASE / "optical_flow_540_570"

START_FRAME = 540
END_FRAME = 570
N_FRAMES = END_FRAME - START_FRAME + 1

TRACK_RUNS = [
    "single_cpsam_d55",
    "independent_multi_label_candidates",
    "independent_multi_cpsam_candidates",
    "multi_existing_labels_540_570",
]


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


def df_to_vectors(df, scale=4.0):
    """Convert a velocity dataframe to Napari vectors."""
    vectors = []

    for _, row in df.iterrows():
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


def add_track_velocity(viewer, run_name):
    """Add one track-based binned velocity field."""
    csv_path = TRACK_VELOCITY_BASE / run_name / "binned_velocity_field.csv"

    if not csv_path.exists():
        print("Missing track velocity:", csv_path)
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("Empty track velocity:", csv_path)
        return

    vectors = df_to_vectors(df, scale=4.0)

    viewer.add_vectors(
        vectors,
        name=f"track_binned_{run_name}",
        visible=False,
        edge_width=2,
        length=1.0,
    )

    print("Added track velocity:", run_name, len(df))


def add_optical_flow(viewer):
    """Add raw-image optical-flow field."""
    csv_path = OPTICAL_FLOW_DIR / "raw_optical_flow_binned.csv"

    if not csv_path.exists():
        print("Missing optical flow:", csv_path)
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("Empty optical flow:", csv_path)
        return

    vectors = df_to_vectors(df, scale=4.0)

    viewer.add_vectors(
        vectors,
        name="raw_optical_flow_binned",
        visible=False,
        edge_width=2,
        length=1.0,
    )

    print("Added raw optical flow:", len(df))


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

    add_optical_flow(viewer)

    for run_name in TRACK_RUNS:
        add_track_velocity(viewer, run_name)

    print()
    print("Napari frame 0 corresponds to original frame", START_FRAME)
    print("Napari frame", END_FRAME - START_FRAME, "corresponds to original frame", END_FRAME)
    print()
    print("Suggested comparison:")
    print("- raw_optical_flow_binned")
    print("- track_binned_independent_multi_cpsam_candidates")
    print("- track_binned_multi_existing_labels_540_570")
    print()
    print("Turn on one or two vector layers at a time.")

    napari.run()


if __name__ == "__main__":
    main()
