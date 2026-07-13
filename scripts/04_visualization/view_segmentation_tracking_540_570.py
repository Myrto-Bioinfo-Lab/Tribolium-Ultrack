#!/usr/bin/env python3
"""View segmentation candidates, tracked labels and tracks for frames 540-570.

This Napari viewer loads raw images, selected segmentation candidate folders,
selected preprocessing candidate folders, tracked label outputs and tracks from
completed Ultrack runs.

It is intended for visual comparison of segmentation and tracking results in
the late 540-570 frame window. The script only displays existing outputs. It
does not compute segmentation, tracking or velocity fields.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import zarr


LABEL_CANDIDATES = {
    "provided_reference_labels": "cellpose_label_images",
    "cpsam_raw_d45": "independent_cellpose_cpsam_raw_d45_540_570",
    "cpsam_raw_d55": "independent_cellpose_default_diameter55_540_570",
    "cpsam_raw_d65": "independent_cellpose_cpsam_raw_d65_540_570",
    "cpsam_gamma05_d55": "independent_cellpose_cpsam_gamma05_d55_540_570",
    "cpsam_gamma07_d55": "independent_cellpose_cpsam_gamma07_d55_540_570",
    "cpsam_gamma09_d55": "independent_cellpose_cpsam_gamma09_d55_540_570",
    "classic_threshold_otsu": "independent_classic_threshold_otsu_540_570",
    "classic_threshold_yen": "independent_classic_threshold_yen_540_570",
    "classic_threshold_li": "independent_classic_threshold_li_540_570",
    "classic_watershed_otsu": "independent_classic_watershed_otsu_540_570",
    "classic_watershed_yen": "independent_classic_watershed_yen_540_570",
}

IMAGE_CANDIDATES = {
    "imgproc_raw_norm_saved": "independent_ultrack_imgproc_540_570/raw_norm",
    "foreground_threshold_0p05": "independent_ultrack_imgproc_540_570/foreground_threshold_0p05",
    "foreground_threshold_0p10": "independent_ultrack_imgproc_540_570/foreground_threshold_0p10",
    "contours_robust_invert_sigma_1p0": "independent_ultrack_imgproc_540_570/contours_robust_invert_sigma_1p0",
    "contours_robust_invert_sigma_2p0": "independent_ultrack_imgproc_540_570/contours_robust_invert_sigma_2p0",
    "contours_robust_invert_sigma_3p0": "independent_ultrack_imgproc_540_570/contours_robust_invert_sigma_3p0",
    "contours_canny_sigma_1p0": "independent_ultrack_imgproc_540_570/contours_canny_sigma_1p0",
    "contours_canny_sigma_2p0": "independent_ultrack_imgproc_540_570/contours_canny_sigma_2p0",
}

TRACKING_RUNS = {
    "single_cpsam_d55": "runs/independent_cellpose_cpsam_diameter55_540_570",
    "independent_multi_cpsam_candidates": "runs/independent_multi_cpsam_candidates_540_570",
    "independent_multi_label_candidates": "runs/independent_multi_label_candidates_540_570",
    "multi_existing_labels_540_570": "runs/multi_existing_labels_540_570",
    "u3a_threshold005_sigma10": "runs/u3a_threshold005_sigma10_540_570",
    "u3b_threshold010_sigma10": "runs/u3b_threshold010_sigma10_540_570",
    "u3c_threshold005_sigma20": "runs/u3c_threshold005_sigma20_540_570",
    "u3d_threshold005_sigma30": "runs/u3d_threshold005_sigma30_540_570",
    "u4a_tribolium_params_sigma20": "runs/u4a_tribolium_params_sigma20_540_570",
    "u4b_tribolium_params_sigma30": "runs/u4b_tribolium_params_sigma30_540_570",
}


def normalize_for_display(raw_stack: np.ndarray) -> np.ndarray:
    """Normalize raw images frame-wise to [0, 1] for display."""
    raw_norm = np.zeros(raw_stack.shape, dtype=np.float32)
    for t in range(raw_stack.shape[0]):
        img = raw_stack[t].astype(np.float32)
        p1, p99 = np.percentile(img, (1, 99))
        raw_norm[t] = np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)
    return raw_norm


def normalize_stack_for_display(stack: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary image stack globally to [0, 1]."""
    stack = stack.astype(np.float32)
    low, high = np.percentile(stack, (1, 99))
    return np.clip((stack - low) / (high - low + 1e-8), 0, 1).astype(np.float32)


def load_stack(files: list[Path]) -> np.ndarray:
    """Load image files into one stack."""
    return np.stack([imageio.imread(file_path) for file_path in files])


def select_files(files: list[Path], start_frame: int, end_frame: int, expected_frames: int) -> list[Path]:
    """Select either a full-series frame window or an already cropped 31-frame series."""
    if len(files) == expected_frames:
        return files
    if len(files) >= end_frame + 1:
        return files[start_frame:end_frame + 1]
    return files


def load_label_candidate(name: str, folder: Path, start_frame: int, end_frame: int) -> np.ndarray | None:
    """Load one label-mask candidate if possible."""
    expected_frames = end_frame - start_frame + 1
    if not folder.exists():
        print("Missing label candidate:", name, folder)
        return None

    files = sorted(folder.glob("*_cp_masks.png"))
    files = select_files(files, start_frame, end_frame, expected_frames)

    if len(files) != expected_frames:
        print("Skipping label candidate with unexpected file count:", name, len(files))
        return None

    print("Loaded label candidate:", name, len(files))
    return load_stack(files)


def load_image_candidate(name: str, folder: Path, expected_frames: int) -> np.ndarray | None:
    """Load one image-based preprocessing candidate if possible."""
    if not folder.exists():
        print("Missing image candidate:", name, folder)
        return None

    files = sorted(folder.glob("*.png"))
    if len(files) != expected_frames:
        print("Skipping image candidate with unexpected file count:", name, len(files))
        return None

    print("Loaded image candidate:", name, len(files))
    return normalize_stack_for_display(load_stack(files))


def load_tracked_labels(run_dir: Path, start_frame: int, end_frame: int) -> np.ndarray | None:
    """Load tracked labels if tracked_labels.zarr exists."""
    zarr_path = run_dir / "tracked_labels.zarr"
    if not zarr_path.exists():
        return None

    labels = zarr.open(str(zarr_path), mode="r")
    expected_frames = end_frame - start_frame + 1

    if labels.shape[0] == expected_frames:
        return np.asarray(labels)

    if labels.shape[0] >= end_frame + 1:
        return np.asarray(labels[start_frame:end_frame + 1])

    print("Skipping tracked labels with unexpected shape:", zarr_path, labels.shape)
    return None


def load_tracks(run_dir: Path, start_frame: int, end_frame: int) -> pd.DataFrame | None:
    """Load tracks.csv if available and map full-frame time indices to local indices if needed."""
    tracks_path = run_dir / "tracks.csv"
    if not tracks_path.exists():
        return None

    tracks = pd.read_csv(tracks_path)
    required = {"track_id", "t", "y", "x"}
    if not required.issubset(tracks.columns):
        print("Skipping tracks with missing columns:", tracks_path)
        return None

    tracks = tracks.copy()
    if tracks["t"].max() > (end_frame - start_frame):
        tracks["t"] = tracks["t"] - start_frame

    tracks = tracks[(tracks["t"] >= 0) & (tracks["t"] <= end_frame - start_frame)]
    return tracks


def make_velocity_vectors(tracks_df: pd.DataFrame, max_step_distance: float = 80.0) -> np.ndarray | None:
    """Convert track point pairs into Napari vector data."""
    vectors = []

    for _, group in tracks_df.groupby("track_id"):
        group = group.sort_values("t")
        t = group["t"].to_numpy(dtype=float)
        y = group["y"].to_numpy(dtype=float)
        x = group["x"].to_numpy(dtype=float)

        if len(group) < 2:
            continue

        dt = np.diff(t)
        dy = np.diff(y)
        dx = np.diff(x)

        for i in range(len(dt)):
            if dt[i] != 1:
                continue

            step_distance = float(np.sqrt(dx[i] ** 2 + dy[i] ** 2))
            if step_distance > max_step_distance:
                continue

            vectors.append([
                np.array([t[i], y[i], x[i]], dtype=float),
                np.array([0.0, dy[i], dx[i]], dtype=float),
            ])

    if not vectors:
        return None

    return np.asarray(vectors, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("../Tribolium_Daten"))
    parser.add_argument("--start-frame", type=int, default=540)
    parser.add_argument("--end-frame", type=int, default=570)
    args = parser.parse_args()

    import napari

    data_root = args.data_root.expanduser().resolve()
    raw_dir = data_root / "cylinder3_projections"
    expected_frames = args.end_frame - args.start_frame + 1

    raw_files_all = sorted(raw_dir.glob("extr_memb_cyl_2_MIP_tp_*.tif"))
    raw_files = raw_files_all[args.start_frame:args.end_frame + 1]
    if len(raw_files) != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} raw files, found {len(raw_files)}")

    raw_norm = normalize_for_display(load_stack(raw_files))

    viewer = napari.Viewer()
    viewer.add_image(
        raw_norm,
        name="raw_norm_540_570",
        colormap="gray",
        contrast_limits=(0, 1),
        visible=True,
    )

    for name, relative_path in LABEL_CANDIDATES.items():
        labels = load_label_candidate(
            name,
            data_root / relative_path,
            args.start_frame,
            args.end_frame,
        )
        if labels is not None:
            viewer.add_labels(labels, name=f"seg_{name}", opacity=0.45, visible=False)

    for name, relative_path in IMAGE_CANDIDATES.items():
        stack = load_image_candidate(name, data_root / relative_path, expected_frames)
        if stack is None:
            continue

        colormap = "magma" if name.startswith("contours") else "viridis"
        viewer.add_image(
            stack,
            name=f"img_{name}",
            colormap=colormap,
            contrast_limits=(0, 1),
            opacity=0.45,
            visible=False,
        )

    for name, relative_path in TRACKING_RUNS.items():
        run_dir = data_root / relative_path

        tracked_labels = load_tracked_labels(run_dir, args.start_frame, args.end_frame)
        if tracked_labels is not None:
            viewer.add_labels(
                tracked_labels,
                name=f"tracked_labels_{name}",
                opacity=0.45,
                visible=False,
            )

        tracks = load_tracks(run_dir, args.start_frame, args.end_frame)
        if tracks is not None and not tracks.empty:
            tracks_layer_data = tracks[["track_id", "t", "y", "x"]].to_numpy()
            viewer.add_tracks(
                tracks_layer_data,
                name=f"tracks_{name}",
                tail_length=15,
                visible=False,
            )

            vectors = make_velocity_vectors(tracks)
            if vectors is not None:
                viewer.add_vectors(
                    vectors,
                    name=f"velocity_vectors_{name}",
                    visible=False,
                )

    print()
    print("Napari frame 0 corresponds to original frame", args.start_frame)
    print("Napari frame", expected_frames - 1, "corresponds to original frame", args.end_frame)
    print("Suggested order: raw image + one segmentation, tracked-label, track or vector layer at a time.")

    napari.run()


if __name__ == "__main__":
    main()
