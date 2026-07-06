from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE = Path("<PROJECT_ROOT>/Tribolium_Daten")
RUNS_DIR = BASE / "runs"

OUT_BASE = BASE / "velocity_fields_540_570"
OUT_BASE.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Runs to process
# ------------------------------------------------------------

RUNS = {
    "single_cpsam_d55": RUNS_DIR / "independent_cellpose_cpsam_diameter55_540_570",
    "independent_multi_label_candidates": RUNS_DIR / "independent_multi_label_candidates_540_570",
    "independent_multi_cpsam_candidates": RUNS_DIR / "independent_multi_cpsam_candidates_540_570",
    "multi_existing_labels_540_570": RUNS_DIR / "multi_existing_labels_540_570",
    "u3a_threshold005_sigma10": RUNS_DIR / "u3a_threshold005_sigma10_540_570",
    "u3b_threshold010_sigma10": RUNS_DIR / "u3b_threshold010_sigma10_540_570",
    "u3c_threshold005_sigma20": RUNS_DIR / "u3c_threshold005_sigma20_540_570",
    "u3d_threshold005_sigma30": RUNS_DIR / "u3d_threshold005_sigma30_540_570",
    "u4a_tribolium_params_sigma20": RUNS_DIR / "u4a_tribolium_params_sigma20_540_570",
    "u4b_tribolium_params_sigma30": RUNS_DIR / "u4b_tribolium_params_sigma30_540_570",
}


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

# Original image size from the data.
IMAGE_HEIGHT = 1612
IMAGE_WIDTH = 1061

# Grid spacing in pixels for the binned velocity field.
# Larger values produce smoother, coarser fields.
BIN_SIZE = 80

# Filter out unrealistic jumps.
MAX_STEP_DISTANCE = 80.0

# Minimum number of step vectors needed in a bin.
MIN_VECTORS_PER_BIN = 3


def compute_step_vectors(tracks_df, max_step_distance):
    """
    Compute one-step displacement vectors from tracks.

    Each row in the result corresponds to one movement from frame t to t+1.
    Units are pixels per frame.
    """
    rows = []

    for track_id, group in tracks_df.groupby("track_id"):
        group = group.sort_values("t")

        t = group["t"].to_numpy(dtype=int)
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

            speed = float(np.sqrt(dx[i] ** 2 + dy[i] ** 2))

            if speed > max_step_distance:
                continue

            rows.append(
                {
                    "track_id": int(track_id),
                    "t": int(t[i]),
                    "t_next": int(t[i + 1]),
                    "y": float(y[i]),
                    "x": float(x[i]),
                    "y_next": float(y[i + 1]),
                    "x_next": float(x[i + 1]),
                    "dy": float(dy[i]),
                    "dx": float(dx[i]),
                    "speed": speed,
                }
            )

    return pd.DataFrame(rows)


def compute_binned_velocity(step_vectors, image_height, image_width, bin_size):
    """
    Average step vectors in spatial bins for each time point.

    The output contains one average velocity vector per time point and grid bin.
    """
    rows = []

    y_edges = np.arange(0, image_height + bin_size, bin_size)
    x_edges = np.arange(0, image_width + bin_size, bin_size)

    for t, frame_vectors in step_vectors.groupby("t"):
        if len(frame_vectors) == 0:
            continue

        y_bin = np.digitize(frame_vectors["y"].to_numpy(), y_edges) - 1
        x_bin = np.digitize(frame_vectors["x"].to_numpy(), x_edges) - 1

        temp = frame_vectors.copy()
        temp["y_bin"] = y_bin
        temp["x_bin"] = x_bin

        valid = (
            (temp["y_bin"] >= 0)
            & (temp["y_bin"] < len(y_edges) - 1)
            & (temp["x_bin"] >= 0)
            & (temp["x_bin"] < len(x_edges) - 1)
        )
        temp = temp[valid]

        for (yb, xb), group in temp.groupby(["y_bin", "x_bin"]):
            count = len(group)

            if count < MIN_VECTORS_PER_BIN:
                continue

            y_center = (y_edges[yb] + y_edges[yb + 1]) / 2.0
            x_center = (x_edges[xb] + x_edges[xb + 1]) / 2.0

            mean_dy = float(group["dy"].mean())
            mean_dx = float(group["dx"].mean())
            mean_speed = float(np.sqrt(mean_dx ** 2 + mean_dy ** 2))

            rows.append(
                {
                    "t": int(t),
                    "y_bin": int(yb),
                    "x_bin": int(xb),
                    "y": float(y_center),
                    "x": float(x_center),
                    "dy": mean_dy,
                    "dx": mean_dx,
                    "speed": mean_speed,
                    "count": int(count),
                }
            )

    return pd.DataFrame(rows)


def save_npz_from_binned(run_out_dir, binned):
    """
    Save binned velocity field also as npz for later numerical analysis.

    This keeps the same information as the CSV, but in array form.
    """
    if binned.empty:
        return

    np.savez(
        run_out_dir / "binned_velocity_field.npz",
        t=binned["t"].to_numpy(),
        y=binned["y"].to_numpy(),
        x=binned["x"].to_numpy(),
        dy=binned["dy"].to_numpy(),
        dx=binned["dx"].to_numpy(),
        speed=binned["speed"].to_numpy(),
        count=binned["count"].to_numpy(),
    )


def process_run(run_name, run_dir):
    """Compute velocity outputs for one Ultrack run."""
    tracks_path = run_dir / "tracks.csv"

    if not tracks_path.exists():
        print("Skipping missing tracks:", run_name, tracks_path)
        return

    print()
    print("=" * 80)
    print("Run:", run_name)
    print("tracks:", tracks_path)

    tracks_df = pd.read_csv(tracks_path)

    out_dir = OUT_BASE / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    step_vectors = compute_step_vectors(
        tracks_df,
        max_step_distance=MAX_STEP_DISTANCE,
    )

    binned = compute_binned_velocity(
        step_vectors,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        bin_size=BIN_SIZE,
    )

    step_path = out_dir / "track_step_vectors.csv"
    binned_path = out_dir / "binned_velocity_field.csv"
    summary_path = out_dir / "velocity_summary.txt"

    step_vectors.to_csv(step_path, index=False)
    binned.to_csv(binned_path, index=False)
    save_npz_from_binned(out_dir, binned)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Input tracks: {tracks_path}\n")
        f.write(f"Image size: {IMAGE_HEIGHT} x {IMAGE_WIDTH}\n")
        f.write(f"Bin size: {BIN_SIZE}\n")
        f.write(f"Max step distance: {MAX_STEP_DISTANCE}\n")
        f.write(f"Min vectors per bin: {MIN_VECTORS_PER_BIN}\n")
        f.write("\n")
        f.write(f"Number of step vectors: {len(step_vectors)}\n")
        f.write(f"Number of binned vectors: {len(binned)}\n")

        if len(step_vectors) > 0:
            f.write(f"Mean step speed: {step_vectors['speed'].mean():.4f}\n")
            f.write(f"Median step speed: {step_vectors['speed'].median():.4f}\n")
            f.write(f"Max step speed: {step_vectors['speed'].max():.4f}\n")

        if len(binned) > 0:
            f.write(f"Mean binned speed: {binned['speed'].mean():.4f}\n")
            f.write(f"Median binned speed: {binned['speed'].median():.4f}\n")
            f.write(f"Max binned speed: {binned['speed'].max():.4f}\n")

    print("Step vectors:", len(step_vectors))
    print("Binned vectors:", len(binned))
    print("Saved:", step_path)
    print("Saved:", binned_path)
    print("Saved:", summary_path)


def main():
    for run_name, run_dir in RUNS.items():
        process_run(run_name, run_dir)

    print()
    print("Done.")
    print("Output base:", OUT_BASE)


if __name__ == "__main__":
    main()
