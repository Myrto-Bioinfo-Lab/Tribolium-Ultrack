import os
import sys
import yaml
import platform
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError

import pandas as pd
from ultrack import load_config, to_tracks_layer


# ------------------------------------------------------------
# Choose the finished Ultrack output directory.
# Change this path if your finished run has a different name.
# ------------------------------------------------------------

if len(sys.argv) != 2:
    raise SystemExit("Usage: python export_ultrack_results.py <output_dir>")

output_dir = sys.argv[1]


def get_package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


# ------------------------------------------------------------
# Load Ultrack result
# ------------------------------------------------------------

config_path = os.path.join(output_dir, "metadata.toml")

config = load_config(config_path)

config_used_path = os.path.join(output_dir, "config_used.yaml")
description = "No description available."

if os.path.exists(config_used_path):
    with open(config_used_path, "r") as f:
        config_used = yaml.safe_load(f)
    description = config_used.get("notes", {}).get("description", description)

config.data_config.working_dir = output_dir

tracks_df, graph = to_tracks_layer(config)


# ------------------------------------------------------------
# Export full tracks table
# ------------------------------------------------------------

tracks_csv = os.path.join(output_dir, "tracks.csv")
tracks_df.to_csv(tracks_csv, index=False)


# ------------------------------------------------------------
# Export one-row-per-track summary
# ------------------------------------------------------------

summary = (
    tracks_df
    .groupby("track_id")
    .agg(
        start_frame=("t", "min"),
        end_frame=("t", "max"),
        length=("t", "count"),
        mean_y=("y", "mean"),
        mean_x=("x", "mean"),
        min_y=("y", "min"),
        max_y=("y", "max"),
        min_x=("x", "min"),
        max_x=("x", "max"),
    )
    .reset_index()
)

summary_csv = os.path.join(output_dir, "track_summary.csv")
summary.to_csv(summary_csv, index=False)


# ------------------------------------------------------------
# Export frame-wise track counts
# ------------------------------------------------------------

per_frame = (
    tracks_df
    .groupby("t")
    .agg(
        n_track_points=("track_id", "count"),
        n_active_tracks=("track_id", "nunique"),
    )
    .reset_index()
)

per_frame_csv = os.path.join(output_dir, "tracks_per_frame.csv")
per_frame.to_csv(per_frame_csv, index=False)


# ------------------------------------------------------------
# Export run information
# ------------------------------------------------------------

run_info_path = os.path.join(output_dir, "export_summary.txt")

with open(run_info_path, "w") as f:
    f.write("Ultrack run information\n")
    f.write("=======================\n\n")

    f.write(f"Export date: {datetime.now()}\n")
    f.write(f"Output directory: {output_dir}\n")
    f.write(f"Config path: {config_path}\n\n")

    f.write("Track statistics\n")
    f.write("----------------\n")
    f.write(f"Number of track points: {len(tracks_df)}\n")
    f.write(f"Number of tracks: {tracks_df['track_id'].nunique()}\n")
    f.write(f"First frame: {tracks_df['t'].min()}\n")
    f.write(f"Last frame: {tracks_df['t'].max()}\n")
    f.write(f"Mean track length: {summary['length'].mean():.2f}\n")
    f.write(f"Median track length: {summary['length'].median():.2f}\n")
    f.write(f"Max track length: {summary['length'].max()}\n\n")

    f.write("Software versions\n")
    f.write("-----------------\n")
    f.write(f"Python: {platform.python_version()}\n")
    f.write(f"ultrack: {get_package_version('ultrack')}\n")
    f.write(f"numpy: {get_package_version('numpy')}\n")
    f.write(f"pandas: {get_package_version('pandas')}\n")
    f.write(f"scikit-image: {get_package_version('scikit-image')}\n")
    f.write(f"napari: {get_package_version('napari')}\n")
    f.write(f"vispy: {get_package_version('vispy')}\n\n")

    f.write("Notes\n")
    f.write("-----\n")
    f.write(description + "\n")
    f.write("The exact run parameters are stored in config_used.yaml and metadata.toml.\n")


print("Saved:")
print(tracks_csv)
print(summary_csv)
print(per_frame_csv)
print(run_info_path)

print()
print("Quick summary:")
print("Number of track points:", len(tracks_df))
print("Number of tracks:", tracks_df["track_id"].nunique())
print("Frames:", tracks_df["t"].min(), "to", tracks_df["t"].max())
print("Mean track length:", summary["length"].mean())
print("Median track length:", summary["length"].median())
print("Max track length:", summary["length"].max())
