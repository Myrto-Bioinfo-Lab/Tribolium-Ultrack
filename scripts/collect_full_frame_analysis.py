"""Collect full-frame tracking summary statistics.

This script reads selected Ultrack run outputs from ../Tribolium_Daten/runs,
extracts global tracking statistics from export_summary.txt files, and adds
additional track-length and fragmentation metrics from detailed track-analysis
CSV files when available.

The resulting comparison table is written to
../Tribolium_Daten/runs/full_frame_analysis_comparison.csv.

The script is intended for post-processing already completed Ultrack runs. It
does not run segmentation or tracking.
"""

from pathlib import Path
import re

import pandas as pd


RUNS_DIR = Path("../Tribolium_Daten/runs")

# display_name, actual_folder_name
RUNS = [
    ("provided_reference_labels", "provided_reference_labels"),
    ("threshold_robust_full_border", "threshold_robust_full_border"),
    ("multi_cellpose_only_all_frames", "multi_cellpose_only_all_frames"),
    (
        "multi_cellpose_variants_no_baseline_all_frames",
        "multi_cellpose_variants_no_baseline_all_frames",
    ),
]


def extract_number(text, label, number_type=float):
    """Extract one numeric value from an export_summary.txt text."""
    pattern = rf"{re.escape(label)}:\s*([0-9.]+)"
    match = re.search(pattern, text)

    if match is None:
        return None

    value = match.group(1)

    if number_type is int:
        return int(float(value))

    return float(value)


def summarize_track_analysis(track_summary_path, frame_stats_path):
    """Read detailed track-analysis CSV files and compute summary values."""
    if not track_summary_path.exists() or not frame_stats_path.exists():
        return {
            "tracks_le_1": None,
            "tracks_le_2": None,
            "tracks_le_5": None,
            "tracks_le_10": None,
            "tracks_ge_100": None,
            "tracks_ge_300": None,
            "tracks_ge_571": None,
            "active_tracks_mean": None,
            "active_tracks_min": None,
            "active_tracks_max": None,
            "top_fragmentation_frame": None,
            "top_fragmentation_events": None,
        }

    tracks = pd.read_csv(track_summary_path)
    frames = pd.read_csv(frame_stats_path)

    # Recompute fragmentation events if the column is not stored.
    if "fragmentation_events" not in frames.columns:
        frames["fragmentation_events"] = (
            frames["track_starts"] + frames["track_ends"]
        )

    # Exclude frame 0 and the last frame from the fragmentation ranking,
    # because they are naturally special.
    max_frame = frames["t"].max()
    ranking = frames[(frames["t"] > 0) & (frames["t"] < max_frame)].copy()

    if len(ranking) > 0:
        top_row = ranking.sort_values(
            "fragmentation_events",
            ascending=False,
        ).iloc[0]
        top_frame = int(top_row["t"])
        top_events = int(top_row["fragmentation_events"])
    else:
        top_frame = None
        top_events = None

    return {
        "tracks_le_1": int((tracks["length"] <= 1).sum()),
        "tracks_le_2": int((tracks["length"] <= 2).sum()),
        "tracks_le_5": int((tracks["length"] <= 5).sum()),
        "tracks_le_10": int((tracks["length"] <= 10).sum()),
        "tracks_ge_100": int((tracks["length"] >= 100).sum()),
        "tracks_ge_300": int((tracks["length"] >= 300).sum()),
        "tracks_ge_571": int((tracks["length"] >= 571).sum()),
        "active_tracks_mean": float(frames["active_tracks"].mean()),
        "active_tracks_min": int(frames["active_tracks"].min()),
        "active_tracks_max": int(frames["active_tracks"].max()),
        "top_fragmentation_frame": top_frame,
        "top_fragmentation_events": top_events,
    }


def main():
    rows = []

    for display_name, folder_name in RUNS:
        run_dir = RUNS_DIR / folder_name
        summary_path = run_dir / "export_summary.txt"

        if not summary_path.exists():
            print("Missing export summary:", summary_path)
            continue

        text = summary_path.read_text(encoding="utf-8")

        row = {
            "run": display_name,
            "track_points": extract_number(text, "Number of track points", int),
            "tracks": extract_number(text, "Number of tracks", int),
            "first_frame": extract_number(text, "First frame", int),
            "last_frame": extract_number(text, "Last frame", int),
            "mean_track_length": extract_number(text, "Mean track length", float),
            "median_track_length": extract_number(text, "Median track length", float),
            "max_track_length": extract_number(text, "Max track length", int),
        }

        track_analysis_dir = run_dir / "track_analysis"
        extra = summarize_track_analysis(
            track_analysis_dir / "track_summary_detailed.csv",
            track_analysis_dir / "frame_stats.csv",
        )

        row.update(extra)
        rows.append(row)

    df = pd.DataFrame(rows)

    out_csv = RUNS_DIR / "full_frame_analysis_comparison.csv"
    df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print()
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
