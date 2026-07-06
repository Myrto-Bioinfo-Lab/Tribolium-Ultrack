from pathlib import Path
import re
import pandas as pd


RUNS_DIR = Path("../Tribolium_Daten/runs")

RUNS = [
    ("provided_reference_labels", "provided_reference_labels"),
    ("threshold_robust_full_border", "threshold_robust_full_border"),
    ("u3a_threshold005_sigma10_540_570", "u3a_threshold005_sigma10_540_570"),
    ("u3b_threshold010_sigma10_540_570", "u3b_threshold010_sigma10_540_570"),
    ("u3c_threshold005_sigma20_540_570", "u3c_threshold005_sigma20_540_570"),
    ("u3d_threshold005_sigma30_540_570", "u3d_threshold005_sigma30_540_570"),
    ("u4a_tribolium_params_sigma20_540_570", "u4a_tribolium_params_sigma20_540_570"),
    ("u4b_tribolium_params_sigma30_540_570", "u4b_tribolium_params_sigma30_540_570"),
    ("multi_existing_labels_540_570", "multi_existing_labels_540_570"),
    ("multi_cellpose_only_all_frames", "multi_cellpose_only_all_frames"),
    ("multi_cellpose_variants_no_baseline_all_frames", "multi_cellpose_variants_no_baseline_all_frames"),
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


def main():
    rows = []

    for display_name, folder_name in RUNS:
        summary_path = RUNS_DIR / folder_name / "export_summary.txt"

        if not summary_path.exists():
            print("Missing:", summary_path)
            continue

        text = summary_path.read_text(encoding="utf-8")

        row = {
            "run": display_name,
            # "folder": folder_name,
            "track_points": extract_number(text, "Number of track points", int),
            "tracks": extract_number(text, "Number of tracks", int),
            "first_frame": extract_number(text, "First frame", int),
            "last_frame": extract_number(text, "Last frame", int),
            "mean_track_length": extract_number(text, "Mean track length", float),
            "median_track_length": extract_number(text, "Median track length", float),
            "max_track_length": extract_number(text, "Max track length", int),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    out_csv = RUNS_DIR / "run_summary_comparison.csv"
    df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print()
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
