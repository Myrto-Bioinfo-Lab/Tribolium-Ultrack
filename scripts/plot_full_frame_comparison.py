from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUNS_DIR = Path("../Tribolium_Daten/runs")
INPUT_CSV = RUNS_DIR / "full_frame_analysis_comparison.csv"
OUT_DIR = RUNS_DIR / "full_frame_comparison_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


RUN_LABELS = {
    "provided_reference_labels": "provided\nreference",
    "threshold_robust_full_border": "raw-only\nthreshold",
    "multi_cellpose_only_all_frames": "Cellpose\nvariants",
    "multi_cellpose_variants_no_baseline_all_frames": "variants\nno baseline",
}


def prepare_dataframe():
    """Load comparison table and add short display labels."""
    df = pd.read_csv(INPUT_CSV)
    df["label"] = df["run"].map(RUN_LABELS).fillna(df["run"])
    return df


def save_bar_plot(df, column, ylabel, title, output_name):
    """Save one bar plot for a selected numeric column."""
    plt.figure(figsize=(9, 5))
    plt.bar(df["label"], df[column])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=0)
    plt.tight_layout()

    out_path = OUT_DIR / output_name
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("Saved:", out_path)


def main():
    df = prepare_dataframe()

    print("Loaded:", INPUT_CSV)
    print(df.to_string(index=False))

    save_bar_plot(
        df,
        column="tracks",
        ylabel="Number of tracks",
        title="Number of tracks per full-frame run",
        output_name="tracks_per_run.png",
    )

    save_bar_plot(
        df,
        column="mean_track_length",
        ylabel="Mean track length [frames]",
        title="Mean track length per full-frame run",
        output_name="mean_track_length_per_run.png",
    )

    save_bar_plot(
        df,
        column="tracks_le_10",
        ylabel="Tracks with length <= 10 frames",
        title="Short tracks per full-frame run",
        output_name="short_tracks_le_10_per_run.png",
    )

    save_bar_plot(
        df,
        column="tracks_ge_300",
        ylabel="Tracks with length >= 300 frames",
        title="Long tracks per full-frame run",
        output_name="long_tracks_ge_300_per_run.png",
    )

    save_bar_plot(
        df,
        column="tracks_ge_571",
        ylabel="Tracks with length >= 571 frames",
        title="Full-length tracks per full-frame run",
        output_name="full_length_tracks_per_run.png",
    )

    save_bar_plot(
        df,
        column="top_fragmentation_events",
        ylabel="Track starts + ends",
        title="Largest fragmentation peak per full-frame run",
        output_name="top_fragmentation_events_per_run.png",
    )


if __name__ == "__main__":
    main()
