#!/usr/bin/env python3
"""
Create final plots, quiver images and optional videos for the Tribolium/Ultrack workflow.

Run from the project root, for example:

    cd <PROJECT_ROOT>
    python scripts/make_final_analysis_outputs.py --project-root .
    python scripts/make_final_analysis_outputs.py --project-root . --make-videos

The script expects the existing project structure:

    Tribolium_Daten/runs/run_summary_comparison.csv
    Tribolium_Daten/runs/full_frame_analysis_comparison.csv
    Tribolium_Daten/velocity_fields_540_570/<run_name>/binned_velocity_field.csv
    Tribolium_Daten/velocity_fields_540_570/<run_name>/track_step_vectors.csv
    Tribolium_Daten/optical_flow_540_570/raw_optical_flow_binned.csv

If a file is missing, that part is skipped with a warning.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_ORDER = [
    "provided_reference_labels",
    "threshold_robust_full_border",
    "u3a_threshold005_sigma10_540_570",
    "u3b_threshold010_sigma10_540_570",
    "u3c_threshold005_sigma20_540_570",
    "u3d_threshold005_sigma30_540_570",
    "u4a_tribolium_params_sigma20_540_570",
    "u4b_tribolium_params_sigma30_540_570",
    "multi_existing_labels_540_570",
    "multi_cellpose_only_all_frames",
    "multi_cellpose_variants_no_baseline_all_frames",
]

FINAL_540_570_RUNS = [
    "multi_existing_labels_540_570",
    "independent_multi_cpsam_candidates",
    "independent_multi_label_candidates",
    "single_cpsam_d55",
]


def warn(message: str) -> None:
    print(f"[warning] {message}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"Missing file: {path}")
        return None
    return pd.read_csv(path)


def save_bar(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    horizontal: bool = True,
) -> None:
    if df.empty or category_col not in df or value_col not in df:
        warn(f"Cannot create {output_path.name}: missing columns")
        return

    data = df[[category_col, value_col]].dropna().copy()
    if category_col == "run":
        data["_order"] = data[category_col].map({name: i for i, name in enumerate(RUN_ORDER)})
        data = data.sort_values(["_order", category_col]).drop(columns="_order")

    fig_height = max(4.0, 0.36 * len(data) + 1.6)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    if horizontal:
        ax.barh(data[category_col], data[value_col])
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
    else:
        ax.bar(data[category_col], data[value_col])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)

    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Wrote {output_path}")


def create_run_summary_plots(root: Path, output_dir: Path) -> None:
    runs_dir = root / "Tribolium_Daten" / "runs"
    run_summary = read_csv_if_exists(runs_dir / "run_summary_comparison.csv")
    full_summary = read_csv_if_exists(runs_dir / "full_frame_analysis_comparison.csv")

    if run_summary is not None:
        save_bar(
            run_summary,
            "run",
            "tracks",
            "Tracks per run",
            "Number of tracks",
            output_dir / "run_tracks_bar.png",
        )
        save_bar(
            run_summary,
            "run",
            "mean_track_length",
            "Mean track length per run",
            "Mean track length [frames]",
            output_dir / "run_mean_track_length_bar.png",
        )
        save_bar(
            run_summary,
            "run",
            "track_points",
            "Track points per run",
            "Number of track points",
            output_dir / "run_track_points_bar.png",
        )

    if full_summary is not None:
        required = {"run", "tracks_le_10", "tracks_ge_300"}
        if required.issubset(full_summary.columns):
            df = full_summary.copy()
            df["_order"] = df["run"].map({name: i for i, name in enumerate(RUN_ORDER)})
            df = df.sort_values(["_order", "run"]).drop(columns="_order")

            y = np.arange(len(df))
            height = 0.38
            fig, ax = plt.subplots(figsize=(11, max(4.5, 0.5 * len(df) + 1.8)))
            ax.barh(y - height / 2, df["tracks_le_10"], height, label="Tracks <= 10 frames")
            ax.barh(y + height / 2, df["tracks_ge_300"], height, label="Tracks >= 300 frames")
            ax.set_yticks(y)
            ax.set_yticklabels(df["run"])
            ax.invert_yaxis()
            ax.set_xlabel("Number of tracks")
            ax.set_title("Short and long tracks in full-frame runs")
            ax.grid(axis="x", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            out = output_dir / "full_frame_short_vs_long_tracks.png"
            fig.savefig(out, dpi=220)
            plt.close(fig)
            print(f"Wrote {out}")

        if {"run", "active_tracks_mean", "active_tracks_min", "active_tracks_max"}.issubset(full_summary.columns):
            df = full_summary.copy()
            df["_order"] = df["run"].map({name: i for i, name in enumerate(RUN_ORDER)})
            df = df.sort_values(["_order", "run"]).drop(columns="_order")
            y = np.arange(len(df))
            x = df["active_tracks_mean"].to_numpy()
            xerr = np.vstack([
                x - df["active_tracks_min"].to_numpy(),
                df["active_tracks_max"].to_numpy() - x,
            ])
            fig, ax = plt.subplots(figsize=(11, max(4.5, 0.5 * len(df) + 1.8)))
            ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=4)
            ax.set_yticks(y)
            ax.set_yticklabels(df["run"])
            ax.invert_yaxis()
            ax.set_xlabel("Active tracks per frame")
            ax.set_title("Mean active tracks with min/max range")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            out = output_dir / "full_frame_active_tracks_range.png"
            fig.savefig(out, dpi=220)
            plt.close(fig)
            print(f"Wrote {out}")


def first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def infer_vector_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    t_col = first_existing(df.columns, ["t", "frame", "time", "frame_id"])
    x_col = first_existing(df.columns, ["x", "x_center", "center_x", "x_bin_center", "x_mean", "col"])
    y_col = first_existing(df.columns, ["y", "y_center", "center_y", "y_bin_center", "y_mean", "row"])
    dx_col = first_existing(df.columns, ["dx", "vx", "u", "flow_x", "mean_dx", "binned_dx"])
    dy_col = first_existing(df.columns, ["dy", "vy", "v", "flow_y", "mean_dy", "binned_dy"])
    return t_col, x_col, y_col, dx_col, dy_col


def add_speed_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    if "speed" in df.columns:
        return df
    _, _, _, dx_col, dy_col = infer_vector_columns(df)
    if dx_col and dy_col:
        df = df.copy()
        df["speed"] = np.sqrt(df[dx_col].astype(float) ** 2 + df[dy_col].astype(float) ** 2)
    return df


def summarize_vector_file(path: Path, source: str, kind: str) -> dict[str, float | int | str] | None:
    df = read_csv_if_exists(path)
    if df is None or df.empty:
        return None
    df = add_speed_if_missing(df)
    row: dict[str, float | int | str] = {
        "source": source,
        "kind": kind,
        "file": str(path),
        "n_vectors": len(df),
    }
    if "speed" in df.columns:
        speed = pd.to_numeric(df["speed"], errors="coerce").dropna()
        if len(speed):
            row.update(
                {
                    "mean_speed": float(speed.mean()),
                    "median_speed": float(speed.median()),
                    "max_speed": float(speed.max()),
                    "q95_speed": float(speed.quantile(0.95)),
                    "q99_speed": float(speed.quantile(0.99)),
                }
            )
    return row


def create_velocity_summary(root: Path, output_dir: Path) -> pd.DataFrame:
    velocity_root = root / "Tribolium_Daten" / "velocity_fields_540_570"
    optical_root = root / "Tribolium_Daten" / "optical_flow_540_570"

    rows: list[dict[str, float | int | str]] = []
    if velocity_root.exists():
        for run_dir in sorted(p for p in velocity_root.iterdir() if p.is_dir()):
            binned = summarize_vector_file(run_dir / "binned_velocity_field.csv", run_dir.name, "track_binned")
            steps = summarize_vector_file(run_dir / "track_step_vectors.csv", run_dir.name, "track_steps")
            if binned:
                rows.append(binned)
            if steps:
                rows.append(steps)
    else:
        warn(f"Missing directory: {velocity_root}")

    optical = summarize_vector_file(optical_root / "raw_optical_flow_binned.csv", "raw_optical_flow", "optical_flow_binned")
    if optical:
        rows.append(optical)

    summary = pd.DataFrame(rows)
    if summary.empty:
        warn("No velocity or optical-flow data found")
        return summary

    out_csv = output_dir / "velocity_field_comparison.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    binned = summary[summary["kind"].isin(["track_binned", "optical_flow_binned"])].copy()
    if not binned.empty and "mean_speed" in binned.columns:
        save_bar(
            binned.sort_values("mean_speed", ascending=True),
            "source",
            "mean_speed",
            "Mean binned velocity by source",
            "Mean speed [px/frame]",
            output_dir / "velocity_mean_speed_bar.png",
        )
    if not binned.empty and "median_speed" in binned.columns:
        save_bar(
            binned.sort_values("median_speed", ascending=True),
            "source",
            "median_speed",
            "Median binned velocity by source",
            "Median speed [px/frame]",
            output_dir / "velocity_median_speed_bar.png",
        )
    if not binned.empty and "n_vectors" in binned.columns:
        save_bar(
            binned.sort_values("n_vectors", ascending=True),
            "source",
            "n_vectors",
            "Number of binned vectors by source",
            "Number of vectors",
            output_dir / "velocity_vector_count_bar.png",
        )

    return summary


def choose_frames(df: pd.DataFrame, requested: list[int] | None) -> list[int | None]:
    t_col, *_ = infer_vector_columns(df)
    if not t_col:
        return [None]
    available = sorted(pd.to_numeric(df[t_col], errors="coerce").dropna().astype(int).unique().tolist())
    if not available:
        return [None]
    if requested:
        return [frame for frame in requested if frame in available]
    candidates = [available[0], available[len(available) // 3], available[(2 * len(available)) // 3], available[-1]]
    return sorted(set(candidates))


def subset_frame(df: pd.DataFrame, frame: int | None) -> pd.DataFrame:
    t_col, *_ = infer_vector_columns(df)
    if frame is None or not t_col:
        return df.copy()
    return df[pd.to_numeric(df[t_col], errors="coerce") == frame].copy()


def plot_quiver(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    *,
    frame: int | None = None,
    max_vectors: int = 900,
    scale: float | None = None,
    clip_quantile: float | None = 0.99,
) -> bool:
    data = subset_frame(df, frame)
    if data.empty:
        warn(f"No data for quiver plot {title}, frame={frame}")
        return False

    data = add_speed_if_missing(data)
    _, x_col, y_col, dx_col, dy_col = infer_vector_columns(data)
    if not all([x_col, y_col, dx_col, dy_col]):
        warn(f"Cannot infer vector columns for {output_path.name}; columns are {list(data.columns)}")
        return False

    data = data.dropna(subset=[x_col, y_col, dx_col, dy_col]).copy()
    if data.empty:
        return False

    if "speed" in data.columns and clip_quantile is not None and 0 < clip_quantile < 1:
        limit = float(pd.to_numeric(data["speed"], errors="coerce").quantile(clip_quantile))
        if math.isfinite(limit) and limit > 0:
            data = data[pd.to_numeric(data["speed"], errors="coerce") <= limit]

    if len(data) > max_vectors:
        data = data.sample(max_vectors, random_state=1)

    x = pd.to_numeric(data[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(data[y_col], errors="coerce").to_numpy()
    dx = pd.to_numeric(data[dx_col], errors="coerce").to_numpy()
    dy = pd.to_numeric(data[dy_col], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.quiver(x, y, dx, dy, angles="xy", scale_units="xy", scale=scale if scale else None, width=0.003)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_title(title if frame is None else f"{title} | frame {frame}")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return True


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def create_quiver_images(root: Path, output_dir: Path, requested_frames: list[int] | None, max_vectors: int) -> None:
    quiver_dir = ensure_dir(output_dir / "quiver_frames")
    velocity_root = root / "Tribolium_Daten" / "velocity_fields_540_570"
    optical_root = root / "Tribolium_Daten" / "optical_flow_540_570"

    sources: list[tuple[str, Path]] = []
    for run_name in FINAL_540_570_RUNS:
        path = velocity_root / run_name / "binned_velocity_field.csv"
        if path.exists():
            sources.append((run_name, path))
    optical_path = optical_root / "raw_optical_flow_binned.csv"
    if optical_path.exists():
        sources.append(("raw_optical_flow", optical_path))

    if not sources:
        warn("No quiver sources found")
        return

    for source_name, path in sources:
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue
        frames = choose_frames(df, requested_frames)
        for frame in frames:
            suffix = "all" if frame is None else f"t{frame:03d}"
            out = quiver_dir / f"{sanitize_name(source_name)}_{suffix}.png"
            ok = plot_quiver(
                df,
                source_name,
                out,
                frame=frame,
                max_vectors=max_vectors,
            )
            if ok:
                print(f"Wrote {out}")


def render_video_frames(root: Path, output_dir: Path, max_vectors: int) -> dict[str, Path]:
    video_frame_root = ensure_dir(output_dir / "video_frames")
    velocity_root = root / "Tribolium_Daten" / "velocity_fields_540_570"
    optical_root = root / "Tribolium_Daten" / "optical_flow_540_570"

    sources = {
        "multi_existing_labels_540_570": velocity_root / "multi_existing_labels_540_570" / "binned_velocity_field.csv",
        "independent_multi_cpsam_candidates": velocity_root / "independent_multi_cpsam_candidates" / "binned_velocity_field.csv",
        "raw_optical_flow": optical_root / "raw_optical_flow_binned.csv",
    }

    rendered_dirs: dict[str, Path] = {}
    for source_name, path in sources.items():
        if not path.exists():
            warn(f"Skipping video source, missing {path}")
            continue
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue
        t_col, *_ = infer_vector_columns(df)
        if not t_col:
            warn(f"Skipping video source without time column: {source_name}")
            continue
        frames = sorted(pd.to_numeric(df[t_col], errors="coerce").dropna().astype(int).unique().tolist())
        source_frame_dir = ensure_dir(video_frame_root / sanitize_name(source_name))
        rendered_count = 0
        for frame in frames:
            out = source_frame_dir / f"frame_{frame:04d}.png"
            ok = plot_quiver(df, source_name, out, frame=frame, max_vectors=max_vectors)
            if ok:
                rendered_count += 1
        if rendered_count:
            rendered_dirs[source_name] = source_frame_dir
            print(f"Rendered {rendered_count} frames for {source_name}")
    return rendered_dirs


def create_mp4_from_frames(frame_dir: Path, output_path: Path, fps: int = 6) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        warn("ffmpeg not found; MP4 videos were not created. The PNG frames are still available.")
        return
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        str(frame_dir / "frame_*.png"),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote {output_path}")


def create_videos(root: Path, output_dir: Path, max_vectors: int) -> None:
    video_dir = ensure_dir(output_dir / "videos")
    rendered_dirs = render_video_frames(root, output_dir, max_vectors=max_vectors)
    for source_name, frame_dir in rendered_dirs.items():
        create_mp4_from_frames(frame_dir, video_dir / f"{sanitize_name(source_name)}_quiver.mp4")


def parse_frames_arg(text: str | None) -> list[int] | None:
    if not text:
        return None
    frames = []
    for part in text.split(","):
        part = part.strip()
        if part:
            frames.append(int(part))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final Tribolium/Ultrack analysis outputs.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root, e.g. <PROJECT_ROOT>")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for plots and videos")
    parser.add_argument("--quiver-frames", type=str, default=None, help="Comma-separated frame indices, e.g. 0,10,20,30")
    parser.add_argument("--max-vectors", type=int, default=900, help="Maximum vectors per quiver plot")
    parser.add_argument("--make-videos", action="store_true", help="Render MP4 quiver videos if ffmpeg is available")
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else root / "plots" / "final_analysis"
    ensure_dir(output_dir)

    print(f"Project root: {root}")
    print(f"Output dir:   {output_dir}")

    create_run_summary_plots(root, output_dir)
    create_velocity_summary(root, output_dir)
    create_quiver_images(root, output_dir, parse_frames_arg(args.quiver_frames), max_vectors=args.max_vectors)

    if args.make_videos:
        create_videos(root, output_dir, max_vectors=args.max_vectors)

    print("Done.")


if __name__ == "__main__":
    main()
