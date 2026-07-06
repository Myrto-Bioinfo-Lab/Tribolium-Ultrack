#!/usr/bin/env python3
"""
Compute track-based velocity fields from an Ultrack-style tracks.csv file.

This script is intentionally generic. It can be used for the 540-570 window
or for a full-frame run, provided that a tracks.csv file with time, track id,
x and y columns is available.

Outputs:
- track_step_vectors.csv
- binned_velocity_field.csv
- binned_velocity_field.npz
- velocity_summary.txt
- optional quiver PNG frames
- optional MP4 video if ffmpeg is available
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TIME_CANDIDATES = ["t", "frame", "time", "timepoint"]
TRACK_CANDIDATES = ["track_id", "track", "id", "trackId", "label_id"]
X_CANDIDATES = ["x", "centroid_x", "x_centroid", "pos_x"]
Y_CANDIDATES = ["y", "centroid_y", "y_centroid", "pos_y"]


def find_column(columns: list[str], candidates: list[str], label: str) -> str:
    lower_to_original = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    raise ValueError(
        f"Could not find {label} column. Available columns: {columns}. "
        f"Tried: {candidates}"
    )


def find_tracks_csv(project_root: Path, run_name: str) -> Path:
    candidates = []

    # Common structure: Tribolium_Daten/runs/<run>/tracks.csv
    candidates.append(project_root / "Tribolium_Daten" / "runs" / run_name / "tracks.csv")

    # Common older structure: Tribolium_Daten/<run>/tracks.csv
    candidates.append(project_root / "Tribolium_Daten" / run_name / "tracks.csv")

    # Broader search fallback.
    for p in (project_root / "Tribolium_Daten").glob(f"**/{run_name}/tracks.csv"):
        candidates.append(p)

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find tracks.csv for run '{run_name}'. "
        "Use --tracks-csv explicitly, or run:\n"
        "find Tribolium_Daten -maxdepth 4 -name tracks.csv | sort"
    )


def compute_step_vectors(tracks: pd.DataFrame) -> pd.DataFrame:
    columns = list(tracks.columns)
    t_col = find_column(columns, TIME_CANDIDATES, "time")
    track_col = find_column(columns, TRACK_CANDIDATES, "track id")
    x_col = find_column(columns, X_CANDIDATES, "x")
    y_col = find_column(columns, Y_CANDIDATES, "y")

    df = tracks[[track_col, t_col, x_col, y_col]].copy()
    df = df.rename(columns={track_col: "track_id", t_col: "t", x_col: "x", y_col: "y"})
    df = df.dropna(subset=["track_id", "t", "x", "y"])
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["t", "x", "y"])
    df = df.sort_values(["track_id", "t"])

    nxt = df.groupby("track_id", sort=False)[["t", "x", "y"]].shift(-1)
    step = df.copy()
    step["t_next"] = nxt["t"]
    step["x_next"] = nxt["x"]
    step["y_next"] = nxt["y"]
    step = step.dropna(subset=["t_next", "x_next", "y_next"])

    step["dt"] = step["t_next"] - step["t"]
    step = step[step["dt"] == 1].copy()

    step["dx"] = step["x_next"] - step["x"]
    step["dy"] = step["y_next"] - step["y"]
    step["speed"] = np.sqrt(step["dx"] ** 2 + step["dy"] ** 2)

    # Midpoint positions are more natural for a vector between two frames.
    step["x_mid"] = 0.5 * (step["x"] + step["x_next"])
    step["y_mid"] = 0.5 * (step["y"] + step["y_next"])

    return step[
        [
            "track_id",
            "t",
            "t_next",
            "x",
            "y",
            "x_next",
            "y_next",
            "x_mid",
            "y_mid",
            "dx",
            "dy",
            "dt",
            "speed",
        ]
    ]


def bin_step_vectors(step: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    b = step.copy()
    b["x_bin"] = np.floor(b["x_mid"] / bin_size).astype(int)
    b["y_bin"] = np.floor(b["y_mid"] / bin_size).astype(int)

    grouped = (
        b.groupby(["t", "y_bin", "x_bin"], as_index=False)
        .agg(
            y=("y_mid", "mean"),
            x=("x_mid", "mean"),
            dy=("dy", "mean"),
            dx=("dx", "mean"),
            speed=("speed", "mean"),
            count=("speed", "size"),
        )
        .sort_values(["t", "y_bin", "x_bin"])
    )
    return grouped


def write_npz(binned: pd.DataFrame, out_path: Path) -> None:
    np.savez_compressed(
        out_path,
        t=binned["t"].to_numpy(),
        y_bin=binned["y_bin"].to_numpy(),
        x_bin=binned["x_bin"].to_numpy(),
        y=binned["y"].to_numpy(),
        x=binned["x"].to_numpy(),
        dy=binned["dy"].to_numpy(),
        dx=binned["dx"].to_numpy(),
        speed=binned["speed"].to_numpy(),
        count=binned["count"].to_numpy(),
    )


def write_summary(
    summary_path: Path,
    tracks_csv: Path,
    step: pd.DataFrame,
    binned: pd.DataFrame,
    bin_size: int,
) -> None:
    def stats(prefix: str, values: pd.Series) -> list[str]:
        if len(values) == 0:
            return [
                f"{prefix} vectors: 0",
                f"{prefix} mean speed: nan",
                f"{prefix} median speed: nan",
                f"{prefix} max speed: nan",
            ]
        return [
            f"{prefix} vectors: {len(values)}",
            f"{prefix} mean speed: {values.mean():.6f} px/frame",
            f"{prefix} median speed: {values.median():.6f} px/frame",
            f"{prefix} max speed: {values.max():.6f} px/frame",
            f"{prefix} q95 speed: {values.quantile(0.95):.6f} px/frame",
            f"{prefix} q99 speed: {values.quantile(0.99):.6f} px/frame",
        ]

    lines = [
        "Track-based velocity field summary",
        "==================================",
        "",
        f"Input tracks.csv: {tracks_csv}",
        f"Bin size: {bin_size}",
        f"Frame range in step vectors: {int(step['t'].min()) if len(step) else 'nan'}-"
        f"{int(step['t'].max()) if len(step) else 'nan'}",
        f"Number of frame transitions: {step['t'].nunique() if len(step) else 0}",
        "",
        *stats("Step", step["speed"]),
        "",
        *stats("Binned", binned["speed"]),
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def plot_quiver_frames(
    binned: pd.DataFrame,
    out_dir: Path,
    run_name: str,
    selected_times: list[int] | None,
    all_frames: bool,
    scale: float,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if all_frames:
        times = sorted(int(t) for t in binned["t"].dropna().unique())
    elif selected_times:
        available = set(int(t) for t in binned["t"].dropna().unique())
        times = [t for t in selected_times if t in available]
    else:
        available = sorted(int(t) for t in binned["t"].dropna().unique())
        if not available:
            times = []
        else:
            idx = np.linspace(0, len(available) - 1, min(6, len(available))).round().astype(int)
            times = [available[i] for i in idx]

    paths = []
    if len(binned) == 0:
        return paths

    xlim = (0, float(binned["x"].max()) + 50)
    ylim = (float(binned["y"].max()) + 50, 0)

    for t in times:
        frame = binned[binned["t"].astype(int) == int(t)]
        if frame.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 10))
        ax.quiver(
            frame["x"],
            frame["y"],
            frame["dx"],
            frame["dy"],
            frame["speed"],
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=0.003,
        )
        ax.set_title(f"{run_name}: binned velocity field, t={t}")
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        path = out_dir / f"{run_name}_t{t:03d}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    return paths


def make_video(frames_dir: Path, run_name: str, video_dir: Path, fps: int) -> Path | None:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found; skipping video creation.")
        return None

    video_dir.mkdir(parents=True, exist_ok=True)
    output = video_dir / f"{run_name}_quiver.mp4"

    pattern = str(frames_dir / f"{run_name}_t%03d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        print("ffmpeg failed; keeping PNG frames.")
        print(exc.stderr.decode("utf-8", errors="replace")[-2000:])
        return None
    return output


def parse_times(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root, e.g. <PROJECT_ROOT>")
    parser.add_argument("--run", required=True, help="Run name used for output folder and auto-search")
    parser.add_argument("--tracks-csv", default=None, help="Explicit path to tracks.csv")
    parser.add_argument("--output-root", default=None, help="Default: Tribolium_Daten/velocity_fields_full")
    parser.add_argument("--bin-size", type=int, default=80)
    parser.add_argument("--make-quiver-frames", action="store_true")
    parser.add_argument("--quiver-times", default="0,100,200,300,400,500,569")
    parser.add_argument("--all-quiver-frames", action="store_true")
    parser.add_argument("--make-video", action="store_true")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--quiver-scale", type=float, default=1.0)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    run_name = args.run

    if args.tracks_csv:
        tracks_csv = Path(args.tracks_csv).expanduser().resolve()
    else:
        tracks_csv = find_tracks_csv(project_root, run_name)

    if not tracks_csv.exists():
        raise FileNotFoundError(tracks_csv)

    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = project_root / "Tribolium_Daten" / "velocity_fields_full"

    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading tracks: {tracks_csv}")
    tracks = pd.read_csv(tracks_csv)
    print(f"Loaded rows: {len(tracks)}")

    print("Computing step vectors ...")
    step = compute_step_vectors(tracks)
    print(f"Step vectors with dt=1: {len(step)}")

    print("Binning vectors ...")
    binned = bin_step_vectors(step, args.bin_size)
    print(f"Binned vectors: {len(binned)}")

    step_path = output_dir / "track_step_vectors.csv"
    binned_path = output_dir / "binned_velocity_field.csv"
    npz_path = output_dir / "binned_velocity_field.npz"
    summary_path = output_dir / "velocity_summary.txt"

    step.to_csv(step_path, index=False)
    binned.to_csv(binned_path, index=False)
    write_npz(binned, npz_path)
    write_summary(summary_path, tracks_csv, step, binned, args.bin_size)

    print(f"Wrote: {step_path}")
    print(f"Wrote: {binned_path}")
    print(f"Wrote: {npz_path}")
    print(f"Wrote: {summary_path}")

    if args.make_quiver_frames or args.make_video:
        frames_dir = output_dir / "quiver_frames"
        selected_times = parse_times(args.quiver_times)
        paths = plot_quiver_frames(
            binned=binned,
            out_dir=frames_dir,
            run_name=run_name,
            selected_times=selected_times,
            all_frames=args.all_quiver_frames,
            scale=args.quiver_scale,
        )
        print(f"Wrote quiver frames: {len(paths)} to {frames_dir}")

        if args.make_video:
            if not args.all_quiver_frames:
                print(
                    "Video requested, but --all-quiver-frames was not set. "
                    "The video will only include selected frames if their numbering is contiguous; "
                    "otherwise ffmpeg may fail. For a full video use --all-quiver-frames."
                )
            video = make_video(frames_dir, run_name, output_dir / "videos", args.fps)
            if video:
                print(f"Wrote video: {video}")


if __name__ == "__main__":
    main()
