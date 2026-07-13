#!/usr/bin/env python3
"""Export overlay PNG frames or MP4 videos from tracked Ultrack labels.

This script overlays tracked label images and optional track tails on the raw
image sequence. It expects a completed Ultrack run with tracks.csv and a
tracked_labels.zarr file.

The tracked_labels.zarr file can be generated with scripts/01_pipeline/export_tracked_labels.py.

Outputs can be written as PNG frame series, as an MP4 video, or both. The script
is intended for visual inspection and presentation videos. It does not run
segmentation or tracking.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import zarr
from PIL import Image, ImageDraw


RAW_PATTERN = "extr_memb_cyl_2_MIP_tp_*.tif"


def normalize_frame_for_display(img: np.ndarray) -> np.ndarray:
    """Normalize one raw frame to [0, 1] for display."""
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99))
    img_norm = (img - p1) / (p99 - p1 + 1e-8)
    return np.clip(img_norm, 0, 1)


def label_color(label_values: np.ndarray) -> np.ndarray:
    """Create deterministic RGB colors from label IDs."""
    label_values = label_values.astype(np.int64)
    r = ((label_values * 37) % 255).astype(np.uint8)
    g = ((label_values * 67) % 255).astype(np.uint8)
    b = ((label_values * 97) % 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def make_overlay_frame(raw_norm: np.ndarray, labels: np.ndarray, label_alpha: float) -> np.ndarray:
    """Create RGB image from raw image and semi-transparent label overlay."""
    gray = np.round(raw_norm * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    mask = labels > 0
    if np.any(mask):
        colors = label_color(labels[mask])
        rgb[mask] = (1.0 - label_alpha) * rgb[mask] + label_alpha * colors

    return np.clip(rgb, 0, 255).astype(np.uint8)


def track_color(track_id: int) -> tuple[int, int, int]:
    """Create deterministic RGB color for one track ID."""
    track_id = int(track_id)
    return (
        int((track_id * 37) % 255),
        int((track_id * 67) % 255),
        int((track_id * 97) % 255),
    )


def draw_track_tails(
    image: Image.Image,
    tracks_df: pd.DataFrame,
    frame: int,
    scale: float,
    start_frame: int,
    tail_length: int,
) -> Image.Image:
    """Draw track tails up to the current frame."""
    draw = ImageDraw.Draw(image)
    start_tail = max(start_frame, frame - tail_length)

    subset = tracks_df[
        (tracks_df["t"] >= start_tail)
        & (tracks_df["t"] <= frame)
    ]

    for track_id, group in subset.groupby("track_id"):
        group = group.sort_values("t")
        if len(group) < 2:
            continue

        points = [
            (float(row["x"]) * scale, float(row["y"]) * scale)
            for _, row in group.iterrows()
        ]

        color = track_color(track_id)
        draw.line(points, fill=color, width=2)

        x, y = points[-1]
        radius = 2
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
        )

    return image


def add_frame_label(image: Image.Image, frame: int) -> Image.Image:
    """Add frame number in the upper-left corner."""
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), f"frame {frame}", fill=(255, 255, 255))
    return image


def make_frame(
    raw_file: str,
    labels: np.ndarray,
    tracks_df: pd.DataFrame,
    frame: int,
    scale: float,
    start_frame: int,
    label_alpha: float,
    tail_length: int,
    draw_tails: bool,
) -> Image.Image:
    """Create one rendered overlay frame."""
    raw = imageio.imread(raw_file)
    raw_norm = normalize_frame_for_display(raw)
    rgb = make_overlay_frame(raw_norm, labels, label_alpha=label_alpha)
    image = Image.fromarray(rgb)

    if scale != 1.0:
        new_size = (
            int(image.width * scale),
            int(image.height * scale),
        )
        image = image.resize(new_size, resample=Image.BILINEAR)

    if draw_tails:
        image = draw_track_tails(
            image=image,
            tracks_df=tracks_df,
            frame=frame,
            scale=scale,
            start_frame=start_frame,
            tail_length=tail_length,
        )

    return add_frame_label(image, frame)


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Completed Ultrack run directory")
    parser.add_argument("--raw-dir", required=True, help="Raw image directory")
    parser.add_argument("--tracked-labels", default=None, help="Default: <run-dir>/tracked_labels.zarr")
    parser.add_argument("--tracks-csv", default=None, help="Default: <run-dir>/tracks.csv")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=570)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--label-alpha", type=float, default=0.18)
    parser.add_argument("--tail-length", type=int, default=15)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--mode", choices=["png", "video", "both"], default="video")
    parser.add_argument("--no-track-tails", action="store_true")
    parser.add_argument("--video-name", default="tracks_overlay.mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    tracked_labels_path = (
        Path(args.tracked_labels).expanduser().resolve()
        if args.tracked_labels
        else run_dir / "tracked_labels.zarr"
    )
    tracks_csv = (
        Path(args.tracks_csv).expanduser().resolve()
        if args.tracks_csv
        else run_dir / "tracks.csv"
    )
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else run_dir / "overlay_export"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(glob.glob(str(raw_dir / RAW_PATTERN)))
    if not raw_files:
        raise FileNotFoundError(f"No raw files found in {raw_dir}")

    tracked = zarr.open(str(tracked_labels_path), mode="r")
    tracks_df = pd.read_csv(tracks_csv)

    draw_tails = not args.no_track_tails

    print("Run directory:", run_dir)
    print("Raw directory:", raw_dir)
    print("Tracked labels:", tracked_labels_path)
    print("Tracks CSV:", tracks_csv)
    print("Tracked labels shape:", tracked.shape)
    print("Frames:", args.start_frame, "to", args.end_frame)
    print("Output:", out_dir)

    if args.mode in ("png", "both"):
        png_dir = out_dir / "png_frames"
        png_dir.mkdir(parents=True, exist_ok=True)

        for frame in range(args.start_frame, args.end_frame + 1):
            if frame % 25 == 0:
                print(f"Writing PNG frame {frame}/{args.end_frame}")

            image = make_frame(
                raw_file=raw_files[frame],
                labels=np.asarray(tracked[frame]),
                tracks_df=tracks_df,
                frame=frame,
                scale=args.scale,
                start_frame=args.start_frame,
                label_alpha=args.label_alpha,
                tail_length=args.tail_length,
                draw_tails=draw_tails,
            )
            image.save(png_dir / f"overlay_frame_{frame:03d}.png")

        print("PNG frames:", png_dir)

    if args.mode in ("video", "both"):
        video_path = out_dir / args.video_name

        with imageio.get_writer(
            video_path,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
        ) as writer:
            for frame in range(args.start_frame, args.end_frame + 1):
                if frame % 25 == 0:
                    print(f"Writing video frame {frame}/{args.end_frame}")

                image = make_frame(
                    raw_file=raw_files[frame],
                    labels=np.asarray(tracked[frame]),
                    tracks_df=tracks_df,
                    frame=frame,
                    scale=args.scale,
                    start_frame=args.start_frame,
                    label_alpha=args.label_alpha,
                    tail_length=args.tail_length,
                    draw_tails=draw_tails,
                )
                writer.append_data(np.asarray(image))

        print("Video:", video_path)


if __name__ == "__main__":
    main()
