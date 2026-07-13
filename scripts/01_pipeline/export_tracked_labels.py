#!/usr/bin/env python3
"""Export tracked label images from a completed Ultrack run.

This script reads a completed Ultrack output directory, loads the solved tracks,
and converts them into a tracked label image using ultrack.tracks_to_zarr.

The resulting Zarr label image can be used for visual inspection in Napari or
for generating overlay videos. The same tracked object keeps the same label ID
over time.

The script does not run segmentation or tracking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultrack import load_config, to_tracks_layer, tracks_to_zarr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed Ultrack run directory containing metadata.toml",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output Zarr path. Default: <run-dir>/tracked_labels.zarr",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output Zarr store.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    metadata_path = run_dir / "metadata.toml"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.toml: {metadata_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else run_dir / "tracked_labels.zarr"

    config = load_config(metadata_path)
    config.data_config.working_dir = str(run_dir)

    tracks_df, graph = to_tracks_layer(config)

    print("Run directory:", run_dir)
    print("Tracks table:", tracks_df.shape)
    print("Output:", out_path)

    tracks_to_zarr(
        config,
        tracks_df,
        store_or_path=out_path,
        overwrite=args.overwrite,
    )

    print("Done:", out_path)


if __name__ == "__main__":
    main()
