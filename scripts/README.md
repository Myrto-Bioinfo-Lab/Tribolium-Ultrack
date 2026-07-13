# Script overview

This directory contains the scripts used for the Tribolium/Ultrack workflow. The scripts are grouped by workflow stage and roughly ordered by execution order.

## Directory structure

- `01_pipeline/`: main Ultrack execution and core exports
- `02_preprocessing/`: input and candidate generation before Ultrack
- `03_analysis/`: run summaries, velocity fields and derivative analysis
- `04_visualization/`: Napari viewers, quiver plots and overlay videos

## Most important scripts

The central script for running Ultrack is:

- `01_pipeline/run_ultrack_pipeline.py`

It loads a YAML configuration, reads raw images or existing label masks, creates foreground and contour inputs, sets Ultrack parameters and runs the actual tracking.

The most important scripts for reproducing the core workflow are:

- `01_pipeline/run_ultrack_pipeline.py`
- `01_pipeline/export_ultrack_results.py`
- `03_analysis/compute_track_velocity_field_540_570.py`
- `03_analysis/compute_raw_optical_flow_540_570.py`
- `03_analysis/compute_velocity_derivatives.py`
- `03_analysis/make_final_analysis_outputs.py`

## 01_pipeline

- `run_ultrack_pipeline.py`: main YAML-based Ultrack pipeline.
- `export_ultrack_results.py`: exports `tracks.csv`, `track_summary.csv`, `tracks_per_frame.csv` and `export_summary.txt`.
- `export_tracked_labels.py`: exports `tracked_labels.zarr` from a completed Ultrack run.

Example:

```bash
python scripts/01_pipeline/run_ultrack_pipeline.py --config configs/multi_existing_labels_540_570.yaml
```

## 02_preprocessing

These scripts create input variants or segmentation candidates before Ultrack is run.

- `create_gamma_inputs.py`: creates gamma-corrected raw image input folders.
- `create_independent_gamma_inputs_540_570.py`: creates gamma-preprocessed inputs specifically for original frames 540-570.
- `run_independent_cellpose_candidates_540_570.sh`: runs Cellpose/CPSAM on raw and gamma-preprocessed inputs to generate independent candidate masks.
- `create_independent_classic_candidates_540_570.py`: creates classical threshold- and watershed-based segmentation candidates.
- `create_independent_ultrack_imgproc_candidates_540_570.py`: creates raw-image preprocessing candidates such as foreground masks, contour signals and Canny edge images.

Example:

```bash
bash scripts/02_preprocessing/run_independent_cellpose_candidates_540_570.sh
```

## 03_analysis

These scripts summarize completed runs and compute motion-related quantities.

- `collect_run_summaries.py`: collects compact summary statistics from selected Ultrack runs.
- `collect_full_frame_analysis.py`: collects extended full-frame tracking statistics, including short-track and long-track counts.
- `compute_track_velocity_field_540_570.py`: computes track-based step vectors and binned velocity fields for selected runs in frames 540-570.
- `compute_full_frame_velocity_field.py`: computes a track-based velocity field for a selected full-frame run.
- `compute_raw_optical_flow_540_570.py`: computes raw-image optical flow directly from original frames 540-570.
- `compute_velocity_derivatives.py`: computes divergence, curl, gradient magnitude and strain magnitude from binned velocity fields.
- `make_final_analysis_outputs.py`: creates final summary plots, velocity comparisons, quiver images and optional videos.

Example:

```bash
python scripts/03_analysis/compute_velocity_derivatives.py --csv results/velocity_fields_540_570/multi_existing_labels_540_570/binned_velocity_field.csv --out-dir results/derivatives/multi_existing_labels_540_570 --bin-size 80
```

## 04_visualization

These scripts are used for visual inspection and presentation outputs.

- `view_velocity_field_napari.py`: opens one binned velocity field in Napari, optionally with raw images as background.
- `view_velocity_fields_540_570.py`: opens several track-based velocity fields for frames 540-570 in Napari.
- `view_track_vs_optical_flow_540_570.py`: opens selected track-based velocity fields and raw-image optical flow together in Napari.
- `view_segmentation_tracking_540_570.py`: opens raw images, segmentation candidates, preprocessing candidates, tracked labels, tracks and velocity vectors for frames 540-570.
- `plot_velocity_quiver_scaled.py`: creates static quiver plots with visually enlarged arrows; numerical velocity values are not changed.
- `plot_full_frame_comparison.py`: creates bar plots from the full-frame tracking comparison table.
- `export_track_overlay_video.py`: creates overlay PNG frames or MP4 videos from raw images, tracked labels and optional track tails.

Example:

```bash
python scripts/04_visualization/view_segmentation_tracking_540_570.py --data-root ../Tribolium_Daten
```

## Recommended execution order

A typical reproduction follows this order:

1. Generate optional preprocessing inputs and candidate masks.
2. Run Ultrack with `01_pipeline/run_ultrack_pipeline.py`.
3. Export completed runs with `01_pipeline/export_ultrack_results.py`.
4. Optionally export `tracked_labels.zarr` with `01_pipeline/export_tracked_labels.py`.
5. Collect run summaries.
6. Compute track-based velocity fields and raw-image optical flow.
7. Compute derivative-based motion descriptors.
8. Create final plots, quiver images, videos and Napari visualizations.
