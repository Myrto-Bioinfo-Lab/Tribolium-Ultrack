# Tribolium Serosa Ultrack Motion Workflow

This repository documents a workflow for segmentation, tracking and motion analysis of a 2D time series of the extraembryonic membranes of *Tribolium*. The focus is on serosa motion, especially in the late part of the sequence before tissue rupture.

## Overview

The analysed time series contains 571 frames. Raw-image-based segmentation and Ultrack approaches were tested first. These raw-only variants served as methodological controls, but showed strong fragmentation and were not visually stable enough for the final analysis.

More stable results were obtained with label-based multi-input approaches. In these runs, several Cellpose/CPSAM segmentation variants were provided to Ultrack as alternative candidate sources.

The original frames 540--570 were analysed in more detail because this window is biologically relevant: serosa motion increases visibly in this late phase, and the tissue approaches rupture.

## Main findings

- Raw-only, threshold-based, watershed-based and contour-only approaches were not sufficiently stable for these data.
- Label-based multi-input approaches produced more coherent tracking results.
- The full-frame run based on automatically generated Cellpose variants was statistically very similar to a multi-input run using project-specific reference labels.
- For frames 540--570, `multi_existing_labels_540_570` remained the most coherent visual reference run.
- The independent run `independent_multi_cpsam_candidates_540_570` is methodologically important because it does not rely on project-specific reference labels or a project-specific Cellpose model.
- Track-based velocity fields and raw-image optical flow show similar global motion patterns, but differ locally and in the presence of outliers.
- Divergence, curl, gradient magnitude and strain magnitude provide exploratory indicators of local expansion, rotation and deformation.

## Repository structure

```text
configs/          YAML configuration files for the documented Ultrack runs
scripts/          Scripts for tracking, export, velocity fields and plots
documentation/    Final workflow reports as PDF files
results/          Tables, summaries and selected CSV outputs
plots/            Final plots, quiver images and heatmaps
assets/videos/    Compressed videos for tracking and motion fields
```

## Data note

The repository contains scripts, configurations, compact result files, plots, videos and documentation. Large raw data and intermediate processing outputs are not included, such as the original TIFF image sequence, full Cellpose output folders, Ultrack databases, Zarr/NPY/NPZ files, Conda environments and temporary test outputs.

To rerun the complete workflow from raw data, the original image data must be available locally and paths in the YAML configuration files may need to be adapted.

## Key result files

```text
results/tables/run_summary_comparison.csv
results/tables/full_frame_analysis_comparison.csv
results/tables/velocity_field_comparison.csv
results/velocity_fields_540_570/
results/optical_flow_540_570/
results/velocity_fields_full/
results/derivatives/
```

## Videos

Full-frame tracking overlays:

```text
assets/videos/full_frame_tracks/multi_cellpose_variants_no_baseline_tracks_overlay_compressed.mp4
assets/videos/full_frame_tracks/multi_cellpose_only_tracks_overlay_compressed.mp4
```

Motion-field videos for frames 540--570:

```text
assets/videos/motion_fields/multi_existing_labels_540_570_scaled_quiver.mp4
assets/videos/motion_fields/independent_multi_cpsam_candidates_scaled_quiver.mp4
assets/videos/motion_fields/raw_optical_flow_scaled_quiver.mp4
```

For the quiver videos and quiver plots, arrow lengths were visually scaled to improve readability. The numerical velocity values in the CSV files remain unchanged in px/frame.


## Environment

The workflow was run in a Conda environment. A compact environment specification and package list are included:

```text
environment.yml
environment_conda_list.txt
```

## Important scripts
A detailed overview of all scripts, their roles and the recommended execution order is provided in `scripts/README.md`.

```text
scripts/01_pipeline/run_ultrack_pipeline.py
scripts/01_pipeline/export_ultrack_results.py
scripts/03_analysis/collect_run_summaries.py
scripts/03_analysis/collect_full_frame_analysis.py
scripts/03_analysis/compute_track_velocity_field_540_570.py
scripts/03_analysis/compute_raw_optical_flow_540_570.py
scripts/03_analysis/compute_full_frame_velocity_field.py
scripts/03_analysis/compute_velocity_derivatives.py
scripts/03_analysis/make_final_analysis_outputs.py
scripts/04_visualization/plot_velocity_quiver_scaled.py
scripts/04_visualization/view_velocity_field_napari.py
```

## Additional documentation

A focused full-frame tracking comparison is included as:

```text
documentation/full_frame_tracking_comparison.pdf
```

The final workflow report is included as:

```text
documentation/tribolium_ultrack_workflow_protokoll_final.pdf
```

## Interpretation

Global track statistics are useful but not sufficient on their own. For this reason, quantitative statistics, visual inspection, velocity fields and derivative-based motion descriptors were considered together.

The analysis does not fully explain the mechanical cause of tissue rupture. It provides a reproducible basis for further investigation of local motion, expansion, rotation and deformation of the serosa before rupture.
