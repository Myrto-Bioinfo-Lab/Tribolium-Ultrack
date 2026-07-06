import argparse
import glob
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import imageio.v2 as imageio
import numpy as np
import yaml

from skimage.filters import gaussian
from skimage.morphology import remove_small_holes, remove_small_objects

from ultrack import MainConfig, Tracker
from ultrack.imgproc import detect_foreground, robust_invert
from ultrack.utils import labels_to_contours


# ------------------------------------------------------------
# Small helper functions
# ------------------------------------------------------------

def load_yaml_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_package_version(package_name):
    """Return the installed package version or a fallback string."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def format_seconds(seconds):
    """Format seconds as a readable string."""
    minutes = seconds / 60.0
    hours = seconds / 3600.0
    return f"{seconds:.2f} s | {minutes:.2f} min | {hours:.2f} h"


def select_file_range(files, n_frames=None, start_frame=None, end_frame=None):
    """Select files either by start/end frame or by the first n_frames."""
    if start_frame is not None or end_frame is not None:
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = len(files) - 1

        if start_frame < 0:
            raise ValueError("start_frame must be >= 0")

        if end_frame >= len(files):
            raise ValueError(
                f"end_frame={end_frame} is too large. "
                f"Only {len(files)} files found, last valid frame is {len(files) - 1}."
            )

        if start_frame > end_frame:
            raise ValueError("start_frame must be <= end_frame")

        return files[start_frame:end_frame + 1]

    if n_frames is not None:
        return files[:n_frames]

    return files


def prepare_output_dir(config, config_path):
    """Create the output directory and save a copy of the used config."""
    output_dir = Path(config["output"]["output_dir"])
    overwrite = config["output"].get("overwrite", False)

    if output_dir.exists():
        if overwrite:
            print("Removing old output directory:")
            print(output_dir)
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Set output.overwrite: true in the YAML file if you want to overwrite it."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    if config["export"].get("save_config_used", True):
        shutil.copy(config_path, output_dir / "config_used.yaml")

    # Save a copy of this pipeline script for reproducibility.
    script_path = Path(__file__).resolve()
    shutil.copy(script_path, output_dir / "run_ultrack_pipeline_used.py")

    return output_dir


def write_status(output_dir, status, message=""):
    """Write a small status file for the run."""
    status_path = output_dir / "status.txt"
    with open(status_path, "w") as f:
        f.write(f"status: {status}\n")
        f.write(f"time: {datetime.now()}\n")
        if message:
            f.write(f"message: {message}\n")


def append_run_info(output_dir, text):
    """Append text to run_info.txt."""
    with open(output_dir / "run_info.txt", "a") as f:
        f.write(text)


def write_run_info_start(output_dir, config, start_time):
    """Write initial run information."""
    run_info_path = output_dir / "run_info.txt"

    input_config = config["input"]

    with open(run_info_path, "w") as f:
        f.write("Ultrack pipeline run information\n")
        f.write("================================\n\n")

        f.write("Run metadata\n")
        f.write("------------\n")
        f.write(f"Run name: {config['name']}\n")
        f.write(f"Start time: {start_time}\n")
        f.write(f"Pipeline mode: {config['pipeline']['mode']}\n")
        f.write(f"Output directory: {config['output']['output_dir']}\n\n")

        f.write("Input\n")
        f.write("-----\n")
        f.write(f"Input directory: {input_config['input_dir']}\n")
        f.write(f"File pattern: {input_config['file_pattern']}\n")
        f.write(f"n_frames: {input_config.get('n_frames', None)}\n")
        f.write(f"start_frame: {input_config.get('start_frame', None)}\n")
        f.write(f"end_frame: {input_config.get('end_frame', None)}\n\n")

        crop = input_config["crop"]
        f.write("Crop\n")
        f.write("----\n")
        f.write(f"enabled: {crop['enabled']}\n")
        f.write(f"y_start: {crop['y_start']}\n")
        f.write(f"y_end: {crop['y_end']}\n")
        f.write(f"x_start: {crop['x_start']}\n")
        f.write(f"x_end: {crop['x_end']}\n\n")

        f.write("Notes\n")
        f.write("-----\n")
        f.write(f"{config.get('notes', {}).get('description', '')}\n\n")


def write_versions(output_dir):
    """Write relevant software versions to run_info.txt."""
    packages = [
        "numpy",
        "imageio",
        "scikit-image",
        "pandas",
        "ultrack",
        "cellpose",
        "torch",
        "napari",
        "vispy",
        "zarr",
        "gurobipy",
        "pyyaml",
    ]

    text = "\nSoftware versions\n"
    text += "-----------------\n"
    text += "Python packages:\n"

    for package in packages:
        text += f"{package}: {get_package_version(package)}\n"

    text += "\n"
    append_run_info(output_dir, text)


# ------------------------------------------------------------
# Image loading
# ------------------------------------------------------------

def load_raw_images(config):
    """Load image stack according to the YAML config.

    Depending on the pipeline mode, these can be raw TIFFs or label masks.
    """
    input_config = config["input"]

    input_dir = input_config["input_dir"]
    file_pattern = input_config["file_pattern"]
    n_frames = input_config.get("n_frames", None)
    start_frame = input_config.get("start_frame", None)
    end_frame = input_config.get("end_frame", None)

    files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No files found in {input_dir} with pattern {file_pattern}"
        )

    print("Found files:", len(files))
    print("First file:", files[0])
    print("Last file:", files[-1])

    files = select_file_range(
        files,
        n_frames=n_frames,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    print("Used files:", len(files))
    print("First used file:", files[0])
    print("Last used file:", files[-1])

    crop = input_config["crop"]
    use_crop = crop["enabled"]

    images = []

    for file_path in files:
        img = imageio.imread(file_path)

        if use_crop:
            y_start = crop["y_start"]
            y_end = crop["y_end"]
            x_start = crop["x_start"]
            x_end = crop["x_end"]
            img = img[y_start:y_end, x_start:x_end]

        images.append(img)

    stack = np.stack(images)

    return stack, files


def load_label_stack_from_source(source, config):
    """Load one label stack from a multi-label source dictionary."""
    input_config = config["input"]

    input_dir = source["input_dir"]
    file_pattern = source["file_pattern"]

    n_frames = input_config.get("n_frames", None)
    start_frame = input_config.get("start_frame", None)
    end_frame = input_config.get("end_frame", None)

    files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No files found in {input_dir} with pattern {file_pattern}"
        )

    files = select_file_range(
        files,
        n_frames=n_frames,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    print()
    print("Loading label source:", source.get("name", input_dir))
    print("Input directory:", input_dir)
    print("Number of used files:", len(files))
    print("First used file:", files[0])
    print("Last used file:", files[-1])

    labels = []

    for file_path in files:
        img = imageio.imread(file_path)
        labels.append(img)

    labels = np.stack(labels)

    print("labels.shape:", labels.shape)
    print("labels.dtype:", labels.dtype)
    print("labels min/max:", labels.min(), labels.max())

    return labels


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

def normalize_images(raw, config):
    """Normalize each frame using percentile normalization and optional gamma correction."""
    norm_config = config["normalization"]

    if not norm_config["enabled"]:
        return raw.astype(np.float32)

    p_low = norm_config["percentile_low"]
    p_high = norm_config["percentile_high"]
    gamma = norm_config["gamma"]

    raw_norm = np.zeros(raw.shape, dtype=np.float32)

    for t in range(raw.shape[0]):
        img = raw[t].astype(np.float32)

        p1, p99 = np.percentile(img, (p_low, p_high))

        img_norm = (img - p1) / (p99 - p1 + 1e-8)
        img_norm = np.clip(img_norm, 0, 1)

        if gamma is not None and gamma != 1.0:
            img_norm = np.power(img_norm, gamma)

        raw_norm[t] = img_norm

    return raw_norm


def create_ultrack_imgproc_inputs(raw_norm, config):
    """Create foreground and contours using Ultrack's image processing functions."""
    imgproc_config = config["ultrack_imgproc"]

    fg_config = imgproc_config["detect_foreground"]
    inv_config = imgproc_config["robust_invert"]

    print("Creating foreground with detect_foreground...")
    foreground = detect_foreground(
        raw_norm,
        sigma=fg_config["sigma"],
        remove_hist_mode=fg_config["remove_hist_mode"],
        min_foreground=fg_config["min_foreground"],
    )

    print("Creating contours with robust_invert...")
    contours = robust_invert(
        raw_norm,
        sigma=inv_config["sigma"],
        lower_quantile=inv_config["lower_quantile"],
        upper_quantile=inv_config["upper_quantile"],
    )

    foreground = foreground.astype(np.uint8)
    contours = contours.astype(np.float32)

    return foreground, contours


def create_threshold_robust_inputs(raw_norm, config):
    """Create a generous threshold foreground and robust_invert contours."""
    raw_config = config["raw_sobel"]
    inv_config = config["ultrack_imgproc"]["robust_invert"]

    foreground = np.zeros(raw_norm.shape, dtype=np.uint8)

    sigma = raw_config["gaussian_sigma"]
    threshold = raw_config["threshold"]
    min_size = raw_config["min_size"]
    hole_area_threshold = raw_config["hole_area_threshold"]

    print("Creating generous threshold foreground...")
    print("threshold:", threshold)
    print("gaussian_sigma:", sigma)

    for t in range(raw_norm.shape[0]):
        img = raw_norm[t]

        smooth = gaussian(img, sigma=sigma, preserve_range=True)
        fg = smooth > threshold

        fg = remove_small_objects(fg, min_size=min_size)
        fg = remove_small_holes(fg, area_threshold=hole_area_threshold)

        border_config = config["preprocessing"]["border"]

        if border_config["enabled"]:
            top = border_config["top"]
            bottom = border_config["bottom"]
            left = border_config["left"]
            right = border_config["right"]

            if top > 0:
                fg[:top, :] = False
            if bottom > 0:
                fg[-bottom:, :] = False
            if left > 0:
                fg[:, :left] = False
            if right > 0:
                fg[:, -right:] = False

        foreground[t] = fg.astype(np.uint8)

        print(
            f"Frame {t}: threshold={threshold:.4f}, "
            f"foreground pixels={foreground[t].sum()}"
        )

    print("Creating contours with robust_invert...")
    contours = robust_invert(
        raw_norm,
        sigma=inv_config["sigma"],
        lower_quantile=inv_config["lower_quantile"],
        upper_quantile=inv_config["upper_quantile"],
    ).astype(np.float32)

    return foreground, contours


def create_existing_labels_inputs(labels, config):
    """Create Ultrack foreground and contours from one existing label stack."""
    existing_config = config.get("existing_labels", {})
    sigma = existing_config.get("labels_to_contours_sigma", None)

    print("Creating foreground and contours from existing labels...")
    print("labels.shape:", labels.shape)
    print("labels.dtype:", labels.dtype)
    print("labels min/max:", labels.min(), labels.max())
    print("labels_to_contours sigma:", sigma)

    foreground, contours = labels_to_contours(
        labels,
        sigma=sigma,
        overwrite=True,
    )

    foreground = np.asarray(foreground).astype(np.uint8)
    contours = np.asarray(contours).astype(np.float32)

    return foreground, contours


def create_multi_existing_labels_inputs(config):
    """Create Ultrack foreground and contours from multiple label stacks."""
    multi_config = config["multi_existing_labels"]
    sources = multi_config["sources"]
    sigma = multi_config.get("labels_to_contours_sigma", None)

    label_stacks = []

    print("Creating foreground and contours from multiple existing label sources...")
    print("Number of label sources:", len(sources))
    print("labels_to_contours sigma:", sigma)

    for source in sources:
        labels = load_label_stack_from_source(source, config)
        label_stacks.append(labels)

    print()
    print("Calling labels_to_contours with multiple label stacks...")

    foreground, contours = labels_to_contours(
        label_stacks,
        sigma=sigma,
        overwrite=True,
    )

    foreground = np.asarray(foreground).astype(np.uint8)
    contours = np.asarray(contours).astype(np.float32)

    return foreground, contours, label_stacks


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run an Ultrack experiment from a YAML config."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    total_start_wall_time = datetime.now()
    total_start_perf_time = time.perf_counter()

    phase_times = {}

    try:
        config = load_yaml_config(config_path)

        print("Loaded config:")
        print(config_path)
        print("Run name:", config["name"])
        print("Pipeline mode:", config["pipeline"]["mode"])

        output_dir = prepare_output_dir(config, config_path)

        write_status(output_dir, "running")
        write_run_info_start(output_dir, config, total_start_wall_time)

        # ----------------------------------------------------
        # Phase 1: load primary image stack
        # ----------------------------------------------------
        phase_start = time.perf_counter()

        raw, used_files = load_raw_images(config)

        phase_times["load_images"] = time.perf_counter() - phase_start

        print("raw.shape:", raw.shape)
        print("raw.dtype:", raw.dtype)
        print("raw min/max:", raw.min(), raw.max())

        append_run_info(
            output_dir,
            "Loaded primary image data\n"
            "-------------------------\n"
            f"Number of used files: {len(used_files)}\n"
            f"First used file: {used_files[0]}\n"
            f"Last used file: {used_files[-1]}\n"
            f"raw.shape: {raw.shape}\n"
            f"raw.dtype: {raw.dtype}\n"
            f"raw min/max: {raw.min()} / {raw.max()}\n\n",
        )

        # ----------------------------------------------------
        # Phase 2: normalization
        # ----------------------------------------------------
        mode = config["pipeline"]["mode"]

        if mode in ("existing_labels", "multi_existing_labels"):
            raw_norm = None
            phase_times["normalization"] = 0.0

            append_run_info(
                output_dir,
                "Normalized image data\n"
                "---------------------\n"
                "Skipped because pipeline mode uses existing label masks.\n"
                "The loaded images are label masks and label IDs must be preserved.\n\n",
            )

        else:
            phase_start = time.perf_counter()

            raw_norm = normalize_images(raw, config)

            phase_times["normalization"] = time.perf_counter() - phase_start

            print("raw_norm.shape:", raw_norm.shape)
            print("raw_norm.dtype:", raw_norm.dtype)
            print("raw_norm min/max:", raw_norm.min(), raw_norm.max())

            append_run_info(
                output_dir,
                "Normalized image data\n"
                "---------------------\n"
                f"raw_norm.shape: {raw_norm.shape}\n"
                f"raw_norm.dtype: {raw_norm.dtype}\n"
                f"raw_norm min/max: {raw_norm.min()} / {raw_norm.max()}\n"
                f"percentile_low: {config['normalization']['percentile_low']}\n"
                f"percentile_high: {config['normalization']['percentile_high']}\n"
                f"gamma: {config['normalization']['gamma']}\n\n",
            )

        # ----------------------------------------------------
        # Phase 3: create foreground and contours
        # ----------------------------------------------------
        phase_start = time.perf_counter()

        label_stacks = None

        if mode == "ultrack_imgproc":
            foreground, contours = create_ultrack_imgproc_inputs(raw_norm, config)

        elif mode == "threshold_robust":
            foreground, contours = create_threshold_robust_inputs(raw_norm, config)

        elif mode == "existing_labels":
            foreground, contours = create_existing_labels_inputs(raw, config)

        elif mode == "multi_existing_labels":
            foreground, contours, label_stacks = create_multi_existing_labels_inputs(config)

        else:
            raise NotImplementedError(
                f"Pipeline mode is not implemented yet: {mode}"
            )

        phase_times["create_foreground_contours"] = time.perf_counter() - phase_start

        print("foreground.shape:", foreground.shape)
        print("foreground.dtype:", foreground.dtype)
        print("foreground min/max:", foreground.min(), foreground.max())
        print("foreground pixels:", foreground.sum())

        print("contours.shape:", contours.shape)
        print("contours.dtype:", contours.dtype)
        print("contours min/max:", contours.min(), contours.max())

        append_run_info(
            output_dir,
            "Foreground and contours\n"
            "-----------------------\n"
            f"method: {mode}\n"
            f"foreground.shape: {foreground.shape}\n"
            f"foreground.dtype: {foreground.dtype}\n"
            f"foreground min/max: {foreground.min()} / {foreground.max()}\n"
            f"foreground pixels: {foreground.sum()}\n"
            f"contours.shape: {contours.shape}\n"
            f"contours.dtype: {contours.dtype}\n"
            f"contours min/max: {contours.min()} / {contours.max()}\n\n",
        )

        # ----------------------------------------------------
        # Phase 4: save intermediate data
        # ----------------------------------------------------
        phase_start = time.perf_counter()

        if config["export"].get("save_raw_norm", True) and raw_norm is not None:
            np.save(output_dir / "raw_norm.npy", raw_norm)

        if mode == "existing_labels":
            np.save(output_dir / "labels.npy", raw)

        if mode == "multi_existing_labels" and label_stacks is not None:
            for idx, labels in enumerate(label_stacks):
                print(f"Skipping labels_source_{idx}.npy save to avoid large temporary files; labels shape: {labels.shape}")

        if config["export"].get("save_foreground", True):
            np.save(output_dir / "foreground.npy", foreground)

        if config["export"].get("save_contours", True):
            np.save(output_dir / "contours.npy", contours)

        phase_times["save_intermediate"] = time.perf_counter() - phase_start

        # ----------------------------------------------------
        # Phase 5: run Ultrack tracking
        # ----------------------------------------------------
        phase_start = time.perf_counter()

        ultrack_config = MainConfig()
        ultrack_config.data_config.working_dir = str(output_dir)

        ultrack_config.segmentation_config.min_area = config["ultrack"]["min_area"]
        ultrack_config.segmentation_config.max_area = config["ultrack"]["max_area"]

        min_frontier = config["ultrack"].get("min_frontier", None)
        if min_frontier is not None:
            ultrack_config.segmentation_config.min_frontier = min_frontier

        ultrack_config.linking_config.max_distance = config["ultrack"]["max_distance"]
        ultrack_config.linking_config.max_neighbors = config["ultrack"]["max_neighbors"]

        tracker = Tracker(config=ultrack_config)

        tracker.track(
            foreground=foreground,
            contours=contours,
        )

        phase_times["ultrack_tracking"] = time.perf_counter() - phase_start

        print()
        print("Image loading, preprocessing, and Ultrack tracking finished.")

        # ----------------------------------------------------
        # Write timing and versions
        # ----------------------------------------------------
        total_end_wall_time = datetime.now()
        total_runtime = time.perf_counter() - total_start_perf_time

        append_run_info(output_dir, "Timing\n")
        append_run_info(output_dir, "------\n")

        for phase_name, seconds in phase_times.items():
            append_run_info(output_dir, f"{phase_name}: {format_seconds(seconds)}\n")

        append_run_info(output_dir, f"total_runtime: {format_seconds(total_runtime)}\n")
        append_run_info(output_dir, f"End time: {total_end_wall_time}\n\n")

        write_versions(output_dir)
        write_status(output_dir, "completed")

        print()
        print("Pipeline finished successfully.")
        print("Output directory:")
        print(output_dir)

    except Exception as error:
        print("ERROR:")
        print(error)

        try:
            output_dir
            write_status(output_dir, "failed", str(error))
            append_run_info(output_dir, "\nFailure\n")
            append_run_info(output_dir, "-------\n")
            append_run_info(output_dir, f"{error}\n")
        except Exception:
            pass

        raise


if __name__ == "__main__":
    main()
