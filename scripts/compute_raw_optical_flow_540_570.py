from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from skimage.registration import optical_flow_tvl1
from skimage.transform import resize


# ------------------------------------------------------------
# Paths and frame range
# ------------------------------------------------------------

BASE = Path("<PROJECT_ROOT>/Tribolium_Daten")
RAW_DIR = BASE / "cylinder3_projections"

OUT_DIR = BASE / "optical_flow_540_570"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_FRAME = 540
END_FRAME = 570

# ------------------------------------------------------------
# Optical flow settings
# ------------------------------------------------------------

# Downscaling makes optical flow faster.
# 0.5 means half height and half width.
DOWNSCALE = 0.5

# Spatial bin size in full-resolution pixels.
BIN_SIZE = 80

# Foreground threshold on normalized raw images.
# Used to avoid averaging motion in completely dark background.
FOREGROUND_THRESHOLD = 0.05

# Minimum number of foreground pixels in a bin.
MIN_PIXELS_PER_BIN = 20

# Display/interpretation note:
# Depending on optical-flow convention, vector direction may need sign checking.
# Keep False for first run; visual comparison decides whether sign inversion is needed.
FLIP_SIGN = False


def normalize_frame(img):
    """Normalize one raw frame to [0, 1] using robust percentiles."""
    img = img.astype(np.float32)

    p1, p99 = np.percentile(img, (1, 99))
    img_norm = (img - p1) / (p99 - p1 + 1e-8)
    img_norm = np.clip(img_norm, 0, 1)

    return img_norm.astype(np.float32)


def load_raw_stack():
    """Load and normalize original raw frames 540-570."""
    raw_files_all = sorted(RAW_DIR.glob("extr_memb_cyl_2_MIP_tp_*.tif"))
    raw_files = raw_files_all[START_FRAME:END_FRAME + 1]

    if len(raw_files) != END_FRAME - START_FRAME + 1:
        raise RuntimeError(f"Expected 31 raw files, found {len(raw_files)}")

    stack = []

    for file_path in raw_files:
        img = imageio.imread(file_path)
        stack.append(normalize_frame(img))

    return np.stack(stack), raw_files


def downscale_frame(img, downscale):
    """Resize one frame for faster optical flow."""
    new_shape = (
        int(round(img.shape[0] * downscale)),
        int(round(img.shape[1] * downscale)),
    )

    return resize(
        img,
        new_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def compute_binned_flow_for_pair(frame_t, frame_next, t):
    """
    Compute optical flow between two consecutive frames and average it in bins.

    Output vectors are converted back to full-resolution pixel units per frame.
    """
    small_t = downscale_frame(frame_t, DOWNSCALE)
    small_next = downscale_frame(frame_next, DOWNSCALE)

    # skimage returns flow components in row/column order: v = dy, u = dx.
    v_small, u_small = optical_flow_tvl1(small_t, small_next)

    if FLIP_SIGN:
        v_small = -v_small
        u_small = -u_small

    # Convert low-resolution displacement to full-resolution pixel units.
    v_full_units = v_small / DOWNSCALE
    u_full_units = u_small / DOWNSCALE

    foreground = frame_t > FOREGROUND_THRESHOLD
    foreground_small = resize(
        foreground.astype(np.float32),
        small_t.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) > 0.5

    full_height, full_width = frame_t.shape

    y_edges = np.arange(0, full_height + BIN_SIZE, BIN_SIZE)
    x_edges = np.arange(0, full_width + BIN_SIZE, BIN_SIZE)

    rows = []

    for yb in range(len(y_edges) - 1):
        for xb in range(len(x_edges) - 1):
            y0_full = y_edges[yb]
            y1_full = min(y_edges[yb + 1], full_height)
            x0_full = x_edges[xb]
            x1_full = min(x_edges[xb + 1], full_width)

            y0_small = int(round(y0_full * DOWNSCALE))
            y1_small = int(round(y1_full * DOWNSCALE))
            x0_small = int(round(x0_full * DOWNSCALE))
            x1_small = int(round(x1_full * DOWNSCALE))

            if y1_small <= y0_small or x1_small <= x0_small:
                continue

            mask_bin = foreground_small[y0_small:y1_small, x0_small:x1_small]

            if int(mask_bin.sum()) < MIN_PIXELS_PER_BIN:
                continue

            dy_values = v_full_units[y0_small:y1_small, x0_small:x1_small][mask_bin]
            dx_values = u_full_units[y0_small:y1_small, x0_small:x1_small][mask_bin]

            if len(dy_values) == 0:
                continue

            mean_dy = float(np.mean(dy_values))
            mean_dx = float(np.mean(dx_values))
            speed = float(np.sqrt(mean_dy ** 2 + mean_dx ** 2))

            rows.append(
                {
                    "t": int(t),
                    "y_bin": int(yb),
                    "x_bin": int(xb),
                    "y": float((y0_full + y1_full) / 2.0),
                    "x": float((x0_full + x1_full) / 2.0),
                    "dy": mean_dy,
                    "dx": mean_dx,
                    "speed": speed,
                    "count": int(mask_bin.sum()),
                }
            )

    return rows


def main():
    raw_norm, raw_files = load_raw_stack()

    print("Raw stack:", raw_norm.shape, raw_norm.dtype)
    print("First raw:", raw_files[0])
    print("Last raw:", raw_files[-1])
    print("Output:", OUT_DIR)
    print("DOWNSCALE:", DOWNSCALE)
    print("BIN_SIZE:", BIN_SIZE)
    print("FOREGROUND_THRESHOLD:", FOREGROUND_THRESHOLD)
    print("FLIP_SIGN:", FLIP_SIGN)

    all_rows = []

    for t in range(raw_norm.shape[0] - 1):
        original_frame = START_FRAME + t
        print(f"Computing optical flow for original frame {original_frame} -> {original_frame + 1}")

        rows = compute_binned_flow_for_pair(
            raw_norm[t],
            raw_norm[t + 1],
            t=t,
        )
        all_rows.extend(rows)

        print("  binned vectors:", len(rows))

    flow_df = pd.DataFrame(all_rows)

    out_csv = OUT_DIR / "raw_optical_flow_binned.csv"
    out_npz = OUT_DIR / "raw_optical_flow_binned.npz"
    summary_path = OUT_DIR / "raw_optical_flow_summary.txt"

    flow_df.to_csv(out_csv, index=False)

    if len(flow_df) > 0:
        np.savez(
            out_npz,
            t=flow_df["t"].to_numpy(),
            y=flow_df["y"].to_numpy(),
            x=flow_df["x"].to_numpy(),
            dy=flow_df["dy"].to_numpy(),
            dx=flow_df["dx"].to_numpy(),
            speed=flow_df["speed"].to_numpy(),
            count=flow_df["count"].to_numpy(),
        )

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Raw-image optical flow summary\n")
        f.write("==============================\n\n")
        f.write(f"Frames: original {START_FRAME}-{END_FRAME}\n")
        f.write(f"Number of frame pairs: {raw_norm.shape[0] - 1}\n")
        f.write(f"Image shape: {raw_norm.shape[1:]}\n")
        f.write(f"Downscale: {DOWNSCALE}\n")
        f.write(f"Bin size: {BIN_SIZE}\n")
        f.write(f"Foreground threshold: {FOREGROUND_THRESHOLD}\n")
        f.write(f"Minimum pixels per bin: {MIN_PIXELS_PER_BIN}\n")
        f.write(f"Flip sign: {FLIP_SIGN}\n\n")
        f.write(f"Number of binned optical-flow vectors: {len(flow_df)}\n")

        if len(flow_df) > 0:
            f.write(f"Mean speed: {flow_df['speed'].mean():.4f} px/frame\n")
            f.write(f"Median speed: {flow_df['speed'].median():.4f} px/frame\n")
            f.write(f"Max speed: {flow_df['speed'].max():.4f} px/frame\n")

    print()
    print("Done.")
    print("Saved:", out_csv)
    print("Saved:", out_npz)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
