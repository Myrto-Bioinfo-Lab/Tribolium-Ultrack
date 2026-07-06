#!/usr/bin/env python3
"""
Compute derivative-based motion descriptors from a binned velocity field.

Input:
- binned_velocity_field.csv or raw_optical_flow_binned.csv
  Required columns: t, y_bin, x_bin, y, x, dy, dx, speed, count

Outputs:
- velocity_derivatives.csv
- derivative_summary_by_time.csv
- global_derivative_summary.txt
- heatmap PNGs for selected time points

Computed descriptors:
- divergence: d(dx)/dx + d(dy)/dy
- curl: d(dy)/dx - d(dx)/dy
- gradient_magnitude: Frobenius norm of velocity gradient
- strain_magnitude: symmetric deformation/strain-rate magnitude

All derivatives are approximated on the binned grid and are therefore exploratory.
They describe local patterns in the binned velocity field, not cell-level mechanics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_times(text: str | None) -> list[int] | None:
    if text is None or text.strip() == "":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def robust_abs_limit(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    lim = np.percentile(np.abs(finite), percentile)
    if not np.isfinite(lim) or lim == 0:
        lim = np.nanmax(np.abs(finite))
    if not np.isfinite(lim) or lim == 0:
        lim = 1.0
    return float(lim)


def frame_to_grid(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_bins = np.sort(frame["y_bin"].unique())
    x_bins = np.sort(frame["x_bin"].unique())

    y_index = {v: i for i, v in enumerate(y_bins)}
    x_index = {v: i for i, v in enumerate(x_bins)}

    shape = (len(y_bins), len(x_bins))
    u = np.full(shape, np.nan, dtype=float)  # dx
    v = np.full(shape, np.nan, dtype=float)  # dy
    speed = np.full(shape, np.nan, dtype=float)
    count = np.full(shape, np.nan, dtype=float)

    for row in frame.itertuples(index=False):
        yi = y_index[getattr(row, "y_bin")]
        xi = x_index[getattr(row, "x_bin")]
        u[yi, xi] = getattr(row, "dx")
        v[yi, xi] = getattr(row, "dy")
        speed[yi, xi] = getattr(row, "speed")
        count[yi, xi] = getattr(row, "count")

    return y_bins, x_bins, u, v, speed


def finite_gradient(arr: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    # np.gradient handles edges with one-sided differences.
    # Missing bins are filled with nearest neutral values only locally.
    filled = arr.copy()
    if np.isnan(filled).all():
        return np.full_like(arr, np.nan)

    # Simple iterative neighbor filling for isolated gaps.
    # This avoids huge NaN holes while keeping the calculation lightweight.
    mask = np.isnan(filled)
    if mask.any():
        global_mean = np.nanmean(filled)
        filled[mask] = global_mean

    grad = np.gradient(filled, spacing, axis=axis)
    grad[np.isnan(arr)] = np.nan
    return grad


def compute_derivatives_for_frame(frame: pd.DataFrame, bin_size: float) -> pd.DataFrame:
    y_bins, x_bins, u, v, speed_grid = frame_to_grid(frame)

    # Axis 1 is x direction, axis 0 is y direction.
    dudx = finite_gradient(u, bin_size, axis=1)
    dudy = finite_gradient(u, bin_size, axis=0)
    dvdx = finite_gradient(v, bin_size, axis=1)
    dvdy = finite_gradient(v, bin_size, axis=0)

    divergence = dudx + dvdy
    curl = dvdx - dudy
    gradient_magnitude = np.sqrt(dudx**2 + dudy**2 + dvdx**2 + dvdy**2)

    # Symmetric part of velocity gradient.
    exx = dudx
    eyy = dvdy
    exy = 0.5 * (dudy + dvdx)
    strain_magnitude = np.sqrt(exx**2 + eyy**2 + 2.0 * exy**2)

    records = []
    t_val = int(frame["t"].iloc[0])
    for yi, yb in enumerate(y_bins):
        for xi, xb in enumerate(x_bins):
            if np.isnan(u[yi, xi]) or np.isnan(v[yi, xi]):
                continue
            records.append(
                {
                    "t": t_val,
                    "y_bin": int(yb),
                    "x_bin": int(xb),
                    "x": float(frame[(frame["y_bin"] == yb) & (frame["x_bin"] == xb)]["x"].iloc[0]),
                    "y": float(frame[(frame["y_bin"] == yb) & (frame["x_bin"] == xb)]["y"].iloc[0]),
                    "dx": float(u[yi, xi]),
                    "dy": float(v[yi, xi]),
                    "speed": float(speed_grid[yi, xi]),
                    "dudx": float(dudx[yi, xi]),
                    "dudy": float(dudy[yi, xi]),
                    "dvdx": float(dvdx[yi, xi]),
                    "dvdy": float(dvdy[yi, xi]),
                    "divergence": float(divergence[yi, xi]),
                    "curl": float(curl[yi, xi]),
                    "gradient_magnitude": float(gradient_magnitude[yi, xi]),
                    "strain_magnitude": float(strain_magnitude[yi, xi]),
                }
            )
    return pd.DataFrame.from_records(records)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in df.groupby("t"):
        rows.append(
            {
                "t": int(t),
                "n_vectors": len(g),
                "mean_speed": g["speed"].mean(),
                "median_speed": g["speed"].median(),
                "mean_abs_divergence": g["divergence"].abs().mean(),
                "median_abs_divergence": g["divergence"].abs().median(),
                "mean_abs_curl": g["curl"].abs().mean(),
                "median_abs_curl": g["curl"].abs().median(),
                "mean_gradient_magnitude": g["gradient_magnitude"].mean(),
                "median_gradient_magnitude": g["gradient_magnitude"].median(),
                "mean_strain_magnitude": g["strain_magnitude"].mean(),
                "median_strain_magnitude": g["strain_magnitude"].median(),
                "max_strain_magnitude": g["strain_magnitude"].max(),
                "max_gradient_magnitude": g["gradient_magnitude"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("t")


def plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    time: int,
    out_dir: Path,
    title_prefix: str,
    symmetric: bool,
) -> None:
    g = df[df["t"] == time]
    if g.empty:
        print(f"Skipping {value_col} t={time}: no data")
        return

    y_bins = np.sort(g["y_bin"].unique())
    x_bins = np.sort(g["x_bin"].unique())
    mat = np.full((len(y_bins), len(x_bins)), np.nan)
    y_index = {v: i for i, v in enumerate(y_bins)}
    x_index = {v: i for i, v in enumerate(x_bins)}

    for row in g.itertuples(index=False):
        mat[y_index[getattr(row, "y_bin")], x_index[getattr(row, "x_bin")]] = getattr(row, value_col)

    fig, ax = plt.subplots(figsize=(8, 10))
    if symmetric:
        lim = robust_abs_limit(mat)
        im = ax.imshow(mat, origin="upper", vmin=-lim, vmax=lim, aspect="equal")
    else:
        finite = mat[np.isfinite(mat)]
        vmax = np.percentile(finite, 99) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        im = ax.imshow(mat, origin="upper", vmin=0, vmax=vmax, aspect="equal")

    ax.set_title(f"{title_prefix}: {value_col}, t={time}")
    ax.set_xlabel("x bin")
    ax.set_ylabel("y bin")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_col)

    out_path = out_dir / f"{title_prefix}_{value_col}_t{time:03d}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")


def write_global_summary(path: Path, source: Path, df: pd.DataFrame, summary: pd.DataFrame, bin_size: int) -> None:
    lines = [
        "Velocity derivative summary",
        "===========================",
        "",
        f"Input: {source}",
        f"Bin size: {bin_size} px",
        f"Frames: {int(df['t'].min())}-{int(df['t'].max())}",
        f"Total derivative vectors: {len(df)}",
        "",
        "Global statistics",
        "-----------------",
        f"Mean speed: {df['speed'].mean():.6f} px/frame",
        f"Median speed: {df['speed'].median():.6f} px/frame",
        f"Mean absolute divergence: {df['divergence'].abs().mean():.8f} 1/frame",
        f"Median absolute divergence: {df['divergence'].abs().median():.8f} 1/frame",
        f"Mean absolute curl: {df['curl'].abs().mean():.8f} 1/frame",
        f"Median absolute curl: {df['curl'].abs().median():.8f} 1/frame",
        f"Mean gradient magnitude: {df['gradient_magnitude'].mean():.8f} 1/frame",
        f"Median gradient magnitude: {df['gradient_magnitude'].median():.8f} 1/frame",
        f"Mean strain magnitude: {df['strain_magnitude'].mean():.8f} 1/frame",
        f"Median strain magnitude: {df['strain_magnitude'].median():.8f} 1/frame",
        "",
        "Most active frames by mean strain magnitude",
        "-------------------------------------------",
    ]

    top = summary.sort_values("mean_strain_magnitude", ascending=False).head(10)
    for row in top.itertuples(index=False):
        lines.append(
            f"t={int(row.t):03d}: mean_strain={row.mean_strain_magnitude:.8f}, "
            f"mean_abs_div={row.mean_abs_divergence:.8f}, "
            f"mean_abs_curl={row.mean_abs_curl:.8f}, "
            f"mean_speed={row.mean_speed:.6f}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bin-size", type=int, default=80)
    parser.add_argument("--times", default="0,10,20,29")
    parser.add_argument("--title-prefix", default=None)
    args = parser.parse_args()

    source = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    title_prefix = args.title_prefix or source.parent.name

    binned = pd.read_csv(source)
    required = {"t", "y_bin", "x_bin", "y", "x", "dy", "dx", "speed", "count"}
    missing = required - set(binned.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

    frames = []
    for _, frame in binned.groupby("t"):
        frames.append(compute_derivatives_for_frame(frame, args.bin_size))

    derivatives = pd.concat(frames, ignore_index=True)
    summary = compute_summary(derivatives)

    derivatives_path = out_dir / "velocity_derivatives.csv"
    summary_path = out_dir / "derivative_summary_by_time.csv"
    text_summary_path = out_dir / "global_derivative_summary.txt"

    derivatives.to_csv(derivatives_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_global_summary(text_summary_path, source, derivatives, summary, args.bin_size)

    print(f"Wrote {derivatives_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {text_summary_path}")

    heatmap_dir = out_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    times = parse_times(args.times)
    if times is None:
        times = sorted(int(t) for t in derivatives["t"].unique())

    for t in times:
        plot_heatmap(derivatives, "divergence", t, heatmap_dir, title_prefix, symmetric=True)
        plot_heatmap(derivatives, "curl", t, heatmap_dir, title_prefix, symmetric=True)
        plot_heatmap(derivatives, "strain_magnitude", t, heatmap_dir, title_prefix, symmetric=False)
        plot_heatmap(derivatives, "gradient_magnitude", t, heatmap_dir, title_prefix, symmetric=False)


if __name__ == "__main__":
    main()
