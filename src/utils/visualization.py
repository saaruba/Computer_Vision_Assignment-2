"""
Reusable plotting helpers for feature and tracking analysis.
This module contains standardized plotting functions used across the project.
All plots are saved directly to disk (no interactive display), which keeps and the pipeline terminal-friendly and reproducible.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def _ensure_output_dir(output_path):
    """
    Ensure the parent directory for an output file exists.

    Why this matters:
        Plot saving fails if the destination folder does not exist. This helper
        keeps plotting functions robust and avoids repeated directory checks.

    Args:
        output_path (str): Full path of the output file to be written.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def plot_line_over_frames(df, x_col, y_col, title, ylabel, output_path):
    """
    Plot a single feature/error line over frames and save it as an image.

    Args:
        df (pd.DataFrame): Data table containing plotting columns.
        x_col (str): Column name for x-axis (usually frame index).
        y_col (str): Column name for y-axis.
        title (str): Plot title.
        ylabel (str): Label for y-axis.
        output_path (str): Destination image path.
    """
    _ensure_output_dir(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df[y_col], linewidth=1.8)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    # Save figure with consistent publication-quality resolution.
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_two_lines_over_frames(
    df,
    x_col,
    y1_col,
    y2_col,
    title,
    y_label,
    label1,
    label2,
    output_path,
):
    
    # Plot two lines on the same frame axis for direct visual comparison.

    _ensure_output_dir(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df[y1_col], linewidth=1.8, label=label1)
    plt.plot(df[x_col], df[y2_col], linewidth=1.8, label=label2)
    plt.xlabel(x_col)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save without showing to keep scripts non-blocking in terminal runs.
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_normalized_features(df, frame_col, feature_cols, title, output_path):
    """
    Plot multiple features after per-feature min-max normalization.
    here i used normalization for Features may have very different value ranges. Normalizing each feature
    to [0, 1] allows fair visual comparison of trends over frames.

    """
    _ensure_output_dir(output_path)

    plt.figure(figsize=(9, 5))

    for col in feature_cols:
        values = df[col].to_numpy(dtype=float)

        # Handle empty/all-NaN columns safely to avoid runtime warnings/errors.
        if values.size == 0 or np.all(np.isnan(values)):
            normalized = np.zeros_like(values, dtype=float)
        else:
            col_min = np.nanmin(values)
            col_max = np.nanmax(values)

            # If a feature is constant, keep it at zero after normalization.
            if np.isclose(col_max, col_min):
                normalized = np.zeros_like(values, dtype=float)
            else:
                normalized = (values - col_min) / (col_max - col_min)

        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        plt.plot(df[frame_col], normalized, linewidth=1.8, label=col)

    plt.xlabel(frame_col)
    plt.ylabel("normalized_value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save all output plots at consistent high resolution.
    plt.savefig(output_path, dpi=300)
    plt.close()
