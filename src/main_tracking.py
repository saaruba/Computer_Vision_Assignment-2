import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tracking.kalman_rotation import RotationKalmanFilter
from tracking.kalman_translation import TranslationKalmanFilter


def compute_rotation_error(pred_theta, gt_theta):
    """
    Compute wrapped rotation error in degrees.

    Args:
        pred_theta (float): Predicted angle in degrees.
        gt_theta (float): Ground-truth angle in degrees.

    Returns:
        float: Rotation error using min(|d|, 180 - |d|).
    """
    angle_diff = abs(pred_theta - gt_theta)
    return min(angle_diff, 180.0 - angle_diff)


def main():
    """Run Kalman tracking evaluation for translation and rotation."""
    input_csv_path = "results/tables/shape_features.csv"
    tables_dir = "results/tables"
    plots_dir = "results/plots"
    output_csv_path = os.path.join(tables_dir, "tracking_errors.csv")

    # Ensure output directories exist.
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Load feature table.
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)

    required_columns = ["frame_index", "centroid_x", "centroid_y", "orientation_deg"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in shape features CSV: {missing_columns}")

    # Keep required columns, remove missing values, and keep only frames 0..50.
    df = df[required_columns].copy()
    df = df.dropna(subset=["centroid_x", "centroid_y", "orientation_deg"]).copy()
    df["frame_index"] = df["frame_index"].astype(int)
    df = df[(df["frame_index"] >= 0) & (df["frame_index"] <= 50)].copy()
    df = df.sort_values("frame_index")
    df = df.drop_duplicates(subset="frame_index", keep="first")

    if df.empty:
        raise ValueError("No valid rows available after dropping NaNs and filtering frames 0..50.")

    # Build quick lookup by frame index.
    gt_by_frame = df.set_index("frame_index")

    if 0 not in gt_by_frame.index:
        raise ValueError("Frame 0 is required for Kalman filter initialization.")

    # Initialize translation and rotation Kalman filters using frame 0 ground truth.
    frame0 = gt_by_frame.loc[0]
    translation_kf = TranslationKalmanFilter(dt=1.0, process_noise=1.0, measurement_noise=5.0)
    rotation_kf = RotationKalmanFilter(dt=1.0, process_noise=1.0, measurement_noise=5.0)

    translation_kf.initialize(frame0["centroid_x"], frame0["centroid_y"])
    rotation_kf.initialize(frame0["orientation_deg"])

    # Frames 0..40: filtering/tuning stage (predict then update when GT exists).
    for frame_idx in range(1, 41):
        translation_kf.predict()
        rotation_kf.predict()

        if frame_idx in gt_by_frame.index:
            gt_row = gt_by_frame.loc[frame_idx]
            translation_kf.update(gt_row["centroid_x"], gt_row["centroid_y"])
            rotation_kf.update(gt_row["orientation_deg"])

    # Frames 41..50: prediction-only evaluation (no updates).
    results = []
    for frame_idx in range(41, 51):
        translation_kf.predict()
        rotation_kf.predict()

        if frame_idx not in gt_by_frame.index:
            continue

        gt_row = gt_by_frame.loc[frame_idx]
        pred_x, pred_y = translation_kf.get_position()
        pred_theta = rotation_kf.get_angle()

        gt_x = float(gt_row["centroid_x"])
        gt_y = float(gt_row["centroid_y"])
        gt_theta = rotation_kf.normalize_angle(gt_row["orientation_deg"])

        translation_error = float(np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2))
        rotation_error = float(compute_rotation_error(pred_theta, gt_theta))

        results.append(
            {
                "frame_index": frame_idx,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "translation_error": translation_error,
                "pred_theta": pred_theta,
                "gt_theta": gt_theta,
                "rotation_error": rotation_error,
            }
        )

    tracking_errors_df = pd.DataFrame(
        results,
        columns=[
            "frame_index",
            "pred_x",
            "pred_y",
            "gt_x",
            "gt_y",
            "translation_error",
            "pred_theta",
            "gt_theta",
            "rotation_error",
        ],
    )

    # Save error table.
    tracking_errors_df.to_csv(output_csv_path, index=False)

    # Print outputs.
    print("Tracking error table:")
    print(tracking_errors_df)

    mean_translation_error = tracking_errors_df["translation_error"].mean() if not tracking_errors_df.empty else np.nan
    mean_rotation_error = tracking_errors_df["rotation_error"].mean() if not tracking_errors_df.empty else np.nan

    print(f"\nMean translation error: {mean_translation_error}")
    print(f"Mean rotation error: {mean_rotation_error}")
    print(f"\nTracking errors CSV saved to: {output_csv_path}")

    # Plot translation error.
    plt.figure(figsize=(8, 5))
    plt.plot(
        tracking_errors_df["frame_index"],
        tracking_errors_df["translation_error"],
        linewidth=1.8,
    )
    plt.xlabel("frame_index")
    plt.ylabel("translation_error")
    plt.title("translation_error over frames")
    plt.grid(True)
    plt.tight_layout()
    translation_plot_path = os.path.join(plots_dir, "translation_error_over_frames.png")
    plt.savefig(translation_plot_path, dpi=300)
    plt.close()

    # Plot rotation error.
    plt.figure(figsize=(8, 5))
    plt.plot(
        tracking_errors_df["frame_index"],
        tracking_errors_df["rotation_error"],
        linewidth=1.8,
    )
    plt.xlabel("frame_index")
    plt.ylabel("rotation_error")
    plt.title("rotation_error over frames")
    plt.grid(True)
    plt.tight_layout()
    rotation_plot_path = os.path.join(plots_dir, "rotation_error_over_frames.png")
    plt.savefig(rotation_plot_path, dpi=300)
    plt.close()

    # Plot predicted vs ground-truth centroid path for frames 41..50.
    centroid_comparison_plot_path = os.path.join(
        plots_dir, "predicted_vs_ground_truth_centroid.png"
    )
    plt.figure(figsize=(8, 6))
    plt.plot(
        tracking_errors_df["gt_x"],
        tracking_errors_df["gt_y"],
        marker="o",
        linewidth=1.8,
        label="Ground Truth (frames 41-50)",
    )
    plt.plot(
        tracking_errors_df["pred_x"],
        tracking_errors_df["pred_y"],
        marker="o",
        linewidth=1.8,
        label="Predicted (frames 41-50)",
    )
    plt.xlabel("centroid_x")
    plt.ylabel("centroid_y")
    plt.title("Predicted vs Ground Truth Centroid Tracking")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(centroid_comparison_plot_path, dpi=300)
    plt.close()

    print(f"Translation error plot saved to: {translation_plot_path}")
    print(f"Rotation error plot saved to: {rotation_plot_path}")
    print(f"Centroid comparison plot saved to: {centroid_comparison_plot_path}")


if __name__ == "__main__":
    main()
