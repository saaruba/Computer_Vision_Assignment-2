"""
Main script for Kalman-based parachute tracking evaluation.
In this PYthon script evaluates translation and rotation tracking using shape-feature
ground truth (centroid and orientation). and this code performs:
1. Main tracking run with fixed parameters.
2. Error table and summary metric export.
3. Plot generation for errors and prediction vs ground-truth trends.
4. Kalman parameter search over process/measurement noise values.
5. Overlay frame rendering and optional GIF export.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from tracking.kalman_rotation import RotationKalmanFilter
from tracking.kalman_translation import TranslationKalmanFilter
from utils.data_loader import DatasetLoader
from utils.metrics import rmse, rotation_error, summary_stats, translation_error
from utils.visualization import plot_line_over_frames, plot_two_lines_over_frames

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def run_tracking_for_params(
    gt_by_frame,
    process_noise,
    measurement_noise,
    dt=1.0,
    train_start=0,
    train_end=40,
    eval_start=41,
    eval_end=50,
):
    """
    Running this translation and rotation Kalman filters for one parameter pair.
        Training/tuning frames: predict + update with ground truth.
        Evaluation frames: predict only (no update), then compute errors.
        It simulates forecasting by stopping measurement updates during the
        evaluation window, so prediction quality can be measured fairly.
    """
    if train_start not in gt_by_frame.index:
        raise ValueError(f"Frame {train_start} is required for Kalman filter initialization.")

    frame0 = gt_by_frame.loc[train_start]

    # Instantiate both filters with the same noise settings for joint evaluation.
    translation_kf = TranslationKalmanFilter(
        dt=dt, process_noise=process_noise, measurement_noise=measurement_noise
    )
    rotation_kf = RotationKalmanFilter(
        dt=dt, process_noise=process_noise, measurement_noise=measurement_noise
    )

    translation_kf.initialize(frame0["centroid_x"], frame0["centroid_y"])
    rotation_kf.initialize(frame0["orientation_deg"])

    
    # Training/tuning stage

    for frame_idx in range(train_start + 1, train_end + 1):
        # Kalman prediction step projects state to the next frame.
        translation_kf.predict()
        rotation_kf.predict()

        if frame_idx in gt_by_frame.index:
            gt_row = gt_by_frame.loc[frame_idx]

            # Kalman update step corrects prediction with available measurement.
            translation_kf.update(gt_row["centroid_x"], gt_row["centroid_y"])
            rotation_kf.update(gt_row["orientation_deg"])


    # Prediction-only evaluation

    rows = []
    for frame_idx in range(eval_start, eval_end + 1):
        # Predict only: no measurement update in evaluation window.
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

        # Translation error: Euclidean distance between predicted and GT centroid.
        t_error = translation_error(pred_x, pred_y, gt_x, gt_y)

        # Rotation error: wrapped orientation difference in degree space.
        r_error = rotation_error(pred_theta, gt_theta)

        rows.append(
            {
                "frame_index": frame_idx,
                "pred_x": float(pred_x),
                "pred_y": float(pred_y),
                "gt_x": gt_x,
                "gt_y": gt_y,
                "translation_error": t_error,
                "pred_theta": float(pred_theta),
                "gt_theta": float(gt_theta),
                "rotation_error": r_error,
            }
        )

    return pd.DataFrame(
        rows,
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

"""
Run full tracking evaluation, plotting, parameter search, and overlays.
        - tracking_errors.csv
        - tracking_summary.csv
        - kalman_parameter_search.csv
        - tracking error and comparison plots
        - per-frame overlay images and optional GIF
"""

def main():

    images_path = "Data_Set/images"
    masks_path = "Data_Set/GT"

    input_csv_path = "results/tables/shape_features.csv"
    tables_dir = "results/tables"
    plots_dir = "results/plots"
    overlay_dir = os.path.join(plots_dir, "tracking_overlay")

    tracking_errors_path = os.path.join(tables_dir, "tracking_errors.csv")
    tracking_summary_path = os.path.join(tables_dir, "tracking_summary.csv")
    parameter_search_path = os.path.join(tables_dir, "kalman_parameter_search.csv")

    # Create output folders before any save operations.
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    
    # Load tracking ground truth
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)
    required_columns = ["frame_index", "centroid_x", "centroid_y", "orientation_deg"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in shape features CSV: {missing_columns}")

    # Drop rows with missing centroid/orientation to keep tracking robust.
    df = df[required_columns].copy()
    df = df.dropna(subset=["centroid_x", "centroid_y", "orientation_deg"]).copy()
    df["frame_index"] = df["frame_index"].astype(int)
    df = df[(df["frame_index"] >= 0) & (df["frame_index"] <= 50)].copy()
    df = df.sort_values("frame_index")
    df = df.drop_duplicates(subset="frame_index", keep="first")

    if df.empty:
        raise ValueError("No valid rows available after dropping NaNs and filtering frames 0..50.")

    gt_by_frame = df.set_index("frame_index")


    # Main tracking run
    default_process_noise = 0.1
    default_measurement_noise = 1.0
    print("Main tracking run uses process_noise=0.1 and measurement_noise=1.0")

    tracking_errors_df = run_tracking_for_params(
        gt_by_frame=gt_by_frame,
        process_noise=default_process_noise,
        measurement_noise=default_measurement_noise,
        dt=1.0,
        train_start=0,
        train_end=40,
        eval_start=41,
        eval_end=50,
    )

    # Save detailed per-frame tracking errors.
    tracking_errors_df.to_csv(tracking_errors_path, index=False)

    print("Tracking error table:")
    print(tracking_errors_df)
    print(f"\nTracking errors CSV saved to: {tracking_errors_path}")


    # Tracking summary metrics

    translation_values = tracking_errors_df["translation_error"].to_numpy(dtype=float)
    rotation_values = tracking_errors_df["rotation_error"].to_numpy(dtype=float)

    translation_stats = summary_stats(translation_values)
    rotation_stats = summary_stats(rotation_values)

    tracking_summary = {
        "mean_translation_error": translation_stats["mean"],
        "rmse_translation_error": rmse(translation_values),
        "max_translation_error": translation_stats["max"],
        "std_translation_error": translation_stats["std"],
        "mean_rotation_error": rotation_stats["mean"],
        "rmse_rotation_error": rmse(rotation_values),
        "max_rotation_error": rotation_stats["max"],
        "std_rotation_error": rotation_stats["std"],
    }

    tracking_summary_df = pd.DataFrame([tracking_summary])
    tracking_summary_df.to_csv(tracking_summary_path, index=False)

    print("\nTracking summary:")
    print(tracking_summary_df)
    print(f"\nTracking summary CSV saved to: {tracking_summary_path}")

    # Plot tracking errors
    translation_plot_path = os.path.join(plots_dir, "translation_error_over_frames.png")
    plot_line_over_frames(
        df=tracking_errors_df,
        x_col="frame_index",
        y_col="translation_error",
        title="translation_error over frames",
        ylabel="translation_error",
        output_path=translation_plot_path,
    )

    rotation_plot_path = os.path.join(plots_dir, "rotation_error_over_frames.png")
    plot_line_over_frames(
        df=tracking_errors_df,
        x_col="frame_index",
        y_col="rotation_error",
        title="rotation_error over frames",
        ylabel="rotation_error",
        output_path=rotation_plot_path,
    )

    # Plot predicted and ground-truth centroid paths in image coordinate space.
    centroid_comparison_plot_path = os.path.join(plots_dir, "predicted_vs_ground_truth_centroid.png")
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

    # Plot per-frame centroid_x predicted vs GT.
    pred_vs_gt_x_path = os.path.join(plots_dir, "pred_vs_gt_centroid_x_over_frames.png")
    plot_two_lines_over_frames(
        df=tracking_errors_df,
        x_col="frame_index",
        y1_col="gt_x",
        y2_col="pred_x",
        title="Predicted vs Ground Truth centroid_x over frames",
        y_label="centroid_x",
        label1="Ground Truth centroid_x",
        label2="Predicted centroid_x",
        output_path=pred_vs_gt_x_path,
    )

    # Plot per-frame centroid_y predicted vs GT.
    pred_vs_gt_y_path = os.path.join(plots_dir, "pred_vs_gt_centroid_y_over_frames.png")
    plot_two_lines_over_frames(
        df=tracking_errors_df,
        x_col="frame_index",
        y1_col="gt_y",
        y2_col="pred_y",
        title="Predicted vs Ground Truth centroid_y over frames",
        y_label="centroid_y",
        label1="Ground Truth centroid_y",
        label2="Predicted centroid_y",
        output_path=pred_vs_gt_y_path,
    )

    print(f"Translation error plot saved to: {translation_plot_path}")
    print(f"Rotation error plot saved to: {rotation_plot_path}")
    print(f"Centroid comparison plot saved to: {centroid_comparison_plot_path}")
    print(f"Pred vs GT centroid_x plot saved to: {pred_vs_gt_x_path}")
    print(f"Pred vs GT centroid_y plot saved to: {pred_vs_gt_y_path}")

    
    # Kalman parameter search
    process_noise_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    measurement_noise_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    search_rows = []

    for process_noise in process_noise_values:
        for measurement_noise in measurement_noise_values:
            # Re-run the same train/eval logic for each parameter pair.
            search_df = run_tracking_for_params(
                gt_by_frame=gt_by_frame,
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                dt=1.0,
                train_start=0,
                train_end=40,
                eval_start=41,
                eval_end=50,
            )

            search_translation = search_df["translation_error"].to_numpy(dtype=float)
            search_rotation = search_df["rotation_error"].to_numpy(dtype=float)
            search_translation_stats = summary_stats(search_translation)
            search_rotation_stats = summary_stats(search_rotation)

            search_rows.append(
                {
                    "process_noise": process_noise,
                    "measurement_noise": measurement_noise,
                    "mean_translation_error": search_translation_stats["mean"],
                    "rmse_translation_error": rmse(search_translation),
                    "max_translation_error": search_translation_stats["max"],
                    "mean_rotation_error": search_rotation_stats["mean"],
                    "rmse_rotation_error": rmse(search_rotation),
                    "max_rotation_error": search_rotation_stats["max"],
                }
            )

    parameter_search_df = pd.DataFrame(search_rows)
    parameter_search_df.to_csv(parameter_search_path, index=False)
    print(f"Kalman parameter search CSV saved to: {parameter_search_path}")

    # Select best combination using smallest mean translation error.
    valid_search_df = parameter_search_df.dropna(subset=["mean_translation_error"])
    if not valid_search_df.empty:
        best_idx = valid_search_df["mean_translation_error"].idxmin()
        best_row = valid_search_df.loc[best_idx]
        print(
            "Best parameters by mean_translation_error: "
            f"process_noise={best_row['process_noise']}, "
            f"measurement_noise={best_row['measurement_noise']}, "
            f"mean_translation_error={best_row['mean_translation_error']}"
        )
    else:
        print("Best parameters could not be determined due to missing evaluation rows.")


    # Tracking overlays on images
    # Reload image frames to render prediction-vs-ground-truth points.
    loader = DatasetLoader(images_path, masks_path)
    images, _ = loader.get_data()

    tracking_vis_df = pd.read_csv(tracking_errors_path)
    tracking_vis_df = tracking_vis_df[
        tracking_vis_df["frame_index"].between(41, 50, inclusive="both")
    ].copy()
    tracking_vis_df = tracking_vis_df.sort_values("frame_index")

    saved_overlay_paths = []
    for _, row in tracking_vis_df.iterrows():
        frame_idx = int(row["frame_index"])
        if frame_idx < 0 or frame_idx >= len(images):
            continue

        image_rgb = images[frame_idx]
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])
        pred_x = float(row["pred_x"])
        pred_y = float(row["pred_y"])

        # Draw GT and predicted centroids over the corresponding frame.
        plt.figure(figsize=(8, 6))
        plt.imshow(image_rgb)
        plt.scatter(gt_x, gt_y, s=50, c="blue", marker="o", label="Ground Truth Centroid")
        plt.scatter(pred_x, pred_y, s=50, c="orange", marker="o", label="Predicted Centroid")
        plt.title(f"Frame {frame_idx}")
        plt.legend()
        plt.axis("off")
        plt.tight_layout()

        overlay_path = os.path.join(overlay_dir, f"frame_{frame_idx:02d}.png")
        plt.savefig(overlay_path, dpi=300)
        plt.close()
        saved_overlay_paths.append(overlay_path)

    print(f"Tracking overlay frames saved to: {overlay_dir}")

    # Optional GIF to view tracking behavior as a short animation.
    if imageio is not None and saved_overlay_paths:
        gif_path = os.path.join(plots_dir, "tracking_overlay.gif")
        gif_frames = [imageio.imread(path) for path in saved_overlay_paths]
        imageio.mimsave(gif_path, gif_frames, duration=0.4)
        print(f"Tracking overlay GIF saved to: {gif_path}")


if __name__ == "__main__":
    main()
