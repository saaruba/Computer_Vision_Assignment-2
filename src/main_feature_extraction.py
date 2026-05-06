"""
In this Main script for feature extraction in the CMP9135M parachute project.

This script runs the complete feature pipeline:
1. Load RGB images and binary masks.
2. Extract shape features from masks.
3. Extract HoG texture features from images (optionally mask-focused).
4. Save feature tables and analysis plots.
5. Build a compact summary table for reporting.

All outputs are written to the `results/` directory structure.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from feature_extraction.hog_features import HOGFeatureExtractor
from feature_extraction.shape_features import ShapeFeatureExtractor
from utils.data_loader import DatasetLoader
from utils.metrics import summary_stats
from utils.visualization import plot_line_over_frames, plot_normalized_features

"""
here we Execute feature extraction and reporting for the assignment dataset.

Outputs generated:
        - shape_features.csv
        - hog_features.csv
        - feature_summary.csv
        - per-feature plots over frames
        - normalized shape-feature comparison plot
        - sample image/mask figure
"""

def main():

    images_path = "Data_Set/images"
    masks_path = "Data_Set/GT"

    tables_dir = "results/tables"
    plots_dir = "results/plots"
    shape_output_csv_path = os.path.join(tables_dir, "shape_features.csv")
    hog_output_csv_path = os.path.join(tables_dir, "hog_features.csv")
    feature_summary_path = os.path.join(tables_dir, "feature_summary.csv")

    shape_features_to_plot = [
        "solidity",
        "non_compactness",
        "circularity",
        "eccentricity",
    ]
    hog_features_to_plot = [
        "hog_0_deg",
        "hog_45_deg",
        "hog_90_deg",
        "hog_135_deg",
    ]

    try:
        # Create output folders first so all save operations succeed.
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Load dataset once; this includes RGB conversion and binary mask conversion.
        loader = DatasetLoader(images_path, masks_path)
        images, masks = loader.get_data()

        print(f"Number of images loaded: {len(images)}")
        print(f"Number of masks loaded: {len(masks)}")

        if len(images) == 0 or len(masks) == 0:
            print("Dataset loaded but no images or masks were found.")
            return

        print(f"Shape of first image: {images[0].shape}")
        print(f"Shape of first mask: {masks[0].shape}")

        # Shape feature extraction

        shape_extractor = ShapeFeatureExtractor(masks)
        shape_features_df = shape_extractor.extract_features()

        # Save CSV so tracking/evaluation scripts can reuse these features.
        shape_extractor.save_features(shape_output_csv_path)

        print("\nFirst 5 rows of shape features:")
        print(shape_features_df.head())
        print(f"\nShape features CSV saved to: {shape_output_csv_path}")

        # Plot each required shape feature against frame index.
        for feature_name in shape_features_to_plot:
            output_plot_path = os.path.join(plots_dir, f"{feature_name}_over_frames.png")
            plot_line_over_frames(
                df=shape_features_df,
                x_col="frame_index",
                y_col=feature_name,
                title=f"{feature_name} over frames",
                ylabel=feature_name,
                output_path=output_plot_path,
            )

        print(f"Shape feature plots saved to: {plots_dir}")

    
        # HoG feature extraction
    
        hog_extractor = HOGFeatureExtractor(images, masks)
        hog_features_df = hog_extractor.extract_features()

        # Save HoG table for downstream analysis.
        hog_extractor.save_features(hog_output_csv_path)

        print("\nFirst 5 rows of HoG features:")
        print(hog_features_df.head())
        print(f"\nHoG features CSV saved to: {hog_output_csv_path}")

        # Plot each required HoG orientation-bin feature over time.
        for feature_name in hog_features_to_plot:
            output_plot_path = os.path.join(plots_dir, f"{feature_name}_over_frames.png")
            plot_line_over_frames(
                df=hog_features_df,
                x_col="frame_index",
                y_col=feature_name,
                title=f"{feature_name} over frames",
                ylabel="HoG value",
                output_path=output_plot_path,
            )

        print(f"HoG plots saved to: {plots_dir}")

    
        # Combined summary statistics
    
        combined_df = pd.merge(
            shape_features_df[["frame_index"] + shape_features_to_plot],
            hog_features_df[["frame_index"] + hog_features_to_plot],
            on="frame_index",
            how="inner",
        )

        summary_rows = []
        combined_features = shape_features_to_plot + hog_features_to_plot
        for feature_name in combined_features:
            # Use shared metric helper for consistent mean/std/min/max reporting.
            stats = summary_stats(combined_df[feature_name])
            summary_rows.append(
                {
                    "feature": feature_name,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )

        feature_summary_df = pd.DataFrame(summary_rows, columns=["feature", "mean", "std", "min", "max"])
        feature_summary_df.to_csv(feature_summary_path, index=False)

        print("\nFeature summary table:")
        print(feature_summary_df)
        print(f"\nFeature summary CSV saved to: {feature_summary_path}")

        # Plot normalized shape features together to compare trends fairly.
        normalized_plot_path = os.path.join(plots_dir, "shape_features_normalized_comparison.png")
        plot_normalized_features(
            df=shape_features_df,
            frame_col="frame_index",
            feature_cols=shape_features_to_plot,
            title="Normalized Shape Features Comparison",
            output_path=normalized_plot_path,
        )
        print(f"Normalized shape feature comparison plot saved to: {normalized_plot_path}")

        # Save a static sample preview figure (no plt.show to avoid terminal blocking).
        show_sample = False
        if not show_sample:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(images[0])
            axes[0].set_title("Sample Image")
            axes[0].axis("off")

            axes[1].imshow(masks[0], cmap="gray")
            axes[1].set_title("Sample Mask")
            axes[1].axis("off")

            plt.tight_layout()
            sample_plot_path = os.path.join(plots_dir, "sample_image_and_mask.png")
            plt.savefig(sample_plot_path, dpi=300)
            plt.close()
            print(f"Sample image/mask figure saved to: {sample_plot_path}")

    except Exception as exc:
        # Catch-and-report keeps failures clear during assignment testing.
        print(f"Failed to run feature extraction pipeline: {exc}")


if __name__ == "__main__":
    main()
