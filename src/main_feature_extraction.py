import os

import matplotlib.pyplot as plt

from feature_extraction.shape_features import ShapeFeatureExtractor
from feature_extraction.hog_features import HOGFeatureExtractor
from utils.data_loader import DatasetLoader


def main():
    """Load dataset, extract shape features, save outputs, and optionally preview sample."""
    images_path = "Data_Set/images"
    masks_path = "Data_Set/GT"

    tables_dir = "results/tables"
    plots_dir = "results/plots"
    output_csv_path = os.path.join(tables_dir, "shape_features.csv")

    try:
        # Ensure output directories exist.
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Load images and masks.
        loader = DatasetLoader(images_path, masks_path)
        images, masks = loader.get_data()

        # Basic dataset checks and prints.
        print(f"Number of images loaded: {len(images)}")
        print(f"Number of masks loaded: {len(masks)}")

        if len(images) == 0 or len(masks) == 0:
            print("Dataset loaded but no images or masks were found.")
            return

        print(f"Shape of first image: {images[0].shape}")
        print(f"Shape of first mask: {masks[0].shape}")

        # Extract shape features and save CSV.
        feature_extractor = ShapeFeatureExtractor(masks)
        shape_features_df = feature_extractor.extract_features()
        feature_extractor.save_features(output_csv_path)

        print("\nFirst 5 rows of shape features:")
        print(shape_features_df.head())
        print(f"\nShape features CSV saved to: {output_csv_path}")

        # Required feature plots over frames.
        features_to_plot = [
            "solidity",
            "non_compactness",
            "circularity",
            "eccentricity",
        ]

        for feature_name in features_to_plot:
            plt.figure(figsize=(8, 5))
            plt.plot(
                shape_features_df["frame_index"],
                shape_features_df[feature_name],
                linewidth=1.8,
            )
            plt.xlabel("frame_index")
            plt.ylabel(feature_name)
            plt.title(f"{feature_name} over frames")
            plt.grid(True)
            plt.tight_layout()

            output_plot_path = os.path.join(plots_dir, f"{feature_name}_over_frames.png")
            plt.savefig(output_plot_path, dpi=300)
            plt.close()

        print(f"Feature plots saved to: {plots_dir}")

        # Extract HoG texture features and save CSV.
        hog_output_csv_path = os.path.join(tables_dir, "hog_features.csv")
        hog_extractor = HOGFeatureExtractor(images, masks)
        hog_features_df = hog_extractor.extract_features()
        hog_extractor.save_features(hog_output_csv_path)

        print("\nFirst 5 rows of HoG features:")
        print(hog_features_df.head())
        print(f"\nHoG features CSV saved to: {hog_output_csv_path}")

        # Required HoG plots over frames.
        hog_features_to_plot = [
            "hog_0_deg",
            "hog_45_deg",
            "hog_90_deg",
            "hog_135_deg",
        ]

        for feature_name in hog_features_to_plot:
            plt.figure(figsize=(8, 5))
            plt.plot(
                hog_features_df["frame_index"],
                hog_features_df[feature_name],
                linewidth=1.8,
            )
            plt.xlabel("frame_index")
            plt.ylabel("HoG value")
            plt.title(f"{feature_name} over frames")
            plt.grid(True)
            plt.tight_layout()

            output_plot_path = os.path.join(plots_dir, f"{feature_name}_over_frames.png")
            plt.savefig(output_plot_path, dpi=300)
            plt.close()

        print(f"HoG plots saved to: {plots_dir}")

        # Save a sample image/mask figure instead of displaying interactively.
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
        print(f"Failed to run feature extraction pipeline: {exc}")


if __name__ == "__main__":
    main()

