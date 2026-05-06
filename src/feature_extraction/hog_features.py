"""
HoG texture feature extraction for parachute image frames.This module computes Histogram of Oriented Gradients (HoG) descriptors on each
frame and summarizes them into four orientation bins (0 deg, 45 deg, 90 deg,135 deg). Optional segmentation masks focus texture analysis on the parachute region.
"""

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog


class HOGFeatureExtractor:
    """
    Here we Extract framelevel HoG orientation responses from RGB images.The extractor computes HoG with 4 orientations and reports the mean
    response for each orientation bin as compact texture features.
    """

    def __init__(self, images, masks=None):
        
        #Initialize the HoG extractor with image frames and optional masks.

        self.images = np.asarray(images)
        self.masks = None if masks is None else np.asarray(masks)
        self._features_df = None

        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError("images must have shape (N, H, W, 3).")

        if self.masks is not None:
            if self.masks.ndim != 3:
                raise ValueError("masks must have shape (N, H, W).")
            if len(self.masks) != len(self.images):
                raise ValueError("images and masks must contain the same number of frames.")

        # HoG settings fixed by assignment requirements.
        self.orientations = 4  # Bins map to 0 deg, 45 deg, 90 deg, and 135 deg.
        self.pixels_per_cell = (8, 8)
        self.cells_per_block = (1, 1)

    def _mask_and_crop_region(self, gray_image, mask):
        
        #Applying mask to grayscale image and crop around foreground region.

        if mask is None:
            return gray_image

        # Convert any non-zero mask values to 1 for robust foreground detection.
        binary_mask = (mask > 0).astype(np.uint8)
        if np.count_nonzero(binary_mask) == 0:
            # Empty mask fallback: use full image to avoid losing the frame.
            return gray_image

        # Zero-out background so gradients mainly come from the target object.
        masked = gray_image.copy()
        masked[binary_mask == 0] = 0

        # Compute tight bounding box around foreground pixels.
        ys, xs = np.where(binary_mask > 0)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        # Small padding keeps some local context around object boundaries.
        padding = 5
        y_min = max(0, y_min - padding)
        y_max = min(gray_image.shape[0] - 1, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(gray_image.shape[1] - 1, x_max + padding)

        cropped = masked[y_min : y_max + 1, x_min : x_max + 1]
        if cropped.size == 0:
            return gray_image

        return cropped

    def _ensure_minimum_size(self, image_2d):
        
        #Ensure image region is large enough for HoG cell configuration.

        min_h = self.pixels_per_cell[0]
        min_w = self.pixels_per_cell[1]

        h, w = image_2d.shape[:2]
        target_h = max(h, min_h)
        target_w = max(w, min_w)

        if h != target_h or w != target_w:
            image_2d = cv2.resize(image_2d, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return image_2d

    def _compute_orientation_means(self, region):

        if region.ndim != 2:
            return 0.0, 0.0, 0.0, 0.0

        safe_region = self._ensure_minimum_size(region)

        try:
            # HoG captures local gradient orientation distribution (texture/edge structure).
            hog_map = hog(
                safe_region,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=False,
                feature_vector=False,
            )
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

        hog_array = np.asarray(hog_map, dtype=np.float64)
        if hog_array.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Last dimension stores orientation bins; other dimensions are spatial/block axes.
        if hog_array.ndim < 1 or hog_array.shape[-1] != self.orientations:
            return 0.0, 0.0, 0.0, 0.0

        # Flatten spatial structure and summarize each orientation by mean response.
        reshaped = hog_array.reshape(-1, self.orientations)
        means = reshaped.mean(axis=0)

        # Replace invalid values to keep CSV outputs stable.
        means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        return float(means[0]), float(means[1]), float(means[2]), float(means[3])

    def extract_features(self):
        """
        Extract HoG orientation-bin features for every frame.

        This Pipeline Can
            1. Convert RGB image to grayscale.
            2. Optionally apply mask and crop around foreground.
            3. Compute HoG orientation-bin means.
            4. Save one row with frame index + 4 HoG features.

        """
        rows = []

        for frame_index, image_rgb in enumerate(self.images):
            # Convert RGB to grayscale before gradient-based HoG extraction.
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            frame_mask = None if self.masks is None else self.masks[frame_index]
            region = self._mask_and_crop_region(gray, frame_mask)

            # Extract mean responses for 0 deg, 45 deg, 90 deg, and 135 deg bins.
            hog_0, hog_45, hog_90, hog_135 = self._compute_orientation_means(region)

            rows.append(
                {
                    "frame_index": frame_index,
                    "hog_0_deg": hog_0,
                    "hog_45_deg": hog_45,
                    "hog_90_deg": hog_90,
                    "hog_135_deg": hog_135,
                }
            )

        self._features_df = pd.DataFrame(rows)
        return self._features_df

    def save_features(self, output_path):
        
        #Save extracted HoG features to a CSV file.

        if self._features_df is None:
            self.extract_features()

        # Export HoG table so downstream scripts can use consistent inputs.
        self._features_df.to_csv(output_path, index=False)
        return self._features_df
