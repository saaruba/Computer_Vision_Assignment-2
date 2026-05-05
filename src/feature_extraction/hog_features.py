import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog


class HOGFeatureExtractor:
    """
    Extract HoG orientation-bin summary features from RGB image frames.

    For each frame, the extractor computes mean HoG responses for 4 bins:
    0, 45, 90, and 135 degrees.
    """

    def __init__(self, images, masks=None):
        """
        Initialize HoG feature extractor.

        Args:
            images (np.ndarray or list): RGB frames with shape (N, H, W, 3).
            masks (np.ndarray or list, optional): Binary masks with shape (N, H, W).
                Foreground values can be 1/255/True.
        """
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

        # HoG settings requested in the assignment.
        self.orientations = 4
        self.pixels_per_cell = (8, 8)
        self.cells_per_block = (1, 1)

    def _mask_and_crop_region(self, gray_image, mask):
        """
        Apply mask and crop around foreground region with small padding.

        If mask is empty, returns the full grayscale image.

        Args:
            gray_image (np.ndarray): Grayscale image.
            mask (np.ndarray): 2D mask for the frame.

        Returns:
            np.ndarray: Cropped grayscale region.
        """
        if mask is None:
            return gray_image

        binary_mask = (mask > 0).astype(np.uint8)
        if np.count_nonzero(binary_mask) == 0:
            return gray_image

        masked = gray_image.copy()
        masked[binary_mask == 0] = 0

        ys, xs = np.where(binary_mask > 0)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

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
        """
        Ensure the image is large enough for HoG cell configuration.

        Args:
            image_2d (np.ndarray): Grayscale image region.

        Returns:
            np.ndarray: Resized/grayscale-safe image region.
        """
        min_h = self.pixels_per_cell[0]
        min_w = self.pixels_per_cell[1]

        h, w = image_2d.shape[:2]
        target_h = max(h, min_h)
        target_w = max(w, min_w)

        if h != target_h or w != target_w:
            image_2d = cv2.resize(image_2d, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return image_2d

    def _compute_orientation_means(self, region):
        """
        Compute mean HoG response for each of the 4 orientation bins.

        Args:
            region (np.ndarray): Grayscale region for HoG extraction.

        Returns:
            tuple: (hog_0, hog_45, hog_90, hog_135)
        """
        if region.ndim != 2:
            return 0.0, 0.0, 0.0, 0.0

        safe_region = self._ensure_minimum_size(region)

        try:
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

        # Collapse all spatial/block dimensions; keep orientation bins.
        if hog_array.ndim < 1 or hog_array.shape[-1] != self.orientations:
            return 0.0, 0.0, 0.0, 0.0

        reshaped = hog_array.reshape(-1, self.orientations)
        means = reshaped.mean(axis=0)

        # Replace invalid values safely.
        means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        return float(means[0]), float(means[1]), float(means[2]), float(means[3])

    def extract_features(self):
        """
        Extract HoG orientation-bin summary features for all frames.

        Returns:
            pd.DataFrame: One row per frame with HoG features.
        """
        rows = []

        for frame_index, image_rgb in enumerate(self.images):
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            frame_mask = None if self.masks is None else self.masks[frame_index]
            region = self._mask_and_crop_region(gray, frame_mask)

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
        """
        Save extracted HoG features to CSV.

        Args:
            output_path (str): Destination CSV path.

        Returns:
            pd.DataFrame: Saved feature DataFrame.
        """
        if self._features_df is None:
            self.extract_features()

        self._features_df.to_csv(output_path, index=False)
        return self._features_df
