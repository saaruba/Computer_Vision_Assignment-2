import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops


class ShapeFeatureExtractor:
    """
    Extract shape-based features from binary segmentation masks.

    Expected mask values are 0 and 1, with one mask per frame.
    """

    def __init__(self, masks):
        """
        Initialize the extractor.

        Args:
            masks (np.ndarray or list): Binary masks with shape (N, H, W),
                where foreground is 1 and background is 0.
        """
        self.masks = np.asarray(masks)
        if self.masks.ndim != 3:
            raise ValueError("masks must have shape (N, H, W).")
        self._features_df = None

    def _empty_feature_row(self, frame_index):
        """
        Build a safe default row when no foreground region exists.

        Args:
            frame_index (int): Index of current frame.

        Returns:
            dict: Feature row with zeros/NaNs.
        """
        return {
            "frame_index": frame_index,
            "area": 0.0,
            "perimeter": 0.0,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "solidity": 0.0,
            "non_compactness": 0.0,
            "circularity": 0.0,
            "eccentricity": np.nan,
            "orientation_deg": np.nan,
        }

    def _largest_region_and_contour(self, mask):
        """
        Select the largest connected foreground region and its contour.

        Args:
            mask (np.ndarray): 2D binary mask (0/1).

        Returns:
            tuple: (region_prop, contour) for the largest region,
                or (None, None) if no region exists.
        """
        binary_mask = (mask > 0).astype(np.uint8)
        if np.count_nonzero(binary_mask) == 0:
            return None, None

        num_labels, labeled = cv2.connectedComponents(binary_mask)
        if num_labels <= 1:
            return None, None

        props = regionprops(labeled)
        if not props:
            return None, None

        largest_prop = max(props, key=lambda p: p.area)
        largest_region_mask = (labeled == largest_prop.label).astype(np.uint8)

        contours, _ = cv2.findContours(
            largest_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return largest_prop, None

        largest_contour = max(contours, key=cv2.contourArea)
        return largest_prop, largest_contour

    def extract_features(self):
        """
        Extract one shape-feature row per frame.

        Returns:
            pd.DataFrame: Feature table with one row per mask/frame.
        """
        rows = []

        for frame_index, mask in enumerate(self.masks):
            region_prop, contour = self._largest_region_and_contour(mask)

            if region_prop is None:
                rows.append(self._empty_feature_row(frame_index))
                continue

            area = float(region_prop.area)

            perimeter = 0.0
            if contour is not None:
                perimeter = float(cv2.arcLength(contour, True))

            centroid_y, centroid_x = region_prop.centroid

            solidity = 0.0
            if contour is not None:
                hull = cv2.convexHull(contour)
                hull_area = float(cv2.contourArea(hull))
                if hull_area > 0:
                    solidity = area / hull_area

            circularity = 0.0
            if perimeter > 0:
                circularity = (4.0 * np.pi * area) / (perimeter ** 2)

            non_compactness = 0.0
            if area > 0:
                non_compactness = (perimeter ** 2) / area

            eccentricity = float(region_prop.eccentricity)
            orientation_deg = float(np.degrees(region_prop.orientation))

            rows.append(
                {
                    "frame_index": frame_index,
                    "area": area,
                    "perimeter": perimeter,
                    "centroid_x": float(centroid_x),
                    "centroid_y": float(centroid_y),
                    "solidity": solidity,
                    "non_compactness": non_compactness,
                    "circularity": circularity,
                    "eccentricity": eccentricity,
                    "orientation_deg": orientation_deg,
                }
            )

        self._features_df = pd.DataFrame(rows)
        return self._features_df

    def save_features(self, output_path):
        """
        Save extracted features to CSV.

        Args:
            output_path (str): Output CSV path.

        Returns:
            pd.DataFrame: Saved features DataFrame.
        """
        if self._features_df is None:
            self.extract_features()

        self._features_df.to_csv(output_path, index=False)
        return self._features_df
