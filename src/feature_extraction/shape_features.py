"""
Shape feature extraction for parachute segmentation masks. This module computes frame-wise geometric descriptors from binary masks.
Features include area, perimeter, centroid position, solidity, circularity,
non-compactness, eccentricity, and orientation, which are used later in
tracking and analysis.
"""

import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops


class ShapeFeatureExtractor:
    """
    Extract shape descriptors from binary segmentation masks.

    Each mask is expected to represent foreground parachute pixels with value 1
    and background with value 0. If multiple regions exist, the largest region
    is selected to represent the main parachute object.
    """

    def __init__(self, masks):

        self.masks = np.asarray(masks)
        if self.masks.ndim != 3:
            raise ValueError("masks must have shape (N, H, W).")
        self._features_df = None

    def _empty_feature_row(self, frame_index):

        #Create a safe default feature row for frames without foreground.

        # and we need it for Real segmentation outputs can occasionally miss the target object & Returning a complete row keeps frame indexing consistent.


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
        
        #Find the largest connected foreground region and its external contour.

        # Ensure mask is binary 0/1 uint8 for OpenCV operations.
        binary_mask = (mask > 0).astype(np.uint8)
        if np.count_nonzero(binary_mask) == 0:
            return None, None

        # Connected components separate multiple foreground regions.
        num_labels, labeled = cv2.connectedComponents(binary_mask)
        if num_labels <= 1:
            return None, None

        props = regionprops(labeled)
        if not props:
            return None, None

        # Use largest connected component as the parachute target region.
        largest_prop = max(props, key=lambda p: p.area)
        largest_region_mask = (labeled == largest_prop.label).astype(np.uint8)

        # Detect contour for perimeter and convex hull based shape measures.
        contours, _ = cv2.findContours(
            largest_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return largest_prop, None

        largest_contour = max(contours, key=cv2.contourArea)
        return largest_prop, largest_contour

    def extract_features(self):
        
        #Extract shape features for every frame mask.

        rows = []

        for frame_index, mask in enumerate(self.masks):
            region_prop, contour = self._largest_region_and_contour(mask)

            # If no valid foreground exists, fill with safe defaults.
            if region_prop is None:
                rows.append(self._empty_feature_row(frame_index))
                continue

            # Area from regionprops corresponds to foreground pixel count.
            area = float(region_prop.area)

            # Perimeter from contour length (OpenCV arcLength).
            perimeter = 0.0
            if contour is not None:
                perimeter = float(cv2.arcLength(contour, True))

            # regionprops centroid is returned as (row=y, col=x).
            centroid_y, centroid_x = region_prop.centroid

            # Solidity compares occupied area to convex hull area.
            solidity = 0.0
            if contour is not None:
                hull = cv2.convexHull(contour)
                hull_area = float(cv2.contourArea(hull))
                if hull_area > 0:
                    solidity = area / hull_area

            # Circularity measures how close the shape is to a circle.
            circularity = 0.0
            if perimeter > 0:
                circularity = (4.0 * np.pi * area) / (perimeter**2)

            # Non-compactness increases for more irregular or elongated shapes.
            non_compactness = 0.0
            if area > 0:
                non_compactness = (perimeter**2) / area

            # Eccentricity reflects elongation (0 ~ circle, closer to 1 elongated).
            eccentricity = float(region_prop.eccentricity)

            # Orientation is given in radians; convert to degrees for reporting.
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
        
        # This is to save extracted shape features as a CSV file.
 
        if self._features_df is None:
            self.extract_features()

        # Save CSV so later stages (tracking/evaluation) can reuse results.
        self._features_df.to_csv(output_path, index=False)
        return self._features_df
