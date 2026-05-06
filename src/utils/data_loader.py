"""
Dataset loading utilities for the CMP9135M computer vision pipeline. And in this module provides a class that loads RGB image frames and their matching
segmentation masks from disk. It ensures robust file alignment by filename
stem, applies natural sorting for predictable frame order, converts image
channels from OpenCV BGR to RGB, and converts grayscale mask images to
binary masks (0/1) for downstream feature extraction and tracking tasks.
"""

import os

import cv2
import numpy as np


class DatasetLoader:
    """
    Load aligned RGB frames and binary segmentation masks from two folders.
    The class is designed for paired datasets where image and mask filenames
    share the same stem (for example, `frame_01.png` in both directories).
    """

    def __init__(self, images_folder, masks_folder):
    
        #Store dataset folder paths and initialize internal file lists.

        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.image_paths = []
        self.mask_paths = []

        # Supported file extensions keep loading flexible across datasets.
        self.supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def _validate_directories(self):
    
        #Validate that both input directories exist before reading any files.

        if not os.path.isdir(self.images_folder):
            raise FileNotFoundError(f"Images folder not found: {self.images_folder}")
        if not os.path.isdir(self.masks_folder):
            raise FileNotFoundError(f"Masks folder not found: {self.masks_folder}")

    def _natural_sort_key(self, text):
        
        #Build a natural sorting key so numeric filename parts sort correctly.

        key = []
        current = ""
        is_digit = None

        for char in text:
            char_is_digit = char.isdigit()
            if is_digit is None:
                current = char
                is_digit = char_is_digit
                continue
            if char_is_digit == is_digit:
                current += char
            else:
                key.append(int(current) if is_digit else current.lower())
                current = char
                is_digit = char_is_digit

        if current:
            key.append(int(current) if is_digit else current.lower())

        return key

    def _list_supported_files(self, folder_path):
        
        #List files in a folder that match supported image/mask extensions.

        files = []
        for name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, name)
            if os.path.isfile(full_path) and name.lower().endswith(self.supported_extensions):
                files.append(name)
        return files

    def load_file_paths(self):
        
        #Load and align image/mask file paths by filename stem.

        self._validate_directories()

        image_files = self._list_supported_files(self.images_folder)
        mask_files = self._list_supported_files(self.masks_folder)

        # Count check helps detect incomplete datasets early.
        if len(image_files) != len(mask_files):
            raise ValueError(
                f"Image and mask counts do not match: "
                f"{len(image_files)} images vs {len(mask_files)} masks."
            )

        image_map = {}
        for filename in image_files:
            stem = os.path.splitext(filename)[0]
            if stem in image_map:
                raise ValueError(f"Duplicate image filename stem found: {stem}")
            image_map[stem] = filename

        mask_map = {}
        for filename in mask_files:
            stem = os.path.splitext(filename)[0]
            if stem in mask_map:
                raise ValueError(f"Duplicate mask filename stem found: {stem}")
            mask_map[stem] = filename

        image_stems = set(image_map.keys())
        mask_stems = set(mask_map.keys())

        # Explicit mismatch reporting makes debugging dataset issues easier.
        missing_masks = sorted(image_stems - mask_stems, key=self._natural_sort_key)
        missing_images = sorted(mask_stems - image_stems, key=self._natural_sort_key)

        if missing_masks or missing_images:
            message_parts = ["Image/mask filename mismatch detected."]
            if missing_masks:
                message_parts.append(f"Missing masks for: {missing_masks}")
            if missing_images:
                message_parts.append(f"Missing images for: {missing_images}")
            raise ValueError(" ".join(message_parts))

        common_stems = sorted(image_stems, key=self._natural_sort_key)
        self.image_paths = [os.path.join(self.images_folder, image_map[stem]) for stem in common_stems]
        self.mask_paths = [os.path.join(self.masks_folder, mask_map[stem]) for stem in common_stems]

        return self.image_paths, self.mask_paths

    def load_images(self):
        
        # Load aligned color images and convert them from BGR to RGB.

    
        if not self.image_paths or not self.mask_paths:
            self.load_file_paths()

        images = []
        for image_path in self.image_paths:
            # Read color image from disk.
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image file: {image_path}")

            # Convert BGR to RGB so colors are correct in matplotlib/pipeline.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image_rgb)

        images_array = np.array(images)

        # Debug output helps confirm dataset size and image dimensionality.
        print(f"[DatasetLoader] Number of images loaded: {len(images_array)}")
        if len(images_array) > 0:
            print(f"[DatasetLoader] Sample image shape: {images_array[0].shape}")

        return images_array

    def load_masks(self):
        
        # Load aligned mask images in grayscale and binarize them to 0/1 values.

        if not self.image_paths or not self.mask_paths:
            self.load_file_paths()

        masks = []
        for mask_path in self.mask_paths:
            # Read mask as grayscale to get one channel per frame.
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask file: {mask_path}")

            # Convert grayscale mask to binary {0, 1} using a mid-level threshold.
            _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            masks.append(binary_mask.astype(np.uint8))

        masks_array = np.array(masks)

        # Debug output helps validate loading and mask dimensions quickly.
        print(f"[DatasetLoader] Number of masks loaded: {len(masks_array)}")
        if len(masks_array) > 0:
            print(f"[DatasetLoader] Sample mask shape: {masks_array[0].shape}")

        return masks_array

    def get_data(self):
        """
        Load and return aligned image and mask arrays in one call.
        """
        self.load_file_paths()
        images = self.load_images()
        masks = self.load_masks()
        return images, masks
