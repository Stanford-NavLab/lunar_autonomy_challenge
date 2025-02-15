"Semantic and instance segmentation"

import numpy as np
from PIL import Image
from lang_sam import LangSAM
import torch
import cv2 as cv

from lac.util import mask_centroid, color_mask


class Segmentation:
    def __init__(self):
        self.model = LangSAM()

    def segment_rocks(self, image: Image):
        """
        image : RGB PIL Image
        """
        text_prompt = "rock."
        results = self.model.predict([image], [text_prompt])

        full_mask = np.zeros_like(image, dtype=np.uint8).copy()
        for mask in results[0]["masks"]:
            full_mask[mask.astype(bool)] = 255

        return results[0], full_mask

    def stereo_segment_and_depth(self, left_image: Image, right_image: Image):
        """
        left_image : RGB PIL Image
        right_image : RGB PIL Image
        """
        left_results, left_mask = self.segment_rocks(left_image)
        right_results, right_mask = self.segment_rocks(right_image)

        return left_results, right_results, left_mask, right_mask


def overlay_mask(image_gray, mask, color=(1, 0, 0)):
    """
    image_gray : np.ndarray (H, W) - grayscale image
    mask : np.ndarray (H, W) - Binary mask
    color : tuple (3) - RGB color
    """
    image_rgb = cv.cvtColor(image_gray, cv.COLOR_GRAY2BGR)
    mask_colored = color_mask(mask, color).astype(image_rgb.dtype)
    return cv.addWeighted(image_rgb, 1.0, mask_colored, beta=0.5, gamma=0)


def get_mask_centroids(seg_results):
    """
    seg_results : dict - Results from the segmentation model
    """
    mask_centroids = []
    for mask in seg_results["masks"]:
        mask = mask.astype(np.uint8)
        mask_centroids.append(mask_centroid(mask))
    mask_centroids = np.array(mask_centroids)
    # Sort by y-coordinate
    mask_centroids = mask_centroids[np.argsort(mask_centroids[:, 1])]
    return mask_centroids


# def centroid_matching(left_centroids, right_centroids):
#     """
#     Matches each left centroid to the closest right centroid based on the y-coordinate difference.
#     Ensures that each right centroid is matched only once.

#     left_centroids : np.ndarray (N, 2) - Centroids from the left image
#     right_centroids : np.ndarray (M, 2) - Centroids from the right image
#     """
#     matched_pairs = []
#     used_right_indices = set()

#     for left_centroid in left_centroids:
#         # Compute absolute y-coordinate differences
#         y_diff = np.abs(right_centroids[:, 1] - left_centroid[1])

#         # Mask out already matched right centroids
#         valid_indices = [i for i in range(len(right_centroids)) if i not in used_right_indices]
#         if not valid_indices:
#             continue  # No available centroids to match

#         # Find the closest right centroid among available ones
#         closest_idx = min(valid_indices, key=lambda i: y_diff[i])

#         # If the y-coordinate difference is less than 5 pixels, consider it a match
#         if y_diff[closest_idx] < 5:
#             matched_pairs.append((left_centroid, right_centroids[closest_idx]))
#             used_right_indices.add(closest_idx)  # Mark as used

#     return matched_pairs


def centroid_matching(left_centroids, right_centroids):
    """
    Matches left centroids to right centroids based on the closest y-coordinate difference.
    Ensures that each right centroid is matched only once, optimizing globally.

    left_centroids : np.ndarray (N, 2) - Centroids from the left image
    right_centroids : np.ndarray (M, 2) - Centroids from the right image
    """
    matches = []

    # Compute all pairwise y-coordinate differences
    y_diffs = np.abs(left_centroids[:, None, 1] - right_centroids[None, :, 1])

    # Create a list of candidate matches (left_idx, right_idx, y_diff)
    candidates = [
        (i, j, y_diffs[i, j])
        for i in range(len(left_centroids))
        for j in range(len(right_centroids))
    ]

    # Sort candidates by y-coordinate difference
    candidates.sort(key=lambda x: x[2])

    used_left = set()
    used_right = set()

    for left_idx, right_idx, diff in candidates:
        if diff < 5 and left_idx not in used_left and right_idx not in used_right:
            matches.append((left_centroids[left_idx], right_centroids[right_idx]))
            used_left.add(left_idx)
            used_right.add(right_idx)

    return matches
