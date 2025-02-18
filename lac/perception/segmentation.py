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


def centroid_matching(left_centroids, right_centroids, max_y_diff=5, max_x_diff=300):
    """
    Matches left centroids to right centroids based on the closest y-coordinate difference.
    Ensures that each right centroid is matched only once, optimizing globally.

    left_centroids : np.ndarray (N, 2) - Centroids from the left image
    right_centroids : np.ndarray (M, 2) - Centroids from the right image
    max_y_diff : int - Maximum allowed y-coordinate difference for a valid match
    max_x_diff : int - Maximum allowed x-coordinate difference for a valid match

    TODO: the max_y_diff should depend on roll of the camera
    TODO: the max_x_diff should depend on size of the mask and on y-value. Large rocks can have a
    large x_diff when close up, but small rocks should not have large x_diff when far away

    """
    matches = []

    # Compute all pairwise differences
    y_diffs = np.abs(left_centroids[:, None, 1] - right_centroids[None, :, 1])
    x_diffs = np.abs(left_centroids[:, None, 0] - right_centroids[None, :, 0])

    # Create a list of candidate matches (left_idx, right_idx, y_diff, x_diff)
    candidates = [
        (i, j, y_diffs[i, j], x_diffs[i, j])
        for i in range(len(left_centroids))
        for j in range(len(right_centroids))
    ]

    # Sort candidates by y-coordinate difference
    candidates.sort(key=lambda x: x[2])

    used_left = set()
    used_right = set()

    for left_idx, right_idx, y_diff, x_diff in candidates:
        if (
            y_diff < max_y_diff
            and x_diff < max_x_diff
            and left_idx not in used_left
            and right_idx not in used_right
        ):
            matches.append((left_centroids[left_idx], right_centroids[right_idx]))
            used_left.add(left_idx)
            used_right.add(right_idx)

    return matches
