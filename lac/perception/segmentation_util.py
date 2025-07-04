"""Utils for segmentation"""

import numpy as np
import cv2

from lac.util import mask_centroid


def dilate_mask(mask, pixels=1):
    """
    Dilate binary mask using a square kernel.
    """
    size = 2 * pixels + 1
    kernel = np.ones((size, size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask


def get_mask_centroids(masks, sorted=False):
    """
    seg_results : dict - Results from the segmentation model
    """
    mask_centroids = []
    for mask in masks:
        centroid = mask_centroid(mask.astype(np.uint8))
        mask_centroids.append(centroid)
    mask_centroids = np.array(mask_centroids)
    # Sort by y-coordinate
    if len(mask_centroids) > 1 and sorted:
        mask_centroids = mask_centroids[np.argsort(mask_centroids[:, 1])]
    return mask_centroids


def centroid_matching(left_centroids, right_centroids, max_y_diff=25, max_x_diff=300):
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

    assert left_centroids.shape[1] == 2, "Left centroids should have shape (N, 2)"
    assert right_centroids.shape[1] == 2, "Right centroids should have shape (M, 2)"

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
            # matches.append((left_centroids[left_idx], right_centroids[right_idx]))
            matches.append((left_idx, right_idx))
            used_left.add(left_idx)
            used_right.add(right_idx)

    return matches
