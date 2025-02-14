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


def overlay_mask(image_gray, mask, color=(0, 0, 1)):
    """
    image_gray : np.ndarray (H, W) - grayscale image
    mask : np.ndarray (H, W) - Binary mask
    color : tuple (3) - BGR color
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
