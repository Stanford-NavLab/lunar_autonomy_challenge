"""Computer vision utilities"""

import cv2 as cv
import numpy as np
import apriltag


def project_pixel_to_3D(pixel, depth, K):
    """
    Project a pixel to 3D using the depth map and camera intrinsics

    pixel : tuple - (x, y) pixel coordinates
    depth : float - depth value at the pixel
    K : np.ndarray (3, 3) - Camera intrinsics matrix

    Returns:
    np.ndarray (3,) - 3D point in camera frame
    """
    x, y = pixel
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])


def detect_fiducials(image, detector):
    pass
