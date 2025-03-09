"""Basic visual odometry based on pyslam"""

import numpy as np
import cv2 as cv

from lac.params import IMG_FOV_RAD


class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cx = width / 2
        self.cy = height / 2
        self.fx = width / (2 * np.tan(IMG_FOV_RAD / 2))
        self.fy = self.fx


class VisualOdometry:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.pnp = cv.solvePnP
