"""Basic visual odometry based on pyslam"""

import numpy as np
import cv2

from lightglue import LightGlue, SuperPoint


class StereoVisualOdometry:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.pnp = cv2.solvePnP

        self.extractor = None
        self.matcher = None

    def initialize(self):
        # Initialize from first frame
        pass

    def track(self, left_image, right_image, left_kps, right_kps):
        """
        left_image : np.ndarray - Left image
        right_image : np.ndarray - Right image
        left_kps : np.ndarray - Keypoints in the left image
        right_kps : np.ndarray - Keypoints in the right image
        """
        left_kps, left_des = self.extractor(left_image)
        right_kps, right_des = self.extractor(right_image)

        matches = self.matcher(left_kps, right_kps)
        return matches
