"""Basic visual odometry based on pyslam"""

import numpy as np
import cv2

from lightglue import LightGlue, SuperPoint


class StereoVisualOdometry:
    def __init__(self):
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        self.matcher = LightGlue(features="superpoint").eval().cuda()

    def initialize(self, left_image, right_image):
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
