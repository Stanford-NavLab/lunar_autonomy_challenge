"""Frontend for SLAM"""

import numpy as np

from lac.slam.semantic_feature_tracker import SemanticFeatureTracker
from lac.perception.segmentation import UnetSegmentation

KEYFRAME_INTERVAL = 10  # Interval for keyframe selection


class Frontend:
    """Frontend for SLAM"""

    def __init__(self, camera_config: dict):
        self.feature_tracker = SemanticFeatureTracker(camera_config)
        self.segmentation = UnetSegmentation()

    def initialize(self, left_image: np.ndarray, right_image: np.ndarray):
        """Initialize from initial frame"""
        left_pred = self.segmentation.predict(left_image)
        self.feature_tracker.initialize(left_image, right_image, left_pred)

    def process_frame(self, data: dict):
        """Process the data

        data : dict
            - step : int - step number
            - left_image : np.ndarray (H, W, 3) - left image
            - right_image : np.ndarray (H, W, 3) - right image
            - imu : np.ndarray (6,) - IMU measurement

        """
        left_pred = self.segmentation.predict(data["left_image"])

        odometry = self.feature_tracker.track_pnp(data["left_image"], data["right_image"], left_pred)

        # If VO failed, use IMU odometry instead
        if odometry is None:
            # TODO: compute IMU odometry
            odometry = data["imu"]

        # Add frontend outputs
        data["odometry"] = odometry
        # TODO: implement proper keyframe selection based on motion
        data["keyframe"] = data["step"] % KEYFRAME_INTERVAL == 0

        return data
