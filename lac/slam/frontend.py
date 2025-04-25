"""Frontend for SLAM"""

import numpy as np

from lac.slam.semantic_feature_tracker import SemanticFeatureTracker
from lac.perception.segmentation import UnetSegmentation

KEYFRAME_INTERVAL = 20  # Interval for keyframe selection (steps)


class Frontend:
    """Frontend for SLAM"""

    def __init__(self, feature_tracker: SemanticFeatureTracker):
        self.feature_tracker = feature_tracker
        self.segmentation = UnetSegmentation()

    def initialize(self, left_image: np.ndarray, right_image: np.ndarray):
        """Initialize from initial frame"""
        left_pred = self.segmentation.predict(left_image)
        self.feature_tracker.initialize(left_image, right_image, left_pred)

    def process_frame(self, data: dict):
        """Process the data

        data : dict
            - step : int - step number
            - FrontLeft : np.ndarray (H, W, 3) - front left image
            - FrontRight : np.ndarray (H, W, 3) - front right image
            - imu : np.ndarray (6,) - IMU measurement

        """
        left_pred = self.segmentation.predict(data["FrontLeft"])

        odometry = self.feature_tracker.track_pnp(data["FrontLeft"], data["FrontRight"], left_pred)

        # If VO failed, use IMU odometry instead
        if odometry is None:
            # TODO: compute IMU odometry
            odometry = data["imu"]

        # Add frontend outputs
        data["odometry"] = odometry
        # TODO: implement proper keyframe selection based on motion
        data["keyframe"] = data["step"] % KEYFRAME_INTERVAL == 0
        data["tracked_points"] = self.feature_tracker.tracked_points

        return data
