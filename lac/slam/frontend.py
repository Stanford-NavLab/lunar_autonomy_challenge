"""Frontend for SLAM"""

import numpy as np
from rich import print

from lac.slam.semantic_feature_tracker import SemanticFeatureTracker
from lac.perception.segmentation import UnetSegmentation
from lac.perception.depth import (
    stereo_depth_from_segmentation,
    compute_rock_coords_rover_frame,
    compute_rock_radii,
)
from lac.localization.imu_recovery import ImuEstimator
from lac.params import STEREO_BASELINE, FL_X, DT

KEYFRAME_INTERVAL = 20  # Interval for keyframe selection (steps)


class Frontend:
    """Frontend for SLAM"""

    def __init__(
        self,
        feature_tracker: SemanticFeatureTracker,
        back_feature_tracker: SemanticFeatureTracker = None,
        initial_pose: np.ndarray = None,
    ):
        # Modules
        self.feature_tracker = feature_tracker
        self.segmentation = UnetSegmentation()
        self.back_feature_tracker = back_feature_tracker
        self.imu_estimator = ImuEstimator(initial_pose)

        # State variables
        self.current_velocity = np.zeros(3)

    def initialize(self, data: dict):
        """Initialize from initial frame"""
        left_pred = self.segmentation.predict(data["FrontLeft"])
        self.feature_tracker.initialize(data["FrontLeft"], data["FrontRight"], left_pred)

        if self.back_feature_tracker is not None:
            back_left_pred = self.segmentation.predict(data["BackLeft"])
            self.back_feature_tracker.initialize(
                data["BackLeft"], data["BackRight"], back_left_pred
            )

    def process_frame(self, data: dict):
        """Process the data

        data : dict
            - step : int - step number
            - FrontLeft : np.ndarray (H, W, 3) - front left image
            - FrontRight : np.ndarray (H, W, 3) - front right image
            - imu : np.ndarray (6,) - IMU measurement

        """
        # Segmentation
        # left_pred = self.segmentation.predict(data["FrontLeft"])
        left_rock_seg_masks, left_labels, left_pred = self.segmentation.segment_rocks(
            data["FrontLeft"], output_pred=True
        )
        right_rock_seg_masks, right_labels = self.segmentation.segment_rocks(data["FrontRight"])

        # Rock detection
        stereo_depth_results = stereo_depth_from_segmentation(
            left_rock_seg_masks, right_rock_seg_masks, STEREO_BASELINE, FL_X
        )
        rock_coords = compute_rock_coords_rover_frame(
            stereo_depth_results, self.feature_tracker.cam_config
        )
        rock_radii = compute_rock_radii(stereo_depth_results)

        # Feature tracking and VO
        odometry = self.feature_tracker.track_pnp(data["FrontLeft"], data["FrontRight"], left_pred)
        data["odometry_source"] = "VO"

        # If VO failed, use IMU odometry instead
        if odometry is None:
            print("[bold orange]Using IMU odometry")
            for measurement in data["imu_measurements"]:
                self.imu_estimator.update(measurement, exact=False)
            odometry = np.linalg.inv(data["prev_pose"]) @ self.imu_estimator.get_pose()
            # NOTE: Above estimation seems to blow up occasionally, use identity for now
            odometry = np.eye(4)
            data["odometry_source"] = "IMU"
        else:
            # Update IMU estimator with the odometry
            new_pose = data["prev_pose"] @ odometry
            self.imu_estimator.update_pose_from_vo(new_pose)

        self.current_velocity = odometry[:3, 3] / (2 * DT)

        # Back camera feature tracking
        if self.back_feature_tracker is not None:
            back_left_pred = self.segmentation.predict(data["BackLeft"])
            self.back_feature_tracker.track_pnp(data["BackLeft"], data["BackRight"], back_left_pred)
            data["back_tracked_points"] = self.back_feature_tracker.tracked_points

        # Add frontend outputs
        data["odometry"] = odometry
        # TODO: implement proper keyframe selection based on motion
        data["keyframe"] = data["step"] % KEYFRAME_INTERVAL == 0
        data["tracked_points"] = self.feature_tracker.tracked_points
        data["rock_data"] = {"centers": rock_coords, "radii": rock_radii}
        data["rock_depth"] = stereo_depth_results
        data["left_pred"] = left_pred

        return data
