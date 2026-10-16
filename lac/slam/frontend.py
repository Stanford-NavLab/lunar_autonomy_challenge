"""Frontend for SLAM"""

import numpy as np
from rich import print
from scipy.spatial.transform import Rotation

from lac.slam.semantic_feature_tracker import SemanticFeatureTracker
from lac.perception.segmentation import UnetSegmentation
from lac.perception.depth import (
    stereo_depth_from_segmentation,
    compute_rock_coords_rover_frame,
    compute_rock_radii,
)
from lac.localization.imu_recovery import ImuEstimator
from lac.params import STEREO_BASELINE, FL_X, DT
from lac.util import rotation_matrix_error

KEYFRAME_INTERVAL = 20  # Interval for keyframe selection (steps)
# A frame on the keyframe cadence becomes a keyframe only if the rover moved at least this much
# (net motion composed since the last keyframe). A stationary rover otherwise adds a keyframe every
# 2 s at one spot, each a loop-closure candidate for all the others (Map 2 p13, stuck at the
# lander: 200k loop closures).
KEYFRAME_MIN_TRANSLATION = 0.05  # [m]
KEYFRAME_MIN_ROTATION = 2.0  # [deg]

# VO sanity gate per image step (0.1 s): a larger PnP step is rejected and replaced by the IMU
# fallback, as if PnP had failed. Good Map 1 runs peak at 0.106 m and 6 deg per step; the outlier
# that wrecked tab1_r1 p1 (RMSE 3.98 m) was 13.15 m / 60 deg.
VO_MAX_STEP_TRANSLATION = 0.2  # [m]
VO_MAX_STEP_ROTATION = 10.0  # [deg]


def vo_step_valid(odometry: np.ndarray) -> bool:
    """Whether a PnP odometry step passes the VO sanity gate"""
    return (
        np.linalg.norm(odometry[:3, 3]) <= VO_MAX_STEP_TRANSLATION
        and rotation_matrix_error(odometry[:3, :3], np.eye(3)) <= VO_MAX_STEP_ROTATION
    )


def is_keyframe(step: int, motion: np.ndarray) -> bool:
    """Keyframe cadence plus the motion gate (motion: odometry composed since the last keyframe)"""
    return step % KEYFRAME_INTERVAL == 0 and (
        np.linalg.norm(motion[:3, 3]) >= KEYFRAME_MIN_TRANSLATION
        or rotation_matrix_error(motion[:3, :3], np.eye(3)) >= KEYFRAME_MIN_ROTATION
    )


def gravity_aligned_pose(pose: np.ndarray, accel: np.ndarray) -> np.ndarray:
    """pose with roll and pitch from a static (averaged) accelerometer reading; yaw, position kept.

    At rest the IMU measures the reaction to gravity, accel = R^T [0, 0, g] (see
    lac.localization.imu_recovery.recover_translation), i.e. accel / g is the third row of R,
    [-sin(pitch), sin(roll) cos(pitch), cos(roll) cos(pitch)] for ZYX Euler angles.
    """
    roll = np.arctan2(accel[1], accel[2])
    pitch = np.arctan2(-accel[0], np.hypot(accel[1], accel[2]))
    yaw = np.arctan2(pose[1, 0], pose[0, 0])
    aligned = pose.copy()
    aligned[:3, :3] = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    return aligned


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
        self.keyframe_motion = np.eye(4)  # Odometry composed since the last keyframe

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
        if odometry is not None and not vo_step_valid(odometry):
            print("[bold orange]VO step rejected by the sanity gate")
            odometry = None

        # If VO failed or was rejected, use IMU odometry instead
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
        self.keyframe_motion = self.keyframe_motion @ odometry
        data["keyframe"] = is_keyframe(data["step"], self.keyframe_motion)
        if data["keyframe"]:
            self.keyframe_motion = np.eye(4)
        data["tracked_points"] = self.feature_tracker.tracked_points
        data["rock_data"] = {"centers": rock_coords, "radii": rock_radii}
        data["rock_depth"] = stereo_depth_results
        data["left_pred"] = left_pred

        return data
