"""Estimate antenna-to-antenna relative pose from charging fiducial detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import apriltag
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lac.perception.fiducials import get_tag_corners_local
from lac.perception.vision import get_camera_intrinsics
from lac.utils.frames import get_cam_pose_rover, invert_transform_mat, opencv_to_camera


@dataclass
class DockingPoseEstimatorConfig:
    cam_name: str = "Right"
    target_tag_id: int = 69
    lander_target_point_lander: np.ndarray | None = None
    lander_target_axis_lander: np.ndarray | None = None
    rover_target_point_rover: np.ndarray | None = None
    rover_target_axis_rover: np.ndarray | None = None
    locator_point_lander: np.ndarray | None = None
    lander_target_rpy_lander: np.ndarray | None = None
    tag_size_m: float = 0.253


class FiducialDockingPoseEstimator:
    """Pose estimator for rover/lander charging antenna alignment."""

    def __init__(self, camera_config: dict, config: DockingPoseEstimatorConfig | None = None):
        self.cfg = config or DockingPoseEstimatorConfig()
        self.camera_config = camera_config
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
        self.K = get_camera_intrinsics(self.cfg.cam_name, self.camera_config)

        self.tag_size = float(self.cfg.tag_size_m)
        self.tag_corners_local = get_tag_corners_local(self.tag_size).astype(np.float32)
        self.cam_T_rover = invert_transform_mat(get_cam_pose_rover(self.cfg.cam_name))
        self.R_cam_to_rover = self.cam_T_rover[:3, :3]
        self.t_cam_to_rover = self.cam_T_rover[:3, 3]

        self.p_lander_locator_lander = np.asarray(
            (
                self.cfg.locator_point_lander
                if self.cfg.locator_point_lander is not None
                else [0.0, 0.662, 0.325]
            ),
            dtype=np.float64,
        )
        self.p_lander_target_lander = np.asarray(
            (
                self.cfg.lander_target_point_lander
                if self.cfg.lander_target_point_lander is not None
                else [0.0, 1.452, 0.509]
            ),
            dtype=np.float64,
        )
        self.p_rover_target_rover = np.asarray(
            (
                self.cfg.rover_target_point_rover
                if self.cfg.rover_target_point_rover is not None
                else [0.0, -0.20813, 0.34603]
            ),
            dtype=np.float64,
        )
        self.v_lander_target_lander = np.asarray(
            (
                self.cfg.lander_target_axis_lander
                if self.cfg.lander_target_axis_lander is not None
                else [0.0, 1.0, 0.0]
            ),
            dtype=np.float64,
        )
        self.v_lander_target_lander /= np.linalg.norm(self.v_lander_target_lander) + 1e-12
        self.v_rover_target_rover = np.asarray(
            (
                self.cfg.rover_target_axis_rover
                if self.cfg.rover_target_axis_rover is not None
                else [0.0, -1.0, 0.0]
            ),
            dtype=np.float64,
        )
        self.v_rover_target_rover /= np.linalg.norm(self.v_rover_target_rover) + 1e-12

        # Locator frame assumption: aligned with lander frame axes, centered at locator.
        self.p_lander_target_tag = self.p_lander_target_lander - self.p_lander_locator_lander
        self.R_lander_target_lander = Rotation.from_euler(
            "xyz",
            (
                self.cfg.lander_target_rpy_lander
                if self.cfg.lander_target_rpy_lander is not None
                else [0.0, 0.0, np.pi]
            ),
            degrees=False,
        ).as_matrix()
        self.R_lander_target_tag = self.R_lander_target_lander.copy()

    def detect(self, gray_img: np.ndarray) -> list[Any]:
        return self.detector.detect(gray_img)

    def _select_target_detection(self, detections: list[Any]) -> Any | None:
        tagged = [d for d in detections if int(getattr(d, "tag_id", -1)) == self.cfg.target_tag_id]
        if not tagged:
            return None
        if len(tagged) == 1:
            return tagged[0]
        return max(tagged, key=lambda d: cv2.contourArea(np.asarray(d.corners, dtype=np.float32)))

    def estimate(
        self, gray_img: np.ndarray | None
    ) -> tuple[dict[str, Any] | None, list[Any], Any | None]:
        """Estimate docking-relevant relative pose from one image."""
        if gray_img is None:
            return None, [], None
        detections = self.detect(gray_img)
        det = self._select_target_detection(detections)
        if det is None:
            return None, detections, None

        success, rvec, tvec = cv2.solvePnP(
            objectPoints=self.tag_corners_local,
            imagePoints=np.asarray(det.corners, dtype=np.float32),
            cameraMatrix=self.K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None, detections, det

        R_ocv_tag, _ = cv2.Rodrigues(rvec)
        t_ocv_tag = tvec.reshape(3)
        # Convert OpenCV camera convention to rover camera convention.
        R_cam_tag = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64) @ R_ocv_tag
        p_cam_tag = opencv_to_camera(t_ocv_tag[None, :])[0]
        T_cam_tag = np.eye(4, dtype=np.float64)
        T_cam_tag[:3, :3] = R_cam_tag
        T_cam_tag[:3, 3] = p_cam_tag

        # Lander target (antenna for side mode) in camera frame.
        p_cam_lander_target = p_cam_tag + self.p_lander_target_tag @ R_cam_tag.T
        R_cam_lander_target = R_cam_tag @ self.R_lander_target_tag
        v_cam_lander_target = self.v_lander_target_lander @ R_cam_tag.T

        # Convert to rover frame.
        p_rover_lander_target = p_cam_lander_target @ self.R_cam_to_rover.T + self.t_cam_to_rover
        R_rover_lander_target = self.R_cam_to_rover @ R_cam_lander_target
        v_rover_lander_target = v_cam_lander_target @ self.R_cam_to_rover.T
        v_rover_lander_target /= np.linalg.norm(v_rover_lander_target) + 1e-12

        rel_target_rover = p_rover_lander_target - self.p_rover_target_rover
        dot_val = float(
            np.clip(np.dot(self.v_rover_target_rover, v_rover_lander_target), -1.0, 1.0)
        )
        axis_angle = float(np.arccos(dot_val))
        rover_axis_xy = self.v_rover_target_rover[:2]
        lander_axis_xy = v_rover_lander_target[:2]
        rover_axis_xy_n = rover_axis_xy / (np.linalg.norm(rover_axis_xy) + 1e-12)
        lander_axis_xy_n = lander_axis_xy / (np.linalg.norm(lander_axis_xy) + 1e-12)
        # Signed in rover XY plane: positive is CCW from rover axis to lander axis.
        cross_z = float(
            rover_axis_xy_n[0] * lander_axis_xy_n[1] - rover_axis_xy_n[1] * lander_axis_xy_n[0]
        )
        dot_xy = float(np.clip(np.dot(rover_axis_xy_n, lander_axis_xy_n), -1.0, 1.0))
        axis_yaw_err = float(np.arctan2(cross_z, dot_xy))

        R_rel = self.R_lander_target_lander.T @ R_rover_lander_target
        rpy_rel = Rotation.from_matrix(R_rel).as_euler("xyz", degrees=False)
        rot_abs_max = float(np.max(np.abs(rpy_rel)))

        estimate = {
            "tag_id": int(det.tag_id),
            "R_cam_tag": R_cam_tag,
            "t_cam_tag": p_cam_tag,
            "T_cam_tag": T_cam_tag,
            "p_rover_lander_target_m": p_rover_lander_target,
            "p_rover_rover_target_m": self.p_rover_target_rover.copy(),
            "rel_target_pos_rover_m": rel_target_rover,
            "lander_axis_rover": v_rover_lander_target,
            "rover_axis_rover": self.v_rover_target_rover.copy(),
            "axis_angle_rad": axis_angle,
            "axis_yaw_err_rad": axis_yaw_err,
            "rel_target_rpy_rad": rpy_rel,
            "rot_abs_max_rad": rot_abs_max,
            "det_center_uv": np.asarray(det.center, dtype=np.float64),
        }
        return estimate, detections, det
