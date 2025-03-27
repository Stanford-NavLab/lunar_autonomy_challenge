"""Stereo Visual Odometry with LightGlue and 3D-2D PnP"""

import numpy as np
import cv2
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from lac.slam.feature_tracker import FeatureTracker, prune_features
from lac.perception.depth import project_pixel_to_rover
from lac.utils.frames import (
    apply_transform,
    invert_transform_mat,
    OPENCV_TO_CAMERA_PASSIVE,
    get_cam_pose_rover,
)
from lac.params import CAMERA_INTRINSICS


MIN_MATCH_SCORE = 0.5


class StereoVisualOdometry:
    """3D-2D tracking with PnP"""

    def __init__(self, cam_config: dict):
        self.cam_config = cam_config

        self.tracker = FeatureTracker(cam_config, max_keypoints=2048, max_stereo_matches=1000)

        self.feats0_left = None
        self.matches0_stereo = None
        self.points0_world = None
        self.rover_pose = None

    def initialize(self, initial_pose: np.ndarray, left_image: np.ndarray, right_image: np.ndarray):
        """Initialize world points and features"""
        feats_left, feats_right, matches_stereo, depths = self.tracker.process_stereo(
            left_image, right_image, min_score=MIN_MATCH_SCORE
        )
        matched_feats = prune_features(feats_left, matches_stereo[:, 0])
        matched_pts_left = matched_feats["keypoints"][0]
        points_world = self.tracker.project_stereo(initial_pose, matched_pts_left, depths)

        self.feats0_left = feats_left
        self.matches0_stereo = matches_stereo
        self.points0_world = points_world
        self.rover_pose = initial_pose

    def get_pose(self):
        """Get the current pose estimate"""
        return self.rover_pose

    def track(self, left_image: np.ndarray, right_image: np.ndarray):
        """Frame-to-frame tracking"""
        # Process new frame
        feats1_left, feats1_right, matches1_stereo, depths1 = self.tracker.process_stereo(
            left_image, right_image, min_score=MIN_MATCH_SCORE
        )
        # Match with previous frame
        matches01_left = self.tracker.match_feats(
            self.feats0_left, feats1_left, min_score=MIN_MATCH_SCORE
        )

        stereo_indices = self.matches0_stereo[:, 0]
        frame_indices = matches01_left[:, 0]

        # Find overlapping matches between frame 0 stereo and frame 0-1 matches
        common_indices = torch.tensor(
            list(set(stereo_indices.cpu().numpy()) & set(frame_indices.cpu().numpy()))
        ).cuda()
        frame_common = matches01_left[torch.isin(frame_indices, common_indices)]
        points0_world_common = self.points0_world[
            torch.isin(stereo_indices, common_indices).cpu().numpy()
        ]

        # PnP
        points3D = points0_world_common
        points2D = feats1_left["keypoints"][0][frame_common[:, 1]].cpu().numpy()

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points3D,
            imagePoints=points2D,
            cameraMatrix=CAMERA_INTRINSICS,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=8.0,
            iterationsCount=100,
        )
        if success:
            # TODO: clean up this code
            R, _ = cv2.Rodrigues(rvec)
            T = np.hstack((R, tvec))
            est_pose = np.vstack((T, [0, 0, 0, 1]))
            w_T_c = invert_transform_mat(est_pose)
            w_T_c[:3, :3] = w_T_c[:3, :3] @ OPENCV_TO_CAMERA_PASSIVE
            rover_to_cam = get_cam_pose_rover("FrontLeft")
            cam_to_rover = invert_transform_mat(rover_to_cam)
            rover_pose = w_T_c @ cam_to_rover

            matched_kps1_left = feats1_left["keypoints"][0][matches1_stereo[..., 0]]
            matched_kps1_left = matched_kps1_left.cpu().numpy()
            depths1 = depths1.cpu().numpy()

            points1_rover = []
            for pixel, depth in zip(matched_kps1_left, depths1):
                point_rover = project_pixel_to_rover(pixel, depth, "FrontLeft", self.cam_config)
                points1_rover.append(point_rover)
            points1_rover = np.array(points1_rover)
            points1_world = apply_transform(rover_pose, points1_rover)

            self.feats0_left = feats1_left
            self.matches0_stereo = matches1_stereo
            self.points0_world = points1_world
            self.rover_pose = rover_pose
        else:
            print("PnP failed to estimate motion.")
