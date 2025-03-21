"""Basic visual odometry based on pyslam"""

import numpy as np
import cv2
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from lac.perception.depth import project_pixel_to_rover
from lac.utils.frames import (
    apply_transform,
    invert_transform_mat,
    OPENCV_TO_CAMERA_PASSIVE,
    get_cam_pose_rover,
)
from lac.util import grayscale_to_3ch_tensor
from lac.params import FL_X, STEREO_BASELINE, CAMERA_INTRINSICS


class StereoVisualOdometry:
    """3D-2D tracking with PnP"""

    def __init__(self, cam_config: dict):
        self.cam_config = cam_config

        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        self.matcher = LightGlue(features="superpoint").eval().cuda()

        self.feats0_left = None
        self.matches0_stereo = None
        self.points0_world = None
        self.rover_pose = None

    def process_stereo(self, left_image: np.ndarray, right_image: np.ndarray):
        """Process stereo pair to get features and depths"""
        feats_left = self.extractor.extract(grayscale_to_3ch_tensor(left_image).cuda())
        feats_right = self.extractor.extract(grayscale_to_3ch_tensor(right_image).cuda())
        matches_stereo = self.matcher({"image0": feats_left, "image1": feats_right})
        matches_stereo = rbd(matches_stereo)["matches"]
        matched_kps_left = rbd(feats_left)["keypoints"][matches_stereo[..., 0]]
        matched_kps_right = rbd(feats_right)["keypoints"][matches_stereo[..., 1]]
        disparities = matched_kps_left[..., 0] - matched_kps_right[..., 0]
        depths = FL_X * STEREO_BASELINE / disparities

        return feats_left, feats_right, matches_stereo, depths

    def initialize(self, initial_pose: np.ndarray, left_image: np.ndarray, right_image: np.ndarray):
        """Initialize world points and features"""
        feats0_left, feats0_right, matches0_stereo, depths0 = self.process_stereo(
            left_image, right_image
        )
        matched_kps0_left = rbd(feats0_left)["keypoints"][matches0_stereo[..., 0]]

        matched_kps0_left = matched_kps0_left.cpu().numpy()
        depths0 = depths0.cpu().numpy()

        points0_rover = []
        for pixel, depth in zip(matched_kps0_left, depths0):
            point_rover = project_pixel_to_rover(pixel, depth, "FrontLeft", self.cam_config)
            points0_rover.append(point_rover)
        points0_rover = np.array(points0_rover)
        points0_world = apply_transform(initial_pose, points0_rover)

        self.feats0_left = feats0_left
        self.matches0_stereo = matches0_stereo
        self.points0_world = points0_world
        self.rover_pose = initial_pose

    def track(self, left_image: np.ndarray, right_image: np.ndarray):
        """Frame-to-frame tracking"""
        # Process new frame
        feats1_left, feats1_right, matches1_stereo, depths1 = self.process_stereo(
            left_image, right_image
        )
        # Match with previous frame
        matches01_left = self.matcher({"image0": self.feats0_left, "image1": feats1_left})
        matches01_left = rbd(matches01_left)["matches"]
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
        points2D = rbd(feats1_left)["keypoints"][frame_common[:, 1]].cpu().numpy()

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
            R, _ = cv2.Rodrigues(rvec)
            T = np.hstack((R, tvec))
            est_pose = np.vstack((T, [0, 0, 0, 1]))
            w_T_c = invert_transform_mat(est_pose)
            w_T_c[:3, :3] = w_T_c[:3, :3] @ OPENCV_TO_CAMERA_PASSIVE
            rover_to_cam = get_cam_pose_rover("FrontLeft")
            cam_to_rover = invert_transform_mat(rover_to_cam)
            rover_pose = w_T_c @ cam_to_rover

            matched_kps1_left = rbd(feats1_left)["keypoints"][matches1_stereo[..., 0]]
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
