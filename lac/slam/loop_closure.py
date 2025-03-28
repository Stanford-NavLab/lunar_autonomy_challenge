"""Methods for loop closure detection and pose estimation."""

import cv2
import torch
import numpy as np

from lac.slam.feature_tracker import FeatureTracker, prune_features
from lac.utils.frames import (
    invert_transform_mat,
    OPENCV_TO_CAMERA_PASSIVE,
)
from lac.params import CAMERA_INTRINSICS


def estimate_loop_closure_pose(
    tracker: FeatureTracker, left_img1, right_img1, left_img2, right_img2
):
    feats_left1, feats_right1, stereo_matches1, depths1 = tracker.process_stereo(
        left_img1, right_img1
    )
    feats_left2, feats_right2, stereo_matches2, depths2 = tracker.process_stereo(
        left_img2, right_img2
    )

    matched_feats1 = prune_features(feats_left1, stereo_matches1[:, 0])
    matched_pts_left1 = matched_feats1["keypoints"][0]
    points_local1 = tracker.project_stereo(np.eye(4), matched_pts_left1, depths1)

    matches12_left = tracker.match_feats(feats_left1, feats_left2)

    stereo_indices = stereo_matches1[:, 0]
    frame_indices = matches12_left[:, 0]

    common_indices = torch.tensor(
        list(set(stereo_indices.cpu().numpy()) & set(frame_indices.cpu().numpy()))
    ).cuda()
    frame_common = matches12_left[torch.isin(frame_indices, common_indices)]

    points3D = points_local1[torch.isin(stereo_indices, common_indices).cpu().numpy()]
    points2D = feats_left2["keypoints"][0][frame_common[:, 1]].cpu().numpy()

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=points3D,
        imagePoints=points2D,
        cameraMatrix=CAMERA_INTRINSICS,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=8.0,
        iterationsCount=100,
    )

    R, _ = cv2.Rodrigues(rvec)
    T = np.hstack((R, tvec))
    est_pose = np.vstack((T, [0, 0, 0, 1]))
    w_T_c = invert_transform_mat(est_pose)
    w_T_c[:3, :3] = w_T_c[:3, :3] @ OPENCV_TO_CAMERA_PASSIVE
    rel_pose = w_T_c  # relative pose from frame 1 to frame 2, i.e. pose2 = pose1 @ rel_pose
    return rel_pose
