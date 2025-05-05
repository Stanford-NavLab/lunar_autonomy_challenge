"""Methods for loop closure detection and pose estimation."""

import torch
import numpy as np

from lac.slam.feature_tracker import FeatureTracker, prune_features
from lac.perception.vision import solve_vision_pnp


def estimate_loop_closure_pose(
    tracker: FeatureTracker, left_img1, right_img1, left_img2, right_img2
) -> np.ndarray | None:
    """Estimate the loop closure pose using stereo images.

    Returns None if not enough points/matches or PnP fails.

    """
    feats_left1, feats_right1, stereo_matches1, depths1 = tracker.process_stereo(
        left_img1, right_img1
    )
    feats_left2, feats_right2, stereo_matches2, depths2 = tracker.process_stereo(
        left_img2, right_img2
    )

    matched_feats1 = prune_features(feats_left1, stereo_matches1[:, 0])
    matched_pts_left1 = matched_feats1["keypoints"][0]
    points_local1 = tracker.project_stereo(np.eye(4), matched_pts_left1, depths1)

    matches12_left = tracker.match_feats(feats_left1, feats_left2, min_score=0.6)

    stereo_indices = stereo_matches1[:, 0]
    frame_indices = matches12_left[:, 0]

    common_indices = torch.tensor(
        list(set(stereo_indices.cpu().numpy()) & set(frame_indices.cpu().numpy()))
    ).cuda()
    frame_common = matches12_left[torch.isin(frame_indices, common_indices)]

    points3D = points_local1[torch.isin(stereo_indices, common_indices).cpu().numpy()]
    points2D = feats_left2["keypoints"][0][frame_common[:, 1]].cpu().numpy()

    # relative pose from frame 1 to frame 2, i.e. pose2 = pose1 @ rel_pose
    rel_pose = solve_vision_pnp(points3D, points2D)

    return rel_pose


def keyframe_estimate_loop_closure_pose(
    tracker: FeatureTracker, keyframe_data: tuple, left_img2: np.ndarray
) -> np.ndarray | None:
    """Estimate the loop closure pose using stereo images.

    Returns None if not enough points/matches or PnP fails.

    """
    feats_left1, feats_right1, stereo_matches1, depths1 = keyframe_data
    feats_left2 = tracker.extract_feats(left_img2)

    matched_feats1 = prune_features(feats_left1, stereo_matches1[:, 0])
    matched_pts_left1 = matched_feats1["keypoints"][0]
    points_local1 = tracker.project_stereo(np.eye(4), matched_pts_left1, depths1)

    matches12_left = tracker.match_feats(feats_left1, feats_left2, min_score=0.6)

    stereo_indices = stereo_matches1[:, 0]
    frame_indices = matches12_left[:, 0]

    common_indices = torch.tensor(
        list(set(stereo_indices.cpu().numpy()) & set(frame_indices.cpu().numpy()))
    ).cuda()
    frame_common = matches12_left[torch.isin(frame_indices, common_indices)]

    points3D = points_local1[torch.isin(stereo_indices, common_indices).cpu().numpy()]
    points2D = feats_left2["keypoints"][0][frame_common[:, 1]].cpu().numpy()

    # relative pose from frame 1 to frame 2, i.e. pose2 = pose1 @ rel_pose
    rel_pose = solve_vision_pnp(points3D, points2D)

    return rel_pose
