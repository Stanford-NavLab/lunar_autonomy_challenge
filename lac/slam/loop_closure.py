"""Methods for loop closure detection and pose estimation."""

import torch
import numpy as np

from lac.slam.feature_tracker import FeatureTracker, prune_features
from lac.perception.vision import solve_vision_pnp

MIN_MATCHES = 120
MIN_SCORE = 0.25
# A loop closure is accepted only if its PnP has at least this many RANSAC inliers. The match count alone accepted
# relative poses off by up to tens of cm (5 mm loop-closure noise). Offline replays of 4 stored runs
# (analysis/bugfix_replay.py --probe, 3608 candidates, translation error vs the GT relative pose; candidates with
# >= 60 inliers stay below 3.4 cm): error > 5 cm in 0.1% of candidates with >= 30 inliers, 1-2% with 20-29, 8% with
# 15-19, 10% with 10-14, 39% below 10. The gate rejects 78 of the 80 match-count-accepted candidates with > 5 cm
# (and 1101 of 3037 accepted candidates in all).
MIN_INLIERS = 30


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

    matches12_left = tracker.match_feats(feats_left1, feats_left2, min_score=MIN_SCORE)

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
    tracker: FeatureTracker,
    keyframe_data: tuple,
    left_img2: np.ndarray,
    min_score: float = MIN_SCORE,
    min_matches: int = MIN_MATCHES,
    count_stereo_only: bool = False,
) -> np.ndarray | None:
    """Estimate the loop closure pose using stereo images.

    Returns None if not enough points/matches, PnP fails or has fewer than MIN_INLIERS inliers.

    count_stereo_only : if True, the min_matches test counts only the matches whose keyframe feature has
        stereo depth (the 2D-3D set passed to PnP); if False (paper), all matches above min_score.

    """
    feats_left1, feats_right1, stereo_matches1, depths1 = keyframe_data
    feats_left2 = tracker.extract_feats(left_img2)

    matched_feats1 = prune_features(feats_left1, stereo_matches1[:, 0])
    matched_pts_left1 = matched_feats1["keypoints"][0]
    points_local1 = tracker.project_stereo(np.eye(4), matched_pts_left1, depths1)

    matches12_left = tracker.match_feats(feats_left1, feats_left2, min_score=min_score)

    stereo_indices = stereo_matches1[:, 0]
    frame_indices = matches12_left[:, 0]

    common_indices = torch.tensor(
        list(set(stereo_indices.cpu().numpy()) & set(frame_indices.cpu().numpy()))
    ).cuda()
    frame_common = matches12_left[torch.isin(frame_indices, common_indices)]

    num_matches = len(frame_common) if count_stereo_only else len(matches12_left)
    if num_matches < min_matches:
        print("Not enough matches for loop closure")
        return None, num_matches

    points3D = points_local1[torch.isin(stereo_indices, common_indices).cpu().numpy()]
    points2D = feats_left2["keypoints"][0][frame_common[:, 1]].cpu().numpy()

    # relative pose from frame 1 to frame 2, i.e. pose2 = pose1 @ rel_pose
    rel_pose, num_inliers = solve_vision_pnp(points3D, points2D, return_inliers=True)
    if num_inliers < MIN_INLIERS:
        print("Not enough PnP inliers for loop closure")
        return None, num_matches

    return rel_pose, num_matches
