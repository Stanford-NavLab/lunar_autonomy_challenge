"""Semantic Feature Tracker"""

import numpy as np
import cv2
import torch
from dataclasses import dataclass

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from lac.slam.feature_tracker import FeatureTracker, prune_features
from lac.perception.vision import solve_vision_pnp
from lac.perception.depth import project_pixels_to_rover
from lac.utils.frames import (
    apply_transform,
)
from lac.util import grayscale_to_3ch_tensor
from lac.params import FL_X, STEREO_BASELINE, MAX_DEPTH

EXTRACTOR_MAX_KEYPOINTS = 512
MAX_TRACKED_POINTS = 100
MAX_STEREO_MATCHES = 100
MIN_SCORE = 0.01


@dataclass
class TrackedPoints:
    ids: np.ndarray  # Track IDs
    points: np.ndarray  # 2D pixel coordinates in left image
    feats: dict  # Extracted features
    points_local: np.ndarray  # 3D points projected in rover frame
    depths: np.ndarray  # Triangulated depths
    labels: np.ndarray  # Semantic labels


class SemanticFeatureTracker(FeatureTracker):
    def __init__(
        self,
        cam_config: dict,
        max_keypoints: int = EXTRACTOR_MAX_KEYPOINTS,
        max_stereo_matches: int = MAX_STEREO_MATCHES,
    ):
        """Initialize the semantic feature tracker"""
        super().__init__(
            cam_config=cam_config,
            max_keypoints=max_keypoints,
            max_stereo_matches=max_stereo_matches,
        )
        self.tracked_points = None
        self.max_id = 0

    def initialize(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        left_semantic_pred: np.ndarray,
    ):
        """Initialize world points and features"""
        feats_left, feats_right, matches_stereo, depths = self.process_stereo(
            left_image, right_image, max_matches=MAX_STEREO_MATCHES, return_matched_feats=True
        )
        left_pts = feats_left["keypoints"][0].cpu().numpy()
        pixels = np.round(left_pts).astype(int)
        labels = left_semantic_pred[pixels[:, 1], pixels[:, 0]]
        num_pts = len(left_pts)
        points_local = self.project_stereo(np.eye(4), left_pts, depths)

        self.tracked_points = TrackedPoints(
            ids=np.arange(num_pts),  # 0, 1, 2, ..., num_pts - 1
            points=left_pts,
            feats=feats_left,
            points_local=points_local,
            depths=depths.cpu().numpy(),
            labels=labels,
        )
        self.max_id = num_pts  # Next ID to assign

    def update_tracks(
        self,
        feats_left: dict,
        matches_frame: torch.Tensor,
    ):
        """Update tracked points"""
        left_pts = feats_left["keypoints"][0].cpu().numpy()
        num_new_pts = len(left_pts)

        new_track_ids = np.zeros(num_new_pts, dtype=int) - 1
        idxs = set(range(num_new_pts))
        matched_idxs = matches_frame[:, 0].cpu().numpy()
        unmatched_idxs = list(idxs - set(matched_idxs))
        matched_ids = self.track_ids[matches_frame[:, 1].cpu().numpy()]
        new_track_ids[matched_idxs] = matched_ids

        new_ids = np.arange(self.max_id, self.max_id + len(unmatched_idxs))
        new_track_ids[unmatched_idxs] = new_ids

    def track_pnp(self, left_image: np.ndarray, right_image: np.ndarray, left_semantic_pred: np.ndarray):
        """Track points and estimate odometry with PnP"""
        # Extract features and stereo matching
        feats_left, feats_right, matches_stereo, depths = self.process_stereo(
            left_image, right_image, return_matched_feats=True
        )
        # Match with previous frame
        matches_frame = self.match_feats(self.prev_feats, feats_left)

        # PnP
        # odometry = solve_vision_pnp()
        points3D = self.tracked_points.points_local[matches_frame[:, 0].cpu().numpy()]
        points2D = feats_left["keypoints"][0][matches_frame[:, 1]].cpu().numpy()
        odometry = solve_vision_pnp(points3D, points2D)

        # Propagate tracks (limit based on number and depth)

        return odometry

    def track_keyframe(self, curr_pose: np.ndarray, left_image: np.ndarray, right_image: np.ndarray):
        """Initialize world points and features"""
        # Triangulate new points
        feats_left, feats_right, matches_stereo, depths = self.process_stereo(
            left_image,
            right_image,
            max_matches=MAX_STEREO_MATCHES,
            max_depth=10.0,
            return_matched_feats=True,
        )
        # new_feats = prune_features(feats_left, matches_stereo[:, 0])
        matched_pts_left = feats_left["keypoints"][0]
        num_new_pts = len(matched_pts_left)
        points_world = self.project_stereo(curr_pose, matched_pts_left, depths)

        # Match with previously tracked points
        matches = self.match_feats(feats_left, self.prev_feats)

        # Matched features get old track IDs, and unmatched features get new IDs
        new_track_ids = np.zeros(num_new_pts, dtype=int) - 1
        idxs = set(range(num_new_pts))
        matched_idxs = matches[:, 0].cpu().numpy()
        unmatched_idxs = list(idxs - set(matched_idxs))
        matched_ids = self.track_ids[matches[:, 1].cpu().numpy()]
        new_track_ids[matched_idxs] = matched_ids

        new_ids = np.arange(self.max_id, self.max_id + len(unmatched_idxs))
        new_track_ids[unmatched_idxs] = new_ids
        self.max_id += len(unmatched_idxs)

        self.track_ids = new_track_ids
        self.prev_image = left_image
        self.prev_pts = matched_pts_left.cpu().numpy()
        self.prev_pts_right = feats_right["keypoints"][0].cpu().numpy()
        self.prev_feats = feats_left
        self.world_points = points_world
