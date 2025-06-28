"""Semantic Feature Tracker"""

import numpy as np
import torch
from dataclasses import dataclass

from lightglue.utils import rbd

from lac.slam.feature_tracker import FeatureTracker, prune_features
from lac.perception.vision import solve_vision_pnp

EXTRACTOR_MAX_KEYPOINTS = 2048
MAX_STEREO_MATCHES = 1024
MIN_SCORE = 0.01  # (currently unused)
MAX_DEPTH = 15.0  # [m]


@dataclass
class TrackedPoints:
    ids: np.ndarray  # Track IDs
    points: np.ndarray  # 2D pixel coordinates in left image
    feats: dict  # Extracted features
    points_local: np.ndarray  # 3D points projected in rover frame
    depths: np.ndarray  # Triangulated depths
    labels: np.ndarray  # Semantic labels
    lengths: np.ndarray  # Number of frames tracked for

    def get_by_id(self, track_id):
        mask = self.ids == track_id
        return {
            "point": self.points[mask][0],
            "depth": self.depths[mask][0],
            "feat": rbd(prune_features(self.feats, mask)),
            "label": self.labels[mask][0],
            "point_local": self.points_local[mask][0],
            "length": self.lengths[mask],
        }


class SemanticFeatureTracker(FeatureTracker):
    def __init__(
        self, cam_config: dict, max_keypoints: int = EXTRACTOR_MAX_KEYPOINTS, cam: str = "FrontLeft"
    ):
        """Initialize the semantic feature tracker"""
        super().__init__(cam_config=cam_config, max_keypoints=max_keypoints)
        self.tracked_points = None
        self.max_id = 0
        self.cam = cam

    def initialize(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        left_semantic_pred: np.ndarray,
    ):
        """Initialize world points and features"""
        feats_left, feats_right, matches_stereo, depths = self.process_stereo(
            left_image,
            right_image,
            max_matches=MAX_STEREO_MATCHES,
            max_depth=MAX_DEPTH,
            return_matched_feats=True,
        )
        left_pts = feats_left["keypoints"][0].cpu().numpy()
        pixels = np.round(left_pts).astype(int)
        labels = left_semantic_pred[pixels[:, 1], pixels[:, 0]]
        num_pts = len(left_pts)
        points_local = self.project_stereo(np.eye(4), left_pts, depths, cam_name=self.cam)

        self.tracked_points = TrackedPoints(
            ids=np.arange(num_pts),  # 0, 1, 2, ..., num_pts - 1
            points=left_pts,
            feats=feats_left,
            points_local=points_local,
            depths=depths.cpu().numpy(),
            labels=labels,
            lengths=np.zeros(MAX_STEREO_MATCHES, dtype=int),
        )
        self.max_id = num_pts  # Next ID to assign

    def update_tracks(
        self,
        feats_left: dict,
        depths: torch.Tensor,
        matches_frame: np.ndarray,
        left_semantic_pred: np.ndarray,
    ):
        """Update tracked points"""
        left_pts = feats_left["keypoints"][0].cpu().numpy()
        num_new_pts = len(left_pts)

        # Project points
        points_local = self.project_stereo(np.eye(4), left_pts, depths, cam_name=self.cam)

        # Matched features get old track IDs, and unmatched features get new IDs
        new_track_ids = np.zeros(num_new_pts, dtype=int) - 1
        idxs = set(range(num_new_pts))
        matched_idxs = matches_frame[:, 1]
        unmatched_idxs = list(idxs - set(matched_idxs))
        matched_ids = self.tracked_points.ids[matches_frame[:, 0]]
        new_track_ids[matched_idxs] = matched_ids

        new_ids = np.arange(self.max_id, self.max_id + len(unmatched_idxs))
        new_track_ids[unmatched_idxs] = new_ids
        self.max_id += len(unmatched_idxs)

        pixels = np.round(left_pts).astype(int)

        self.tracked_points.ids = new_track_ids
        self.tracked_points.points = left_pts
        self.tracked_points.feats = feats_left
        self.tracked_points.points_local = points_local
        self.tracked_points.depths = depths.cpu().numpy()
        self.tracked_points.labels = left_semantic_pred[pixels[:, 1], pixels[:, 0]]

        # Update lengths
        new_lengths = np.zeros(num_new_pts, dtype=int)
        new_lengths[matched_idxs] = self.tracked_points.lengths[matches_frame[:, 0]] + 1
        self.tracked_points.lengths = new_lengths

    def track_pnp(
        self, left_image: np.ndarray, right_image: np.ndarray, left_semantic_pred: np.ndarray
    ) -> np.ndarray:
        """Track points and estimate odometry with PnP

        TODO: handle semantics, check consistency with matching
        TODO: filter based on depth, scores, etc.

        """
        # Extract features and stereo matching
        feats_left, feats_right, matches_stereo, depths = self.process_stereo(
            left_image,
            right_image,
            max_matches=MAX_STEREO_MATCHES,
            max_depth=MAX_DEPTH,
            return_matched_feats=True,
        )
        # Match with previous frame
        matches_frame = self.match_feats(self.tracked_points.feats, feats_left).cpu().numpy()

        # Estimate odometry with PnP
        points3D = self.tracked_points.points_local[matches_frame[:, 0]]
        points2D = feats_left["keypoints"][0][matches_frame[:, 1]].cpu().numpy()
        odometry = solve_vision_pnp(points3D, points2D)

        # Update tracks (limit based on number and depth)
        self.update_tracks(feats_left, depths, matches_frame, left_semantic_pred)

        return odometry
