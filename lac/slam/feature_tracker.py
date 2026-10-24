"""Hybrid Optical Flow and LightGlue feature tracker

Uses stereo LightGlue matching at each keyframe (every 10 frames) to extract keypoints + descriptors
and triangulate points. In between, uses Lucas-Kanade optical flow to track points.

For now, we only track points in the left image.

NOTE: features and matches from LightGlue come with a batch dimension. We keep the batch dimension
      for the features (since the matching expects it), but remove it for the matches.

"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

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
# Stereo matches must lie on the same image row: the front pair is rectified (same intrinsics and orientation,
# horizontal baseline; docs/geometry.json), and the measured row differences are symmetric about 0. SuperPoint
# keypoints come from the image resized to 1024 px wide, so coordinates are multiples of 1.25 px. Of the stereo
# matches verified against GT terrain (near points within 5 cm of the GT height map, 3.1M matches, 4 runs,
# analysis/bugfix_stereo_gt_height.py), 99.91% have |row difference| <= 3.75 px; matches at >= 5 px are at least 6x more
# often off the terrain by > 0.2 m than matches at 0 px.
STEREO_MAX_ROW_DIFF = 4.0  # [px]

# Feature detector/descriptor/matcher used for stereo matching, frame-to-frame tracking and loop closure.
# "superpoint_lightglue" is the deployed frontend. "orb" and "sift" (OpenCV) are classical alternatives for
# the journal-revision feature ablation; everything downstream (stereo depth, PnP, tracks, loop closure) is
# unchanged. Classical features use the same feature dict layout as SuperPoint (keypoints [1, N, 2],
# keypoint_scores [1, N], descriptors [1, N, D], image_size [1, 2], on the GPU) so that the downstream code
# is shared. Classical matching is brute force (Hamming for ORB, L2 for SIFT) with Lowe's ratio test; a
# match's score is 1 - d1/d2 (nearest over second-nearest descriptor distance) and only ranks matches when
# max_matches truncates. min_score is a LightGlue match confidence and is not applied to classical matches:
# the ratio test is their acceptance test.
FEATURE_TYPES = ("superpoint_lightglue", "orb", "sift")
CLASSICAL_RATIO_TEST = 0.8


def prune_features(feats: dict, indices: np.ndarray) -> dict:
    return {k: v if k == "image_size" else v[0, indices].unsqueeze(0) for k, v in feats.items()}


def highest_score_features(feats: dict, N: int) -> dict:
    if len(feats["keypoints"][0]) <= N:
        return feats
    scores = feats["keypoint_scores"][0]
    _, best_indices = torch.topk(scores, N, largest=True, sorted=False)
    best_indices = best_indices.cpu().numpy()
    return prune_features(feats, best_indices), best_indices


def highest_score_matches(matches: dict, N: int) -> np.ndarray:
    if len(matches["matches"]) <= N:
        return matches["matches"]
    scores = matches["scores"]
    _, best_indices = torch.topk(scores, N, largest=True, sorted=False)
    best_matches = matches["matches"][best_indices]
    return best_matches


class FeatureTracker:
    def __init__(
        self,
        cam_config: dict,
        max_keypoints: int = EXTRACTOR_MAX_KEYPOINTS,
        feature_type: str = "superpoint_lightglue",
    ):
        assert feature_type in FEATURE_TYPES, f"Unknown feature type {feature_type}"
        self.cam_config = cam_config
        self.feature_type = feature_type
        if feature_type == "superpoint_lightglue":
            self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().cuda()
            self.matcher = LightGlue(features="superpoint").eval().cuda()
        elif feature_type == "orb":
            # Second detector instance for the right image of a stereo pair (extracted concurrently)
            self.extractor, self.extractor_right = (cv2.ORB_create(nfeatures=max_keypoints) for _ in range(2))
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            self.descriptor_dim, self.descriptor_dtype = 32, np.uint8
        elif feature_type == "sift":
            self.extractor, self.extractor_right = (cv2.SIFT_create(nfeatures=max_keypoints) for _ in range(2))
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
            self.descriptor_dim, self.descriptor_dtype = 128, np.float32
        if feature_type != "superpoint_lightglue":
            self.thread_pool = ThreadPoolExecutor(max_workers=1)

        # Currently default parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
        )

        # TODO: use a TrackerState dataclass instead and pass that around
        self.track_ids = None
        self.prev_image = None
        self.prev_pts = None
        self.prev_pts_right = None
        self.prev_feats = None
        self.world_points = None
        self.max_id = 0

    def extract_feats(
        self, image: np.ndarray, min_score: float = None, max_keypoints: int = None
    ) -> dict:
        """Extract features from image"""
        # TODO: handle min_score and max_keypoints
        if self.feature_type != "superpoint_lightglue":
            return self.extract_classical_feats(image)
        feats = self.extractor.extract(grayscale_to_3ch_tensor(image).cuda())
        return feats

    def extract_classical_feats(self, image: np.ndarray, extractor=None) -> dict:
        """ORB or SIFT features of a grayscale uint8 image, in the SuperPoint feature dict layout"""
        extractor = self.extractor if extractor is None else extractor
        keypoints, descriptors = extractor.detectAndCompute(image, None)
        if descriptors is None:  # no keypoints
            descriptors = np.zeros((0, self.descriptor_dim), dtype=self.descriptor_dtype)
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)
        height, width = image.shape[:2]
        return {
            "keypoints": torch.from_numpy(pts).unsqueeze(0).cuda(),
            "keypoint_scores": torch.from_numpy(scores).unsqueeze(0).cuda(),
            "descriptors": torch.from_numpy(descriptors).unsqueeze(0).cuda(),
            "image_size": torch.tensor([[width, height]], dtype=torch.float32).cuda(),
        }

    def match_classical_feats(self, feats1: dict, feats2: dict) -> dict:
        """Brute-force matching with the ratio test, in the LightGlue output layout (matches [M, 2], scores [M])"""
        desc1 = feats1["descriptors"][0].cpu().numpy()
        desc2 = feats2["descriptors"][0].cpu().numpy()
        pairs = []
        if len(desc1) > 0 and len(desc2) >= 2:
            for knn in self.matcher.knnMatch(desc1, desc2, k=2):
                best, second = knn
                if best.distance < CLASSICAL_RATIO_TEST * second.distance:
                    # second.distance > 0 here since best.distance >= 0
                    pairs.append((best.queryIdx, best.trainIdx, 1.0 - best.distance / second.distance))
        pairs = np.array(pairs, dtype=np.float64).reshape(-1, 3)
        device = feats1["keypoints"].device
        return {
            "matches": torch.from_numpy(pairs[:, :2].astype(np.int64)).to(device),
            "scores": torch.from_numpy(pairs[:, 2].astype(np.float32)).to(device),
        }

    def match_feats(
        self, feats1: dict, feats2: dict, max_matches: int = None, min_score: float = None
    ) -> torch.Tensor:
        """Match features between two images"""
        if self.feature_type != "superpoint_lightglue":
            matches = self.match_classical_feats(feats1, feats2)
            if max_matches is not None:
                return highest_score_matches(matches, max_matches)
            return matches["matches"]
        matches = rbd(self.matcher({"image0": feats1, "image1": feats2}))
        if min_score is not None:
            mask = matches["scores"] > min_score
            matches["matches"] = matches["matches"][mask]
            matches["scores"] = matches["scores"][mask]
        if max_matches is not None:
            return highest_score_matches(matches, max_matches)
        return matches["matches"]

    def process_stereo(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        max_matches: int = None,
        min_score: float = None,
        max_depth: float = MAX_DEPTH,
        return_matched_feats: bool = False,
    ) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
        """Process stereo pair to get features and depths"""
        if self.feature_type == "superpoint_lightglue":
            feats_left = self.extract_feats(left_image)
            feats_right = self.extract_feats(right_image)
        else:
            # OpenCV detection is CPU-bound and releases the GIL: the right image is extracted in a worker
            # thread (own detector instance) while the left one is extracted here
            future_right = self.thread_pool.submit(
                self.extract_classical_feats, right_image, self.extractor_right
            )
            feats_left = self.extract_classical_feats(left_image)
            feats_right = future_right.result()
        matches = self.match_feats(feats_left, feats_right, max_matches, min_score)
        matched_kps_left = feats_left["keypoints"][0][matches[:, 0]]
        matched_kps_right = feats_right["keypoints"][0][matches[:, 1]]
        disparities = matched_kps_left[:, 0] - matched_kps_right[:, 0]
        depths = FL_X * STEREO_BASELINE / (disparities + 1e-8)  # Avoid division by zero

        # Keep depths in (0, max_depth]. A disparity <= 0 (right keypoint not left of the left one) is a mismatch
        # and gives a negative (or, at 0, a huge) depth. Drop matches off the epipolar line (same row).
        row_diffs = torch.abs(matched_kps_left[:, 1] - matched_kps_right[:, 1])
        outliers = (depths <= 0) | (depths > max_depth) | (row_diffs > STEREO_MAX_ROW_DIFF)
        if torch.sum(outliers) > 0:
            matches = matches[~outliers]
            depths = depths[~outliers]

        if return_matched_feats:
            matched_feats_left = prune_features(feats_left, matches[:, 0])
            matched_feats_right = prune_features(feats_right, matches[:, 1])
            return matched_feats_left, matched_feats_right, matches, depths
        else:
            return feats_left, feats_right, matches, depths

    def project_stereo(
        self,
        pose: np.ndarray,
        pixels: np.ndarray | torch.Tensor,
        depths: np.ndarray | torch.Tensor,
        cam_name: str = "FrontLeft",
    ) -> np.ndarray:
        """Project stereo pixel-depth pairs to world points"""
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.cpu().numpy()
        if isinstance(depths, torch.Tensor):
            depths = depths.cpu().numpy()
        points_rover = project_pixels_to_rover(pixels, depths, cam_name, self.cam_config)
        points_world = apply_transform(pose, points_rover)
        return points_world

    def initialize(self, initial_pose: np.ndarray, left_image: np.ndarray, right_image: np.ndarray):
        """Initialize world points and features"""
        feats_left, feats_right, matches_stereo, depths = self.process_stereo(
            left_image, right_image, max_matches=MAX_STEREO_MATCHES, return_matched_feats=True
        )
        left_pts = feats_left["keypoints"][0]
        num_pts = len(left_pts)
        points_world = self.project_stereo(initial_pose, left_pts, depths)

        self.track_ids = np.arange(num_pts)  # 0, 1, 2, ..., num_pts - 1
        self.prev_image = left_image
        self.prev_pts = left_pts.cpu().numpy()
        self.prev_pts_right = feats_right["keypoints"][0].cpu().numpy()
        self.prev_feats = feats_left
        self.world_points = points_world
        self.max_id = num_pts  # Next ID to assign

    def track(self, next_image: np.ndarray):
        """Track keypoints using optical flow

        TODO: test RAFT for optical flow

        """
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_image, next_image, self.prev_pts, None, **self.lk_params
        )
        tracked = status.squeeze() == 1
        next_pts_tracked = next_pts[tracked]
        tracked_feats = prune_features(self.prev_feats, tracked)
        tracked_feats["keypoints"] = torch.from_numpy(next_pts_tracked).unsqueeze(0).cuda()

        self.track_ids = self.track_ids[tracked]
        self.prev_image = next_image
        self.prev_pts = next_pts_tracked
        self.prev_feats = tracked_feats
        self.world_points = self.world_points[tracked]

    def track_lightglue(self, next_image: np.ndarray):
        """Track keypoints using LightGlue"""
        # TODO
        next_feats = self.extract_feats(next_image)
        matches = self.match_feats(self.prev_feats, next_feats)
        tracked_feats = prune_features(self.prev_feats, matches[:, 0])

    def track_keyframe(
        self, curr_pose: np.ndarray, left_image: np.ndarray, right_image: np.ndarray
    ):
        """Initialize world points and features"""
        # Triangulate new points
        feats_left, feats_right, matches_stereo, depths = self.process_stereo(
            left_image,
            right_image,
            max_matches=MAX_STEREO_MATCHES,
            max_depth=10.0,
            return_matched_feats=True,
        )
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
