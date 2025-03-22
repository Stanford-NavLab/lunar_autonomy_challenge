"""Hybrid Optical Flow and LightGlue feature tracker

Uses stereo LightGlue matching at each keyframe (every 10 frames) to extract keypoints + descriptors
and triangulate points. In between, uses Lucas-Kanade optical flow to track points.

For now, we only track points in the left image.

"""

import numpy as np
import cv2

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
from lac.params import FL_X, STEREO_BASELINE


# def prune_features(feats, indices):
#     pruned_feats = {}
#     for k, v in feats.items():
#         if k == 'image_size':
#             pruned_feats[k] = v
#         else:
#             pruned_feats[k] = v[0, indices].unsqueeze(0)
#     return pruned_feats


def prune_features(feats, indices):
    return {k: v if k == "image_size" else v[0, indices].unsqueeze(0) for k, v in feats.items()}


class FeatureTracker:
    def __init__(self):
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        self.matcher = LightGlue(features="superpoint").eval().cuda()

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.prev_image = None
        self.prev_keypoints = None
        self.prev_feats = None
        self.world_points = None

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
        matched_feats = prune_features(feats0_left, matches0_stereo[..., 0])

        matched_kps0_left = matched_kps0_left.cpu().numpy()
        depths0 = depths0.cpu().numpy()

        points0_rover = []
        for pixel, depth in zip(matched_kps0_left, depths0):
            point_rover = project_pixel_to_rover(pixel, depth, "FrontLeft", self.cam_config)
            points0_rover.append(point_rover)
        points0_rover = np.array(points0_rover)
        points0_world = apply_transform(initial_pose, points0_rover)

        self.prev_feats = matched_feats
        self.prev_keypoints = matched_kps0_left
        self.prev_image = left_image
        self.world_points = points0_world

    def track(self, next_image: np.ndarray):
        pass

    def track_keyframe():
        pass
