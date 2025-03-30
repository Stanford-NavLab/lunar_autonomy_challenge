"""Rock detector and tracker"""

import numpy as np
import cv2
import torch

from lac.slam.feature_tracker import FeatureTracker, prune_features
from lac.perception.segmentation import UnetSegmentation, SemanticClasses
from lac.perception.segmentation_util import dilate_mask

from lac.perception.depth import project_pixels_to_rover
from lac.utils.frames import (
    apply_transform,
)
from lac.params import FL_X, STEREO_BASELINE


class RockTracker:
    def __init__(self, cam_config: dict):
        self.segmentation = UnetSegmentation()
        self.tracker = FeatureTracker(cam_config)

    def detect_rocks(self, left_image: np.ndarray, right_image: np.ndarray):
        """Process stereo pair to get features and depths"""
        # Segmentation
        left_masks, left_labels = self.segmentation.segment_rocks(left_image)
        right_masks, right_labels = self.segmentation.segment_rocks(right_image)
        left_full_mask = np.clip(left_labels, 0, 1).astype(np.uint8)
        right_full_mask = np.clip(right_labels, 0, 1).astype(np.uint8)

        # Feature matching
        left_feats, right_feats, matches, depths = self.tracker.process_stereo(
            left_image, right_image
        )

        left_matched_feats = prune_features(left_feats, matches[:, 0])
        left_matched_pts = left_matched_feats["keypoints"][0]
        right_matched_feats = prune_features(right_feats, matches[:, 1])
        right_matched_pts = right_matched_feats["keypoints"][0]

        # Find matching points within segmentations
        # kernel = np.ones((5, 5), np.uint8)
        left_full_mask_dilated = dilate_mask(left_full_mask, pixels=2)
        right_full_mask_dilated = dilate_mask(right_full_mask, pixels=2)

        # rock_pt_idxs = []
        rock_pt_idxs = {}
        MAX_DEPTH = 5.0

        # for i in range(len(left_matched_pts)):
        #     x_left, y_left = left_matched_pts[i]
        #     x_right, y_right = right_matched_pts[i]
        #     if (
        #         left_full_mask_dilated[int(y_left), int(x_left)]
        #         and right_full_mask_dilated[int(y_right), int(x_right)]
        #     ):
        #         rock_pt_idxs.append(i)
        for i in range(len(left_matched_pts)):
            if depths[i] > MAX_DEPTH:
                continue
            x_left, y_left = left_matched_pts[i]
            x_right, y_right = right_matched_pts[i]
            id = left_labels[int(y_left), int(x_left)]
            if (
                left_full_mask_dilated[int(y_left), int(x_left)] != 0
                and right_full_mask_dilated[int(y_right), int(x_right)] != 0
            ):
                if id not in rock_pt_idxs:
                    rock_pt_idxs[id] = []
                rock_pt_idxs[id].append(i)

        rock_detections = {}

        for i, (id, idxs) in enumerate(rock_pt_idxs.items()):
            rock_detections[i] = {
                "left_keypoints": left_matched_pts[idxs],
                "right_keypoints": right_matched_pts[idxs],
                "depths": depths[idxs],
            }

        # left_rock_matched_pts = left_matched_pts[rock_pt_idxs]
        # right_rock_matched_pts = right_matched_pts[rock_pt_idxs]
        # depths_rock_matched = depths[rock_pt_idxs]

        # rock_points = self.tracker.project_stereo(pose, left_rock_matched_pts, depths_rock_matched)

        return rock_detections, left_full_mask
