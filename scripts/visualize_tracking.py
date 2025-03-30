import cv2
import os
import numpy as np
import random
from pathlib import Path
from norfair import Detection, Tracker

from lac.perception.segmentation import UnetSegmentation
from lac.perception.depth import stereo_depth_from_segmentation
from lac.utils.visualization import (
    overlay_mask,
    overlay_points,
    int_to_color,
)
from lac.params import STEREO_BASELINE, FL_X


if __name__ == "__main__":
    data_path = "/home/shared/data_raw/LAC/segmentation/slam_map1_preset1_teleop"
    left_path = Path(data_path) / "FrontLeft"
    right_path = Path(data_path) / "FrontRight"

    segmentation = UnetSegmentation()
    tracker = Tracker(distance_function="euclidean", distance_threshold=100, hit_counter_max=5)

    for img_name in sorted(os.listdir(left_path)):
        left_img = cv2.imread(str(left_path / img_name), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(right_path / img_name), cv2.IMREAD_GRAYSCALE)

        left_seg_masks, left_seg_labels = segmentation.segment_rocks(left_img)
        right_seg_masks, right_seg_labels = segmentation.segment_rocks(right_img)
        left_seg_full_mask = np.clip(left_seg_labels, 0, 1)

        stereo_depth_results = stereo_depth_from_segmentation(
            left_seg_masks, right_seg_masks, STEREO_BASELINE, FL_X
        )

        left_centroids = [result["left_centroid"] for result in stereo_depth_results]
        detections = [Detection(p) for p in left_centroids]
        tracked_objects = tracker.update(detections)

        overlay = overlay_mask(left_img, left_seg_full_mask, color=(0, 0, 1))
        for obj in tracked_objects:
            color = int_to_color(obj.id)
            cv2.circle(
                overlay,
                tuple(obj.last_detection.points[0].astype(int)),
                5,
                color,
                -1,
            )
        overlay = overlay_points(overlay, left_centroids, size=3)

        cv2.imshow("img", overlay)
        cv2.waitKey(20)

    cv2.destroyAllWindows()
