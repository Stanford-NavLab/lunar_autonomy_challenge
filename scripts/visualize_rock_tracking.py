import cv2
import os
import numpy as np
import random
from pathlib import Path
from norfair import Detection, Tracker
from tqdm import tqdm

from lac.slam.rock_tracker import RockTracker
from lac.perception.segmentation import UnetSegmentation
from lac.perception.depth import stereo_depth_from_segmentation
from lac.utils.visualization import (
    overlay_mask,
    overlay_points,
    int_to_color,
)
from lac.util import load_data
from lac.params import STEREO_BASELINE, FL_X

MAX_N_KEYPOINTS = 10


if __name__ == "__main__":
    data_path = "/home/shared/data_raw/LAC/segmentation/slam_map1_preset1_teleop"
    initial_pose, lander_pose, poses, imu_data, cam_config = load_data(data_path)
    left_path = Path(data_path) / "FrontLeft"
    right_path = Path(data_path) / "FrontRight"

    segmentation = UnetSegmentation()
    tracker = Tracker(distance_function="euclidean", distance_threshold=100, hit_counter_max=5)
    rock_tracker = RockTracker(cam_config)

    frames = sorted(int(img_name.split(".")[0]) for img_name in os.listdir(left_path))

    END_FRAME = frames[-1]
    for frame in tqdm(range(2, END_FRAME, 2)):
        img_name = f"{frame:06}.png"
        left_img = cv2.imread(str(left_path / img_name), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(right_path / img_name), cv2.IMREAD_GRAYSCALE)

        # left_seg_masks, left_seg_labels = segmentation.segment_rocks(left_img)
        # right_seg_masks, right_seg_labels = segmentation.segment_rocks(right_img)
        # left_seg_full_mask = np.clip(left_seg_labels, 0, 1)

        # stereo_depth_results = stereo_depth_from_segmentation(
        #     left_seg_masks, right_seg_masks, STEREO_BASELINE, FL_X
        # )

        # left_centroids = [result["left_centroid"] for result in stereo_depth_results]
        # detections = [Detection(p) for p in left_centroids]
        # tracked_objects = tracker.update(detections)
        rock_detections, left_seg_full_mask = rock_tracker.detect_rocks(left_img, right_img)
        tracker_detections = []
        all_left_keypoints = []
        for detection in rock_detections.values():
            keypoints = detection["left_keypoints"].cpu().numpy()
            centroid = np.mean(keypoints, axis=0)
            tracker_detections.append(Detection(centroid))
            all_left_keypoints.append(keypoints)
        # detections = [Detection(p) for p in rock_detections["left_keypoints"].cpu().numpy()]
        tracked_objects = tracker.update(tracker_detections)

        overlay = overlay_mask(left_img, left_seg_full_mask, color=(0, 0, 1))
        for obj in tracked_objects:
            color = int_to_color(obj.id)
            overlay = overlay_points(overlay, obj.last_detection.points, color=color, size=3)
            # cv2.circle(
            #     overlay,
            #     tuple(obj.last_detection.points[0].astype(int)),
            #     7,
            #     color,
            #     -1,
            # )
        if len(all_left_keypoints) > 0:
            all_left_keypoints = np.vstack(all_left_keypoints)
            # remove nans
            all_left_keypoints = all_left_keypoints[~np.isnan(all_left_keypoints).any(axis=1)]
            overlay = overlay_points(overlay, all_left_keypoints, size=3)

        cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
        cv2.imshow("Left", overlay)
        cv2.waitKey(20)

    cv2.destroyAllWindows()
