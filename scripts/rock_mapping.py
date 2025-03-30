import cv2
import os
import numpy as np
import random
from pathlib import Path
from norfair import Detection, Tracker
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors
import plotly.graph_objects as go

from lac.perception.segmentation import UnetSegmentation
from lac.perception.depth import stereo_depth_from_segmentation, project_pixel_to_world
from lac.utils.visualization import (
    overlay_mask,
    overlay_points,
    int_to_color,
)
from lac.utils.plotting import plot_rock_map, plot_poses, plot_3d_points
from lac.util import load_data
from lac.params import STEREO_BASELINE, FL_X


VIZUALIZE = True


if __name__ == "__main__":
    data_path = "/home/shared/data_raw/LAC/segmentation/slam_map1_preset1_teleop"
    initial_pose, lander_pose, poses, imu_data, cam_config = load_data(data_path)
    left_path = Path(data_path) / "FrontLeft"
    right_path = Path(data_path) / "FrontRight"

    map = np.load(
        "/home/shared/data_raw/LAC/heightmaps/competition/Moon_Map_01_preset_1.dat",
        allow_pickle=True,
    )

    segmentation = UnetSegmentation()
    tracker = Tracker(distance_function="euclidean", distance_threshold=100, hit_counter_max=5)

    rock_detections = {}
    frames = sorted(int(img_name.split(".")[0]) for img_name in os.listdir(left_path))

    # for frame in tqdm(frames):
    END_FRAME = frames[-1]
    for frame in tqdm(range(2, END_FRAME, 2)):
        img_name = f"{frame:06}.png"

        left_img = cv2.imread(str(left_path / img_name), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(right_path / img_name), cv2.IMREAD_GRAYSCALE)

        left_seg_masks, left_seg_labels = segmentation.segment_rocks(left_img)
        right_seg_masks, right_seg_labels = segmentation.segment_rocks(right_img)
        left_seg_full_mask = np.clip(left_seg_labels, 0, 1)

        stereo_depth_results = stereo_depth_from_segmentation(
            left_seg_masks, right_seg_masks, STEREO_BASELINE, FL_X
        )

        detections = []
        centroids = []
        for result in stereo_depth_results:
            centroid = result["left_centroid"]
            depth = result["depth"]
            if depth < 5.0:
                rock_point_world_frame = project_pixel_to_world(
                    poses[frame], centroid, result["depth"], "FrontLeft", cam_config
                )
                centroids.append(centroid)
                detections.append(Detection(points=centroid, data=rock_point_world_frame))
        tracked_objects = tracker.update(detections)

        for rock in tracked_objects:
            centroid_pixel = rock.last_detection.points[0]
            if rock.id not in rock_detections:
                rock_detections[rock.id] = []
            rock_detections[rock.id].append(rock.last_detection.data)

        if VIZUALIZE:
            overlay = overlay_mask(left_img, left_seg_full_mask, color=(0, 0, 1))
            for obj in tracked_objects:
                color = int_to_color(obj.id)
                cv2.circle(
                    overlay,
                    tuple(obj.last_detection.points[0].astype(int)),
                    7,
                    color,
                    -1,
                )
            overlay = overlay_points(overlay, centroids, size=3)

            cv2.imshow("img", overlay)
            cv2.waitKey(20)

    if VIZUALIZE:
        cv2.destroyAllWindows()

    fig = go.Figure()
    fig = plot_rock_map(map, fig=fig)
    fig = plot_poses(poses[:END_FRAME], fig=fig, no_axes=True, color="black")
    for id, points in rock_detections.items():
        points = np.array(points)
        fig = plot_3d_points(points, fig=fig, color=int_to_color(id, hex=True), name=f"rock_{id}")
    fig.show()
    fig.write_html("rock_mapping.html")
