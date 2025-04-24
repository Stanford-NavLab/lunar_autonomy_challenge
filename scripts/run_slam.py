"""Run SLAM on logged run."""

import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

from lac.slam.semantic_feature_tracker import SemanticFeatureTracker
from lac.slam.frontend import Frontend
from lac.slam.backend import Backend
from lac.utils.plotting import plot_poses, plot_loop_closures
from lac.util import load_data, positions_rmse_from_poses
from lac.params import LAC_BASE_PATH

if __name__ == "__main__":
    # Load the data logs
    data_path = Path(LAC_BASE_PATH) / "output/DataCollectionAgent/full_spiral_preset0_loop_closure"
    initial_pose, lander_pose, poses, imu_data, cam_config = load_data(data_path)
    front_left_path = Path(data_path) / "FrontLeft"
    front_right_path = Path(data_path) / "FrontRight"
    frames = sorted(int(img_name.split(".")[0]) for img_name in os.listdir(front_left_path))

    START_FRAME = 80
    # END_FRAME = len(poses) - 1
    END_FRAME = 56000

    feature_tracker = SemanticFeatureTracker(cam_config)
    frontend = Frontend(feature_tracker)
    backend = Backend(poses[START_FRAME], feature_tracker)

    eval_poses = []

    print("Running SLAM...")
    progress_bar = tqdm(range(START_FRAME, END_FRAME, 2), dynamic_ncols=True)

    for frame in progress_bar:
        progress_bar.set_description(f"Processing Frame: {frame}")

        img_name = f"{frame:06}.png"
        left_img = cv2.imread(str(front_left_path / img_name), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(front_right_path / img_name), cv2.IMREAD_GRAYSCALE)
        eval_poses.append(poses[frame])

        if frame == START_FRAME:
            frontend.initialize(left_img, right_img)
            continue

        data = {
            "step": frame,
            "left_image": left_img,
            "right_image": right_img,
            "imu": imu_data[frame],
        }

        data = frontend.process_frame(data)
        backend.update(data)

        current_pose = backend.get_trajectory()[-1]
        current_error = np.linalg.norm(current_pose[:3, 3] - poses[frame][:3, 3])
        progress_bar.set_postfix({"error": f"{current_error:.3f}"})

    trajectory = backend.get_trajectory()
    print(f"RMSE: {positions_rmse_from_poses(eval_poses, trajectory)}")

    fig = plot_poses(eval_poses, no_axes=True, color="black", name="Ground truth")
    fig = plot_poses(trajectory, fig=fig, no_axes=True, color="orange", name="Backend poses")
    fig = plot_loop_closures(trajectory, backend.loop_closures, fig=fig)
    fig.update_layout(height=900, width=1600, scene_aspectmode="data")
    fig.write_html(Path(data_path) / "slam.html")
    fig.show()
