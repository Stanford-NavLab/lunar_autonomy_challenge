"""Run Stereo PnP VO on a logged run."""

import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import cv2

from lac.slam.visual_odometry import StereoVisualOdometry
from lac.utils.plotting import plot_poses
from lac.util import load_data, positions_rmse_from_poses

if __name__ == "__main__":
    # Load the data logs
    data_path = "/home/shared/data_raw/LAC/runs/full_spiral_map1_preset1_recovery_agent"
    # data_path = "/home/shared/data_raw/LAC/runs/full_spiral_map1_preset0"
    initial_pose, lander_pose, poses, imu_data, cam_config = load_data(data_path)
    left_path = Path(data_path) / "FrontLeft"
    right_path = Path(data_path) / "FrontRight"
    frames = sorted(int(img_name.split(".")[0]) for img_name in os.listdir(left_path))

    svo = StereoVisualOdometry(cam_config)
    svo_poses = []
    eval_poses = []

    START_FRAME = 250
    END_FRAME = 2000
    # END_FRAME = len(frames) - 1

    print("Running VO...")
    progress_bar = tqdm(range(START_FRAME, END_FRAME, 2), dynamic_ncols=True)

    for frame in progress_bar:
        progress_bar.set_description(f"Processing Frame: {frame}")

        img_name = f"{frame:06}.png"
        left_img = cv2.imread(str(left_path / img_name), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(right_path / img_name), cv2.IMREAD_GRAYSCALE)
        eval_poses.append(poses[frame])

        if frame == START_FRAME:
            svo.initialize(poses[frame], left_img, right_img)
            svo_poses.append(poses[frame])
            continue

        # svo.track(left_img, right_img)
        svo.track_old(left_img, right_img)
        svo_poses.append(svo.rover_pose)

    print(f"RMSE: {positions_rmse_from_poses(eval_poses, svo_poses)}")
    # np.save(
    #     os.path.join(data_path, "svo_poses.npy"),
    #     np.array(svo_poses),
    # )

    fig = plot_poses(eval_poses, no_axes=True, color="black", name="Ground truth")
    fig = plot_poses(svo_poses, fig=fig, no_axes=True, color="orange", name="VO")
    fig.update_layout(height=900, width=1600, scene_aspectmode="data")
    fig.write_html(os.path.join(data_path, "svo_poses_old.html"))
    fig.show()
