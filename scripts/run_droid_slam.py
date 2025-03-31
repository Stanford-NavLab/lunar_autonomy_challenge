"""Run Stereo PnP VO on a logged run."""

import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import cv2
import torch
import glob
import argparse

from droid_slam.droid import Droid
import droid_backends

from lac.utils.plotting import plot_poses
from lac.util import load_data, load_stereo_images

if __name__ == "__main__":
    # Load the data logs
    data_path = "/home/shared/data_raw/LAC/runs/full_spiral_map1_preset0_recovery_agent"
    initial_pose, lander_pose, poses, imu_data, cam_config = load_data(data_path)
    left_path = Path(data_path) / "FrontLeft"
    right_path = Path(data_path) / "FrontRight"
    frames = sorted(int(img_name.split(".")[0]) for img_name in os.listdir(left_path))

    images_left = sorted(glob.glob(os.path.join(data_path, "FrontLeft/*.png")))
    images_right = sorted(glob.glob(os.path.join(data_path, "FrontRight/*.png")))

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="data/LAC")
    parser.add_argument("--weights", default="/home/lac/opt/DROID-SLAM/droid.pth")
    parser.add_argument("--buffer", type=int, default=2500)

    # Parameters
    DOWNSCALE_FACTOR = 0.5
    image_size = [270, 480]
    intrinsics_vec = [914.0152, 914.0152, 640.0, 360.0]
    STRIDE = 2

    START_FRAME = 30000
    END_FRAME = 33000

    print("Running VO...")
    for frame in tqdm(range(START_FRAME, END_FRAME, 2)):
        tqdm.write(f"Frame: {frame}")
        img_name = f"{frame:06}.png"
        left_img = cv2.imread(str(left_path / img_name), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(right_path / img_name), cv2.IMREAD_GRAYSCALE)
        left_img = cv2.resize(left_img, (image_size[1], image_size[0]))
        right_img = cv2.resize(right_img, (image_size[1], image_size[0]))

        images = [left_img, right_img]
        images = torch.from_numpy(np.stack(images, 0)).permute(0, 3, 1, 2)
        intrinsics = 0.5 * 0.75 * torch.as_tensor(intrinsics_vec)

        if frame == START_FRAME:
            svo.initialize(poses[frame], left_img, right_img)
            svo_poses.append(poses[frame])
            continue

        svo.track(left_img, right_img)
        svo_poses.append(svo.rover_pose)

    fig = plot_poses(poses[START_FRAME:END_FRAME], no_axes=True, color="black", name="Ground truth")
    fig = plot_poses(svo_poses, fig=fig, no_axes=True, color="orange", name="VO")
    fig.update_layout(height=900, width=1600, scene_aspectmode="data")
    fig.show()
