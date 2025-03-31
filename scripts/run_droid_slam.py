"""Run Stereo PnP VO on a logged run."""

import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import cv2
import torch
import glob
import argparse
import time

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

    # Parameters
    DOWNSCALE_FACTOR = 0.5
    IMAGE_SIZE = (DOWNSCALE_FACTOR * np.array([720, 1280])).astype(int)
    INTRINSICS = DOWNSCALE_FACTOR * np.array([914.0152, 914.0152, 640.0, 360.0])
    STRIDE = 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="/home/lac/opt/DROID-SLAM/droid.pth")
    parser.add_argument("--buffer", type=int, default=2500)
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--image_size", default=IMAGE_SIZE.tolist())

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=15)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")

    args = parser.parse_args(["--disable_vis"])
    args.stereo = True

    START_FRAME = 80
    END_FRAME = 16000

    torch.cuda.empty_cache()
    droid = Droid(args)
    stream = []

    print("Running tracking...")
    start_time = time.time()

    for t, frame in tqdm(enumerate(range(START_FRAME, END_FRAME, 2 * STRIDE))):
        img_name = f"{frame:06}.png"
        left_img = cv2.imread(str(left_path / img_name))
        right_img = cv2.imread(str(right_path / img_name))
        left_img = cv2.resize(left_img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        right_img = cv2.resize(right_img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

        images = [left_img, right_img]
        images = torch.from_numpy(np.stack(images, 0)).permute(0, 3, 1, 2)

        print(f"t: {t}, images: {images}, intrinsics: {INTRINSICS}")

        droid.track(t, images, intrinsics=INTRINSICS)
        stream.append((t, images, INTRINSICS))

    print("Tracking ran {} frames in {} seconds".format(len(stream), time.time() - start_time))

    print("Running BA...")
    start_time = time.time()
    traj_est = droid.terminate(stream)
    print("BA ran in {} seconds".format(time.time() - start_time))

    np.savez("droid_traj_.npy", traj_est)

    fig = plot_poses(poses[START_FRAME:END_FRAME], no_axes=True, color="black", name="Ground truth")
    fig = plot_poses(traj_est, fig=fig, no_axes=True, color="orange", name="Droid SLAM")
    fig.update_layout(height=900, width=1600, scene_aspectmode="data")
    fig.show()
