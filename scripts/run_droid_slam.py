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

import warnings

warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument",
)

from droid_slam.droid import Droid
import droid_backends

from lac.utils.plotting import plot_poses
from lac.util import load_data, load_stereo_images
from lac.params import LAC_BASE_PATH

if __name__ == "__main__":
    # Load the data logs
    # data_path = "/home/shared/data_raw/LAC/runs/full_spiral_map1_preset0_recovery_agent"
    data_path = "/home/shared/data_raw/LAC/runs/full_spiral_map1_preset0"
    initial_pose, lander_pose, poses, imu_data, cam_config = load_data(data_path)
    left_path = Path(data_path) / "FrontLeft"
    right_path = Path(data_path) / "FrontRight"
    frames = sorted(int(img_name.split(".")[0]) for img_name in os.listdir(left_path))

    # Parameters
    DOWNSCALE_FACTOR = 0.4  # works with [0.3, 0.5]
    IMAGE_SIZE = (DOWNSCALE_FACTOR * np.array([720, 1280])).astype(int)
    INTRINSICS = torch.as_tensor(DOWNSCALE_FACTOR * np.array([914.0152, 914.0152, 640.0, 360.0]))
    BUFFER = 2000
    STRIDE = 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="/home/lac/opt/DROID-SLAM/droid.pth")
    parser.add_argument("--buffer", type=int, default=BUFFER)
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--image_size", default=IMAGE_SIZE.tolist())

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=4.0)
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
    END_FRAME = frames[-1]

    torch.cuda.empty_cache()
    droid = Droid(args)
    stream = []

    print("Running tracking...")
    start_time = time.time()

    for t, frame in enumerate(tqdm(range(START_FRAME, END_FRAME, 2 * STRIDE))):
        img_name = f"{frame:06}.png"
        left_img = cv2.imread(str(left_path / img_name))
        right_img = cv2.imread(str(right_path / img_name))
        left_img = cv2.resize(left_img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        right_img = cv2.resize(right_img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

        images = [left_img, right_img]
        images = torch.from_numpy(np.stack(images, 0)).permute(0, 3, 1, 2)

        droid.track(t, images, intrinsics=INTRINSICS)
        stream.append((t, images, INTRINSICS))

    print("Tracking ran {} frames in {} seconds".format(len(stream), time.time() - start_time))

    print("Running BA...")
    start_time = time.time()
    traj_est = droid.terminate(stream)
    print("BA ran in {} seconds".format(time.time() - start_time))

    np.savez(
        Path(data_path) / "droid.npz",
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        stride=STRIDE,
        downscale_factor=DOWNSCALE_FACTOR,
        trajectory=traj_est,
    )
    print("Trajectory saved")
