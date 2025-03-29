"""Run Stereo PnP VO on a logged run."""

import numpy as np
from tqdm import tqdm

from lac.slam.visual_odometry import StereoVisualOdometry
from lac.utils.plotting import plot_poses
from lac.util import load_data, load_stereo_images

if __name__ == "__main__":
    # Load the data logs
    data_path = "/home/shared/data_raw/LAC/segmentation/semantics_map1_preset1"
    initial_pose, lander_pose, poses, imu_data, cam_config = load_data(data_path)

    # Load the images
    print("Loading stereo images...")
    left_imgs, right_imgs = load_stereo_images(data_path)
    img_idxs = sorted(left_imgs.keys())

    svo = StereoVisualOdometry(cam_config)
    START_FRAME = 80  # After the arms have been raised
    svo.initialize(initial_pose, left_imgs[START_FRAME], right_imgs[START_FRAME])

    svo_poses = [initial_pose]

    END_FRAME = img_idxs[-1]

    print("Running VO...")
    for idx in tqdm(np.arange(START_FRAME + 2, END_FRAME, 2)):
        svo.track(left_imgs[idx], right_imgs[idx])
        svo_poses.append(svo.rover_pose)

    fig = plot_poses(poses[:END_FRAME], no_axes=True, color="black", name="Ground truth")
    fig = plot_poses(svo_poses, fig=fig, no_axes=True, color="orange", name="VO")
    fig.update_layout(height=900, width=1600, scene_aspectmode="data")
    fig.show()
