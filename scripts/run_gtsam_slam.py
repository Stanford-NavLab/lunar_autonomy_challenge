"""Run GTSAM SLAM on logged run."""

import numpy as np
from tqdm import tqdm
from pathlib import Path
from gtsam.symbol_shorthand import X

from lac.localization.gtsam_factor_graph import GtsamFactorGraph
from lac.localization.slam.feature_tracker import FeatureTracker
from lac.utils.plotting import plot_poses
from lac.util import load_data, load_stereo_images
from lac.params import LAC_BASE_PATH

if __name__ == "__main__":
    # Load the data logs
    data_path = Path(LAC_BASE_PATH) / "output/DataCollectionAgent/map1_preset0_stereo_lights1.0"
    initial_pose, lander_pose, poses, imu_data, cam_config = load_data(data_path)

    # Load the images
    left_imgs, right_imgs = load_stereo_images(data_path)

    START_FRAME = 2000

    # Initialize tracker and graph
    tracker = FeatureTracker(cam_config)
    initial_pose = poses[START_FRAME]
    tracker.initialize(poses[START_FRAME], left_imgs[START_FRAME], right_imgs[START_FRAME])

    graph = GtsamFactorGraph()

    N = 200
    UPDATE_RATE = 10

    idx = 0
    curr_pose = initial_pose
    graph.add_pose(idx, initial_pose)
    graph.add_vision_factors(idx, tracker.world_points, tracker.prev_pts, tracker.track_ids)

    # i is step which is 0 for initial and starts at 1 for the first run_step call
    for i in tqdm(range(2, N)):
        step = i + START_FRAME

        # Run tracker
        if i % 2 == 0:
            if i % 10 == 0:
                tracker.track_keyframe(poses[step], left_imgs[step], right_imgs[step])
            else:
                tracker.track(left_imgs[step])

        # Add new pose and vision factors to graph
        if i % UPDATE_RATE == 0:
            idx += 1
            noisy_pose = poses[step].copy()
            noisy_pose[:3, 3] += np.random.normal(0, 0.0, 3)
            graph.add_pose(idx, noisy_pose)
            graph.add_vision_factors(idx, tracker.world_points, tracker.prev_pts, tracker.track_ids)

    # Optimize
    result = graph.optimize()
    print("initial error = {}".format(graph.graph.error(graph.initial_estimate)))
    print("final error = {}".format(graph.graph.error(result)))

    initial_poses = []
    result_poses = []

    for i in range(idx):
        initial_poses.append(graph.initial_estimate.atPose3(X(i)).matrix())
        result_poses.append(result.atPose3(X(i)).matrix())

    fig = plot_poses(poses[: N + START_FRAME], no_axes=True, color="black", name="Ground truth")
    fig = plot_poses(
        initial_poses, fig=fig, no_axes=True, color="orange", name="GTSAM initial poses"
    )
    fig = plot_poses(
        result_poses, fig=fig, no_axes=True, color="green", name="GTSAM optimized poses"
    )
    fig.update_layout(height=900, width=1600, scene_aspectmode="data")
    fig.show()
