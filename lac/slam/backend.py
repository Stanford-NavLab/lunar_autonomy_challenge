"""Backend for SLAM"""

import numpy as np
import gtsam
from gtsam.symbol_shorthand import X

from lac.slam.gtsam_util import ODOMETRY_NOISE


class Backend:
    """Backend for SLAM"""

    def __init__(self, camera_config: dict):
        """Initialize the backend"""
        self.camera_config = camera_config

        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.opt_params = gtsam.LevenbergMarquardtParams()
        self.pose_idx = 0

        self.point_map = {}  # id -> ((x,y,z), anchor_pose_idx)
        self.keyframes = {}  # store images at keyframes (for loop closure)

    def initialize(self, initial_pose: np.ndarray):
        """Initialize the backend"""
        self.values.insert(X(0), gtsam.Pose3(initial_pose))
        self.graph.add(gtsam.NonlinearEqualityPose3(X(0), gtsam.Pose3(initial_pose)))
        self.pose_idx += 1

    def update(self, input):
        """Update the backend with new data"""
        # Add new pose
        last_pose = np.eye(4)  # TODO: get this
        new_pose = last_pose @ input["odometry"]  # New pose initialization from odometry
        self.values.insert(X(self.pose_idx), gtsam.Pose3(new_pose))

        # Add odometry factor
        self.graph.add(
            gtsam.BetweenFactorPose3(
                X(self.pose_idx - 1),
                X(self.pose_idx),
                gtsam.Pose3(input["odometry"]),
                ODOMETRY_NOISE,
            )
        )

        if input["keyframe"]:
            self.keyframes[self.pose_idx] = input["keyframe"]

        # Proximity check for loop closure (optionally only do this if keyframe)

    def optimize(self):
        """Optimize the graph"""
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, self.opt_params)
        result = optimizer.optimize()

        # TODO: do checks (convergence, etc.)

        # Update the values with the optimized result
        self.values = result
