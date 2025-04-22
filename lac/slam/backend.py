"""Backend for SLAM"""

import numpy as np
import gtsam
from gtsam.symbol_shorthand import X

from lac.slam.gtsam_util import ODOMETRY_NOISE


class Backend:
    """Backend for SLAM"""

    def __init__(self, initial_pose: np.ndarray):
        """Initialize the backend"""
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.opt_params = gtsam.LevenbergMarquardtParams()

        self.values.insert(X(0), gtsam.Pose3(initial_pose))
        self.graph.add(gtsam.NonlinearEqualityPose3(X(0), gtsam.Pose3(initial_pose)))
        self.pose_idx = 1  # Index for the next pose

        self.point_map = {}  # id -> ((x,y,z), anchor_pose_idx)
        self.keyframes = {}  # store data at keyframes (for loop closure)

    def update(self, data: dict):
        """Update the backend with new data

        data : dict
            - left_image : np.ndarray (H, W, 3) - left image
            - right_image : np.ndarray (H, W, 3) - right image
            - imu : np.ndarray (6,) - IMU measurement
            - odometry : np.ndarray (4, 4) - estimated odometry
            - keyframe : bool - whether this is a keyframe

        """
        # Add new pose
        last_pose = self.values.atPose3(X(self.pose_idx - 1)).matrix()
        new_pose = last_pose @ data["odometry"]  # New pose initialization from odometry
        self.values.insert(X(self.pose_idx), gtsam.Pose3(new_pose))

        # Add odometry factor
        self.graph.add(
            gtsam.BetweenFactorPose3(
                X(self.pose_idx - 1),
                X(self.pose_idx),
                gtsam.Pose3(data["odometry"]),
                ODOMETRY_NOISE,
            )
        )

        if data["keyframe"]:
            self.keyframes[self.pose_idx] = data

        # TODO: Proximity check for loop closure (optionally only do this if keyframe)

        self.pose_idx += 1

    def optimize(self):
        """Optimize the graph"""
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, self.opt_params)
        result = optimizer.optimize()

        # TODO: do checks (convergence, etc.)

        # Update the values with the optimized result
        self.values = result

    def get_trajectory(self):
        """Get the trajectory as a list of poses"""
        return [self.values.atPose3(X(i)).matrix() for i in range(self.pose_idx)]
