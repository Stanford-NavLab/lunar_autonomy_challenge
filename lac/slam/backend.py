"""Backend for SLAM"""

import numpy as np
import gtsam
from gtsam.symbol_shorthand import X
from dataclasses import dataclass

from lac.slam.semantic_feature_tracker import SemanticFeatureTracker, TrackedPoints
from lac.slam.loop_closure import keyframe_estimate_loop_closure_pose
from lac.slam.gtsam_util import VO_ODOMETRY_NOISE, IMU_ODOMETRY_NOISE, LOOP_CLOSURE_NOISE
from lac.utils.frames import apply_transform
from lac.util import rotation_matrix_error

LOOP_CLOSURE_EXCLUDE = 10  # Exclude the last N keyframes
LOOP_CLOSURE_DIST_THRESHOLD = 1.0  # meters
LOOP_CLOSURE_ANGLE_THRESHOLD = 5.0  # degrees


@dataclass
class SemanticPointCloud:
    points: np.ndarray  # shape (N, 3)
    labels: np.ndarray  # shape (N,)

    def save(self, filename: str):
        np.savez_compressed(filename, points=self.points, labels=self.labels)

    @classmethod
    def from_file(cls, filename: str) -> "SemanticPointCloud":
        data = np.load(filename)
        return cls(points=data["points"], labels=data["labels"])


class Backend:
    """Backend for SLAM"""

    def __init__(
        self, initial_pose: np.ndarray, feature_tracker: SemanticFeatureTracker, config: dict
    ):
        """Initialize the backend"""
        self.feature_tracker = feature_tracker

        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.opt_params = gtsam.LevenbergMarquardtParams()

        # NOTE: should we run IMU odometry during arm raise and initialize with that pose instead?
        self.values.insert(X(0), gtsam.Pose3(initial_pose))
        self.graph.add(gtsam.NonlinearEqualityPose3(X(0), gtsam.Pose3(initial_pose)))
        self.pose_idx = 1  # Index for the next pose

        self.point_map = {}  # anchor_pose_idx -> (local points, labels)
        self.keyframe_data = {}  # store data at keyframes (for loop closure)
        self.keyframe_traj_list = []
        self.keyframe_traj = None
        self.loop_closures = []

        # Config parameters
        self.LC_MIN_SCORE = config["min_score"]
        self.LC_MIN_MATCHES = config["min_matches"]
        self.LC_DIST_THRESHOLD = config["distance_threshold_m"]
        self.LC_ANGLE_THRESHOLD = config["angle_threshold_deg"]

        # TODO: log these for debugging
        self.pose_idx_map = {}
        self.odometry = []
        self.odometry_sources = []
        self.loop_closures_poses = []
        self.loop_closures_matches = []

    def update(self, data: dict):
        """Update the backend with new data

        data : dict
            - FrontLeft : np.ndarray (H, W, 3) - left image
            - FrontRight : np.ndarray (H, W, 3) - right image
            - imu : np.ndarray (6,) - IMU measurement
            - odometry : np.ndarray (4, 4) - estimated odometry
            - tracked_points : TrackedPoints - tracked points
            - keyframe : bool - whether this is a keyframe

        """
        # Add new pose
        last_pose = self.values.atPose3(X(self.pose_idx - 1)).matrix()
        new_pose = last_pose @ data["odometry"]  # New pose initialization from odometry
        self.values.insert(X(self.pose_idx), gtsam.Pose3(new_pose))

        # Add odometry factor
        if data["odometry_source"] == "VO":
            odometry_noise = VO_ODOMETRY_NOISE
        elif data["odometry_source"] == "IMU":
            odometry_noise = IMU_ODOMETRY_NOISE
        self.graph.add(
            gtsam.BetweenFactorPose3(
                X(self.pose_idx - 1),
                X(self.pose_idx),
                gtsam.Pose3(data["odometry"]),
                odometry_noise,
            )
        )

        # NOTE: logging
        self.odometry.append(data["odometry"])
        self.odometry_sources.append(0 if data["odometry_source"] == "VO" else 1)
        self.pose_idx_map[self.pose_idx] = data["step"]

        # Add tracked points to map
        tracks: TrackedPoints = data["tracked_points"]
        if "back_tracked_points" in data:
            back_tracks: TrackedPoints = data["back_tracked_points"]
            # Stack front and back points
            self.point_map[self.pose_idx] = {
                "points": np.vstack(
                    [
                        tracks.points_local[tracks.lengths == 0],
                        back_tracks.points_local[back_tracks.lengths == 0],
                    ]
                ),
                "labels": np.hstack(
                    [
                        tracks.labels[tracks.lengths == 0],
                        back_tracks.labels[back_tracks.lengths == 0],
                    ]
                ),
            }

        else:
            self.point_map[self.pose_idx] = {
                "points": tracks.points_local[tracks.lengths == 0],
                "labels": tracks.labels[tracks.lengths == 0],
            }

        # Handle keyframe
        if data["keyframe"]:
            self.keyframe_data[self.pose_idx] = self.feature_tracker.process_stereo(
                data["FrontLeft"], data["FrontRight"]
            )
            self.keyframe_traj_list.append(new_pose)
            self.keyframe_traj = np.array(self.keyframe_traj_list)

            # Check for loop closures
            loop_closure_idxs = self.detect_loop_closures(new_pose)
            if len(loop_closure_idxs) > 0:
                self.add_loop_closures(loop_closure_idxs, data)
                self.optimize()

        self.pose_idx += 1

    def detect_loop_closures(self, new_pose: np.ndarray):
        """Check if the new pose is close to any existing keyframes"""
        loop_closure_idxs = []
        if self.keyframe_traj is not None and len(self.keyframe_traj) > LOOP_CLOSURE_EXCLUDE:
            # Exclude the most recent N keyframes
            # Only check 2D (xy) distance
            dists = np.linalg.norm(
                self.keyframe_traj[:-LOOP_CLOSURE_EXCLUDE, :2, 3] - new_pose[:2, 3], axis=1
            )
            dist_check_idxs = np.where(dists < self.LC_DIST_THRESHOLD)[0]
            for idx in dist_check_idxs:
                if (
                    rotation_matrix_error(self.keyframe_traj[idx, :3, :3], new_pose[:3, :3])
                    < self.LC_ANGLE_THRESHOLD
                ):
                    loop_closure_idxs.append(idx)
        return loop_closure_idxs

    def add_loop_closures(self, idxs: np.ndarray, data: dict):
        """Add loop closure factors to the graph"""
        for idx in idxs:
            pose_idx = list(self.keyframe_data.keys())[idx]
            keyframe_data = self.keyframe_data[pose_idx]
            relative_pose, num_matches = keyframe_estimate_loop_closure_pose(
                self.feature_tracker,
                keyframe_data,
                data["FrontLeft"],
                self.LC_MIN_SCORE,
                self.LC_MIN_MATCHES,
            )
            if relative_pose is None:
                continue
            # Add loop closure factor
            self.graph.add(
                gtsam.BetweenFactorPose3(
                    X(pose_idx),
                    X(self.pose_idx),
                    gtsam.Pose3(relative_pose),
                    LOOP_CLOSURE_NOISE,
                )
            )
            self.loop_closures.append((pose_idx, self.pose_idx))
            self.loop_closures_poses.append(relative_pose)
            self.loop_closures_matches.append(num_matches)

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

    def project_point_map(self) -> SemanticPointCloud:
        """Project the point map into world coordinates"""
        all_points = []
        all_labels = []
        for idx, data in self.point_map.items():
            pose = self.values.atPose3(X(idx)).matrix()
            world_points = apply_transform(pose, data["points"])
            all_points.append(world_points)
            all_labels.append(data["labels"])
        all_points = np.vstack(all_points)
        all_labels = np.hstack(all_labels)
        return SemanticPointCloud(all_points, all_labels)

    def get_local_map(self):
        """Output local semantic point cloud for path planning"""
        # TODO
        pass

    def get_state(self):
        """Get the current state of the backend"""
        # TODO: can we save the point map too?
        return {
            "odometry": np.array(self.odometry),
            "odometry_sources": np.array(self.odometry_sources),
            "loop_closures": np.array(self.loop_closures),
            "loop_closures_poses": np.array(self.loop_closures_poses),
        }
