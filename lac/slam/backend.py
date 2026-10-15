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


# Backend options (revision experiments). The defaults reproduce the paper backend (2322a81).
BACKEND_OPTION_DEFAULTS = {
    # Proximity gate against the latest optimized pose of each stored keyframe (paper: the pose
    # recorded when the keyframe was created, never updated)
    "gate_on_optimized_keyframes": False,
    # min_matches counts only matches with stereo depth, the 2D-3D set used by PnP (paper: all
    # LightGlue matches above min_score)
    "count_stereo_matches_only": False,
    # Pose graph over keyframes only: one node per keyframe, between-factor = composed odometry since
    # the previous keyframe with per-component variances summed over the segment (sigma * sqrt(n) for
    # n VO frames). Non-keyframe poses are the optimized keyframe pose chained with VO.
    # (paper: one node per image frame)
    "keyframe_graph": False,
    # Robust loss on loop-closure factors: None (paper), "huber" or "cauchy"; k in whitened units
    "lc_robust_loss": None,
    "lc_robust_k": None,
    # Mahalanobis gate on loop closures: reject if e^T S^-1 e > threshold, e = Logmap of the measured vs
    # current estimated relative pose, S = marginal covariance of the estimated relative pose + loop
    # closure noise. None (paper) = no gate; e.g. 16.81 = chi2(6 dof, 0.99)
    "lc_mahalanobis_gate": None,
}


class Backend:
    """Backend for SLAM"""

    def __init__(
        self,
        initial_pose: np.ndarray,
        feature_tracker: SemanticFeatureTracker,
        config: dict,
        options: dict = None,
    ):
        """Initialize the backend"""
        self.feature_tracker = feature_tracker

        self.options = dict(BACKEND_OPTION_DEFAULTS)
        if options is not None:
            unknown = set(options) - set(BACKEND_OPTION_DEFAULTS)
            assert not unknown, f"Unknown backend options: {unknown}"
            self.options.update(options)
        self.keyframe_graph = self.options["keyframe_graph"]
        self.lc_noise = LOOP_CLOSURE_NOISE
        if self.options["lc_robust_loss"] is not None:
            estimator = {
                "huber": gtsam.noiseModel.mEstimator.Huber,
                "cauchy": gtsam.noiseModel.mEstimator.Cauchy,
            }[self.options["lc_robust_loss"]]
            self.lc_noise = gtsam.noiseModel.Robust.Create(
                estimator.Create(self.options["lc_robust_k"]), LOOP_CLOSURE_NOISE
            )

        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.opt_params = gtsam.LevenbergMarquardtParams()

        # NOTE: should we run IMU odometry during arm raise and initialize with that pose instead?
        self.values.insert(X(0), gtsam.Pose3(initial_pose))
        self.graph.add(gtsam.NonlinearEqualityPose3(X(0), gtsam.Pose3(initial_pose)))
        self.pose_idx = 1  # Index for the next pose

        # Keyframe graph state: node n <-> pose index node_pose_idx[n]; each pose index maps to
        # (node, pose relative to that node)
        self.node_pose_idx = [0]
        self.pose_node = {0: (0, np.eye(4))}
        self.segment_odometry = np.eye(4)
        self.segment_variance = np.zeros(6)

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
        self.loop_closures_gate = []  # (keyframe pose idx, pose idx, d2, accepted) for the Mahalanobis gate

    def key(self, pose_idx: int):
        """Graph key of a pose index (keyframe graph: only keyframes and pose 0 have keys)"""
        if self.keyframe_graph:
            node, rel = self.pose_node[pose_idx]
            assert np.array_equal(rel, np.eye(4)), "only keyframes are graph nodes"
            return X(node)
        return X(pose_idx)

    def get_pose(self, pose_idx: int) -> np.ndarray:
        """Current estimate of a pose"""
        if self.keyframe_graph:
            node, rel = self.pose_node[pose_idx]
            return self.values.atPose3(X(node)).matrix() @ rel
        return self.values.atPose3(X(pose_idx)).matrix()

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
        if data["odometry_source"] == "VO":
            odometry_noise = VO_ODOMETRY_NOISE
        elif data["odometry_source"] == "IMU":
            odometry_noise = IMU_ODOMETRY_NOISE

        if self.keyframe_graph:
            # Accumulate odometry since the last keyframe node
            node = len(self.node_pose_idx) - 1
            self.segment_odometry = self.segment_odometry @ data["odometry"]
            self.segment_variance = self.segment_variance + odometry_noise.sigmas() ** 2
            new_pose = self.values.atPose3(X(node)).matrix() @ self.segment_odometry
            self.pose_node[self.pose_idx] = (node, self.segment_odometry.copy())
            if data["keyframe"]:
                new_node = node + 1
                self.values.insert(X(new_node), gtsam.Pose3(new_pose))
                self.graph.add(
                    gtsam.BetweenFactorPose3(
                        X(node),
                        X(new_node),
                        gtsam.Pose3(self.segment_odometry),
                        gtsam.noiseModel.Diagonal.Sigmas(np.sqrt(self.segment_variance)),
                    )
                )
                self.node_pose_idx.append(self.pose_idx)
                self.pose_node[self.pose_idx] = (new_node, np.eye(4))
                self.segment_odometry = np.eye(4)
                self.segment_variance = np.zeros(6)
        else:
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
            if self.options["gate_on_optimized_keyframes"]:
                keyframe_poses = np.array([self.get_pose(i) for i in self.keyframe_data])
            else:
                keyframe_poses = self.keyframe_traj
            # Exclude the most recent N keyframes
            # Only check 2D (xy) distance
            dists = np.linalg.norm(
                keyframe_poses[:-LOOP_CLOSURE_EXCLUDE, :2, 3] - new_pose[:2, 3], axis=1
            )
            dist_check_idxs = np.where(dists < self.LC_DIST_THRESHOLD)[0]
            for idx in dist_check_idxs:
                if (
                    rotation_matrix_error(keyframe_poses[idx, :3, :3], new_pose[:3, :3])
                    < self.LC_ANGLE_THRESHOLD
                ):
                    loop_closure_idxs.append(idx)
        return loop_closure_idxs

    def loop_closure_mahalanobis(self, marginals, pose_idx: int, relative_pose: np.ndarray) -> float:
        """Squared Mahalanobis distance of a measured loop-closure relative pose against the current
        estimate, with the marginal covariance of the estimated relative pose plus the loop noise"""
        ki, kj = self.key(pose_idx), self.key(self.pose_idx)
        Ti, Tj = self.values.atPose3(ki), self.values.atPose3(kj)
        est_rel = Ti.between(Tj)
        cov = marginals.jointMarginalCovariance(gtsam.KeyVector([ki, kj])).fullMatrix()
        # Tangent-space Jacobian of between(Ti, Tj) = Ti^-1 Tj: [-Ad(est_rel^-1), I]
        H = np.hstack([-est_rel.inverse().AdjointMap(), np.eye(6)])
        S = H @ cov @ H.T + np.diag(LOOP_CLOSURE_NOISE.sigmas() ** 2)
        e = gtsam.Pose3.Logmap(est_rel.between(gtsam.Pose3(relative_pose)))
        return float(e @ np.linalg.solve(S, e))

    def add_loop_closures(self, idxs: np.ndarray, data: dict):
        """Add loop closure factors to the graph"""
        gate = self.options["lc_mahalanobis_gate"]
        marginals = None
        for idx in idxs:
            pose_idx = list(self.keyframe_data.keys())[idx]
            keyframe_data = self.keyframe_data[pose_idx]
            relative_pose, num_matches = keyframe_estimate_loop_closure_pose(
                self.feature_tracker,
                keyframe_data,
                data["FrontLeft"],
                self.LC_MIN_SCORE,
                self.LC_MIN_MATCHES,
                count_stereo_only=self.options["count_stereo_matches_only"],
            )
            if relative_pose is None:
                continue
            if gate is not None:
                # One linearization per keyframe, before this keyframe's loop closures are added
                if marginals is None:
                    marginals = gtsam.Marginals(self.graph, self.values)
                d2 = self.loop_closure_mahalanobis(marginals, pose_idx, relative_pose)
                self.loop_closures_gate.append((pose_idx, self.pose_idx, d2, d2 <= gate))
                if d2 > gate:
                    continue
            # Add loop closure factor
            self.graph.add(
                gtsam.BetweenFactorPose3(
                    self.key(pose_idx),
                    self.key(self.pose_idx),
                    gtsam.Pose3(relative_pose),
                    self.lc_noise,
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
        return [self.get_pose(i) for i in range(self.pose_idx)]

    def project_point_map(self) -> SemanticPointCloud:
        """Project the point map into world coordinates"""
        all_points = []
        all_labels = []
        for idx, data in self.point_map.items():
            pose = self.get_pose(idx)
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
