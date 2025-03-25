"""

Notes on GTSAM:



"""

import numpy as np
import gtsam
import gtsam_unstable
from gtsam.symbol_shorthand import B, V, X, L

from gtsam import (
    Cal3_S2,
    DoglegOptimizer,
    LevenbergMarquardtOptimizer,
    GenericProjectionFactorCal3_S2,
    NonlinearFactorGraph,
    PriorFactorPoint3,
    PriorFactorPose3,
    Values,
    Pose3,
)

from lac.params import FL_X, FL_Y, IMG_HEIGHT, IMG_WIDTH
from lac.utils.frames import get_cam_pose_rover, CAMERA_TO_OPENCV_PASSIVE

# 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
POSE_SIGMA = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
PIXEL_NOISE = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v
POINT_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 1e-2)

# Camera intrinsics
K = Cal3_S2(FL_X, FL_Y, 0.0, IMG_WIDTH / 2, IMG_HEIGHT / 2)

# Rover to camera transform
rover_T_cam = get_cam_pose_rover("FrontLeft")
rover_T_cam[:3, :3] = rover_T_cam[:3, :3] @ CAMERA_TO_OPENCV_PASSIVE
ROVER_T_CAM = Pose3(rover_T_cam)


class GtsamFactorGraph:
    def __init__(self):
        self.graph = NonlinearFactorGraph()
        self.initial_estimate = Values()

        self.landmark_ids = set()

        self.optimizer_params = gtsam.LevenbergMarquardtParams()
        self.optimizer_params.setVerbosity("TERMINATION")

    def add_pose(self, i: int, pose: np.ndarray):
        """Add a pose to the graph"""
        self.initial_estimate.insert(X(i), Pose3(pose))

    def add_pose_prior(self, i: int, pose: np.ndarray, sigma: np.ndarray = POSE_SIGMA):
        """Add a pose prior to the graph"""
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(sigma)
        self.graph.push_back(PriorFactorPose3(X(i), Pose3(pose), pose_noise))

    def add_vision_factors(self, i: int, points: np.ndarray, pixels: np.ndarray, ids: np.ndarray):
        """Add a group of vision factors to the graph"""
        for j, id in enumerate(ids):
            # Reprojection factor
            self.graph.push_back(
                GenericProjectionFactorCal3_S2(pixels[j], PIXEL_NOISE, X(i), L(id), K, ROVER_T_CAM)
            )
            # Add landmark (point) to the graph
            if id not in self.landmark_ids:
                self.landmark_ids.add(id)
                self.initial_estimate.insert(L(id), points[j])
                self.graph.push_back(PriorFactorPoint3(L(id), points[j], POINT_NOISE))

    def optimize(self):
        """Optimize the graph"""
        optimizer = LevenbergMarquardtOptimizer(
            self.graph, self.initial_estimate, self.optimizer_params
        )
        result = optimizer.optimize()
        return result


class GtsamVIO:
    def __init__(self, window_size=3, fix_landmarks=True):
        self.factors: dict[int, list] = {}
        self.poses = {}
        self.latest_pose_idx = 0
        self.window_size = window_size
        self.fix_landmarks = fix_landmarks

        self.projection_factors: dict[int, list] = {}
        self.pose_to_landmark_map: dict[int, np.ndarray] = {}
        self.landmarks = {}

        self.landmark_ids = set()
        self.optimizer_params = gtsam.LevenbergMarquardtParams()
        self.optimizer_params.setVerbosity("DETAIL")

    def add_pose(self, i: int, pose: np.ndarray):
        """Add a pose to the graph"""
        self.poses[i] = pose
        self.latest_pose_idx = i

    def add_vision_factors(self, i: int, points: np.ndarray, pixels: np.ndarray, ids: np.ndarray):
        """Add a group of vision factors"""
        self.pose_to_landmark_map[i] = ids
        self.projection_factors[i] = []

        for j, id in enumerate(ids):
            self.projection_factors[i].append(
                GenericProjectionFactorCal3_S2(pixels[j], PIXEL_NOISE, X(i), L(id), K, ROVER_T_CAM)
            )
            if id not in self.landmark_ids:
                self.landmark_ids.add(id)
                self.landmarks[id] = points[j]

    def optimize(self, verbose=False):
        """Sliding window optimization"""
        if self.latest_pose_idx + 1 < self.window_size:
            print("Not enough poses to optimize")
            return

        # Get the window of poses
        window = list(range(self.latest_pose_idx - self.window_size + 1, self.latest_pose_idx + 1))

        # Build the graph
        graph = NonlinearFactorGraph()
        values = Values()
        active_landmarks = set()
        for i in window:
            values.insert(X(i), Pose3(self.poses[i]))
            for factor in self.projection_factors[i]:
                graph.push_back(factor)
            active_landmarks.update(self.pose_to_landmark_map[i])

        for id in active_landmarks:
            values.insert(L(id), self.landmarks[id])
            if self.fix_landmarks:
                graph.add(gtsam.NonlinearEqualityPoint3(L(id), self.landmarks[id]))
            else:
                graph.push_back(PriorFactorPoint3(L(id), self.landmarks[id], POINT_NOISE))

        # TODO: Remove old landmarks

        # Optimize
        optimizer = LevenbergMarquardtOptimizer(graph, values, self.optimizer_params)
        result = optimizer.optimize()
        if verbose:
            print("initial error = {}".format(graph.error(values)))
            print("final error = {}".format(graph.error(result)))

        # Update the initial poses
        for i in window:
            self.poses[i] = result.atPose3(X(i)).matrix()
        return result


class GtsamSmootherVIO:
    def __init__(self):
        self.factors = NonlinearFactorGraph()
        self.values = Values()
        self.timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()

        self.smoother_lag = 3.0
        self.smoother = gtsam_unstable.BatchFixedLagSmoother(self.smoother_lag)

        self.landmark_ids = set()
        self.landmark_last_observed = {}  # Track last observation timestamp

    def add_pose(self, i: int, pose: np.ndarray):
        """Add a pose timestamped with i"""
        self.values.insert(X(i), Pose3(pose))
        self.timestamps.insert((X(i), i))

    def add_pose_prior(self, i: int, pose: np.ndarray, sigma: np.ndarray = POSE_SIGMA):
        """Add a pose prior to the graph"""
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(sigma)
        self.factors.push_back(PriorFactorPose3(X(i), Pose3(pose), pose_noise))

    def add_vision_factors(self, i: int, points: np.ndarray, pixels: np.ndarray, ids: np.ndarray):
        """Add a group of vision factors to the graph"""
        for j, id in enumerate(ids):
            # Reprojection factor
            self.factors.push_back(
                GenericProjectionFactorCal3_S2(pixels[j], PIXEL_NOISE, X(i), L(id), K, ROVER_T_CAM)
            )
            # Add landmark (point) to the graph
            # NOTE: We add a position prior with low noise as a way of fixing the landmark. Not sure
            # if GTSAM has a way of explicitly not optimizing the landmark positions
            if id not in self.landmark_ids:
                self.landmark_ids.add(id)
                self.values.insert(L(id), points[j])
                self.factors.push_back(PriorFactorPoint3(L(id), points[j], POINT_NOISE))

            # Update last observed timestamp
            self.landmark_last_observed[id] = i

    def solve(self, i: int):
        """Run the batch smoother"""
        self.smoother.update(self.factors, self.values, self.timestamps)
        curr_pose_est = self.smoother.calculateEstimatePose3(X(i))

        self.timestamps.clear()
        self.values.clear()
        self.factors.resize(0)

        return curr_pose_est.matrix()
