"""SLAM master class"""

import numpy as np
import plotly.graph_objects as go
import gtsam
from gtsam.symbol_shorthand import X, L, B, V

from lac.slam.feature_tracker import FeatureTracker
from lac.utils.geometry import in_bbox
from lac.params import FL_X, FL_Y, IMG_HEIGHT, IMG_WIDTH, SCENE_BBOX, DT
from lac.utils.frames import get_cam_pose_rover, CAMERA_TO_OPENCV_PASSIVE
from lac.utils.plotting import plot_poses, plot_3d_points

""" Constants and parameters """

# 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
POSE_SIGMA = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
PIXEL_NOISE = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v
HUBER_PIXEL_NOISE = gtsam.noiseModel.Robust(gtsam.noiseModel.mEstimator.Huber(k=1.5), PIXEL_NOISE)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.00087, 0.00087, 0.00087, 0.005, 0.005, 0.005])  # rotation, translation
)
POINT_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 1e-2)  # for landmarks

# Camera intrinsics
K = gtsam.Cal3_S2(FL_X, FL_Y, 0.0, IMG_WIDTH / 2, IMG_HEIGHT / 2)

# Rover to camera transform
rover_T_cam_FL = get_cam_pose_rover("FrontLeft")
rover_T_cam_FL[:3, :3] = rover_T_cam_FL[:3, :3] @ CAMERA_TO_OPENCV_PASSIVE
ROVER_T_CAM_FRONT_LEFT = gtsam.Pose3(rover_T_cam_FL)
rover_T_cam_FR = get_cam_pose_rover("FrontRight")
rover_T_cam_FR[:3, :3] = rover_T_cam_FR[:3, :3] @ CAMERA_TO_OPENCV_PASSIVE
ROVER_T_CAM_FRONT_RIGHT = gtsam.Pose3(rover_T_cam_FR)

IMU_PARAMS = gtsam.PreintegrationParams.MakeSharedU(g=1.622)
gyro_sigma = 1e-5
accel_sigma = 1e-5
integration_sigma = 1e-5
I_3x3 = np.eye(3)
IMU_PARAMS.setGyroscopeCovariance(gyro_sigma**2 * I_3x3)
IMU_PARAMS.setAccelerometerCovariance(accel_sigma**2 * I_3x3)
IMU_PARAMS.setIntegrationCovariance(integration_sigma**2 * I_3x3)
BIAS_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 1e-5)
ZERO_BIAS = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
INITIAL_VELOCITY_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)


class SLAM:
    """
    Factor graph SLAM class

    Assumes pose indices are in order with no skips (0, 1, 2, ...)

    """

    def __init__(self):
        self.poses = {}

        self.projection_factors: dict[int, list] = {}  # List of projection factors for each pose
        self.odometry_factors = {}
        self.pose_to_landmark_map: dict[int, np.ndarray] = {}  # Map of pose to landmark ids
        self.landmarks = {}

        self.landmark_ids = set()

        self.accum = gtsam.PreintegratedImuMeasurements(IMU_PARAMS)
        self.bias_key = B(0)
        self.imu_factors = {}

        self.lm_params = gtsam.LevenbergMarquardtParams()
        self.lm_params.setVerbosity("TERMINATION")
        self.gnc_params = gtsam.GncLMParams()
        self.gnc_params.setLossType(gtsam.GncLossType.TLS)  # GM, TLS
        self.gnc_params.setVerbosityGNC(gtsam.GncLMParams.Verbosity.SUMMARY)

    def add_pose(self, i: int, pose: np.ndarray):
        """Add a pose to the graph"""
        self.poses[i] = pose

    def add_odometry_factor(self, i: int, odometry: np.ndarray):
        """Add an odometry factor to the graph"""
        self.odometry_factors[i] = gtsam.BetweenFactorPose3(X(i - 1), X(i), gtsam.Pose3(odometry), ODOMETRY_NOISE)

    def accumulate_imu_measurement(self, imu: np.ndarray):
        """Accumulate IMU measurement"""
        acc = imu[:3]
        gyro = np.array([imu[4], -imu[3], imu[5]])
        self.accum.integrateMeasurement(acc, gyro, DT)

    def add_imu_factor(self, i: int):
        """Add an IMU factor to the graph"""
        self.imu_factors[i] = gtsam.ImuFactor(X(i - 1), V(i - 1), X(i), V(i), self.bias_key, self.accum)
        self.accum.resetIntegration()

    def add_vision_factors(self, i: int, tracker: FeatureTracker):
        """Add a group of vision factors"""
        self.pose_to_landmark_map[i] = []
        self.projection_factors[i] = []

        for j, id in enumerate(tracker.track_ids):
            # Don't add landmarks outside scene bbox
            if in_bbox(tracker.world_points[j], SCENE_BBOX):
                self.pose_to_landmark_map[i].append(id)
                # Front left camera
                self.projection_factors[i].append(
                    gtsam.GenericProjectionFactorCal3_S2(
                        tracker.prev_pts[j].copy(),
                        HUBER_PIXEL_NOISE,
                        X(i),
                        L(id),
                        K,
                        ROVER_T_CAM_FRONT_LEFT,
                    )
                )
                # Front right camera
                self.projection_factors[i].append(
                    gtsam.GenericProjectionFactorCal3_S2(
                        tracker.prev_pts_right[j].copy(),
                        HUBER_PIXEL_NOISE,
                        X(i),
                        L(id),
                        K,
                        ROVER_T_CAM_FRONT_RIGHT,
                    )
                )
                if id not in self.landmark_ids:
                    self.landmark_ids.add(id)
                    self.landmarks[id] = tracker.world_points[j].copy()

    def build_graph(
        self,
        window: list,
        use_imu: bool = False,
        use_odometry: bool = False,
        first_pose: str = "fix",  # ["fix", "prior"]
    ):
        """Build the graph for a window of poses"""
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        active_landmarks = set()
        if use_imu:
            values.insert(self.bias_key, ZERO_BIAS)
            graph.add(gtsam.PriorFactorConstantBias(self.bias_key, ZERO_BIAS, BIAS_NOISE))

        for i in window:
            values.insert(X(i), gtsam.Pose3(self.poses[i]))
            if use_imu:
                values.insert(V(i), np.zeros(3))

            # Fix first pose
            if i == window[0]:
                if first_pose == "fix":
                    graph.add(gtsam.NonlinearEqualityPose3(X(i), gtsam.Pose3(self.poses[i])))
                elif first_pose == "prior":
                    graph.add(gtsam.PriorFactorPose3(X(i), gtsam.Pose3(self.poses[i]), POSE_SIGMA))
                if use_imu:
                    # NOTE: currently assuming stationary at first pose
                    graph.add(gtsam.PriorFactorVector(V(i), np.zeros(3), INITIAL_VELOCITY_NOISE))

            else:
                if use_odometry:
                    graph.add(self.odometry_factors[i])
                if use_imu:
                    graph.add(self.imu_factors[i])

            for factor in self.projection_factors[i]:
                graph.add(factor)
            active_landmarks.update(self.pose_to_landmark_map[i])

        for id in active_landmarks:
            values.insert(L(id), self.landmarks[id])

        return graph, values, active_landmarks

    def optimize(
        self,
        window: list,
        use_gnc: bool = False,
        use_odometry: bool = False,
        remove_outliers: bool = False,
        verbose: bool = False,
    ):
        """Optimize over window of poses"""
        # Build the graph
        graph, values, active_landmarks = self.build_graph(window, use_odometry=use_odometry)

        # Optimize
        if use_gnc:
            optimizer = gtsam.GncLMOptimizer(graph, values, self.gnc_params)
        else:
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, self.lm_params)
        result = optimizer.optimize()
        if verbose:
            print(f"initial error = {graph.error(values)}")
            print(f"final error = {graph.error(result)}")

        # Update the poses
        for i in window:
            self.poses[i] = result.atPose3(X(i)).matrix()

        # Update the landmarks
        for id in active_landmarks:
            if in_bbox(result.atPoint3(L(id)), SCENE_BBOX):
                self.landmarks[id] = result.atPoint3(L(id))
            else:
                print(f"Landmark {id} optimized outside scene bbox")
                # Remove landmarks outside scene bbox
                del self.landmarks[id]
                # TODO: Remove associated factors
                for key in self.pose_to_landmark_map:
                    self.pose_to_landmark_map[key] = [x for x in self.pose_to_landmark_map[key] if x != id]

        # Remove outliers
        if remove_outliers:
            for i in window:
                for factor in self.projection_factors[i][:]:
                    residual = factor.unwhitenedError(result)
                    pixel_error = np.linalg.norm(residual)
                    if pixel_error > 5.0:
                        # Remove factor from self.projection_factors[i]
                        self.projection_factors[i].remove(factor)
                        id = gtsam.Symbol(factor.keys()[1]).index()
                        # Remove landmark from self.landmarks
                        if id in self.pose_to_landmark_map[i]:
                            self.pose_to_landmark_map[i].remove(id)

        return result, graph, values

    def plot(
        self,
        start: int = 0,
        end: int = -1,
        step: int = 1,
        show_landmarks: bool = True,
        show_factors: bool = False,
    ):
        """Plot the graph"""
        if end == -1:
            end = len(self.poses)
        idxs = list(range(start, end, step))
        poses_to_plot = [self.poses[i] for i in idxs]
        active_landmarks = set()
        for i in idxs:
            active_landmarks.update(self.pose_to_landmark_map[i])
        landmarks_to_plot = np.array([self.landmarks[j] for j in active_landmarks])
        fig = plot_poses(poses_to_plot, no_axes=True, color="green", name="SLAM poses")

        if show_landmarks:
            fig = plot_3d_points(landmarks_to_plot, fig=fig, color="orange", name="Landmarks")

        # Plot the reprojection factors
        if show_factors:
            for i in idxs:
                for j, id in enumerate(self.pose_to_landmark_map[i]):
                    if id in active_landmarks:
                        fig.add_trace(
                            go.Scatter3d(
                                x=[self.poses[i][0, 3], self.landmarks[id][0]],
                                y=[self.poses[i][1, 3], self.landmarks[id][1]],
                                z=[self.poses[i][2, 3], self.landmarks[id][2]],
                                mode="lines",
                                line=dict(color="red", width=2),
                                name=f"Reprojection {i}_{id}",
                            )
                        )

        fig.update_layout(height=900, width=1600, scene_aspectmode="data")
        return fig


class PoseGraph:
    """Graph for odometry and loop closure relative pose optimization"""

    def __init__(self):
        self.poses = {}
        self.odometry_factors = {}  # i -> factor between i-1 and i
        self.loop_closure_factors = {}  # (i,j) -> factor between i and j

        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        self.lm_params = gtsam.LevenbergMarquardtParams()

    def add_pose(self, i: int, pose: np.ndarray):
        """Add a pose to the graph"""
        self.values.insert(X(i), gtsam.Pose3(pose))

    def add_odometry_factor(self, i: int, odometry: np.ndarray):
        """Add an odometry factor to the graph"""
        self.graph.add(gtsam.BetweenFactorPose3(X(i - 1), X(i), gtsam.Pose3(odometry), ODOMETRY_NOISE))

    def add_loop_closure_factor(self, i: int, j: int, relative_pose: np.ndarray):
        """Add an loop closure factor to the graph"""
        self.graph.add(gtsam.BetweenFactorPose3(X(i), X(j), gtsam.Pose3(relative_pose), ODOMETRY_NOISE))
