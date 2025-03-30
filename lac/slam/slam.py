"""SLAM master class"""

import numpy as np
import gtsam
import gtsam_unstable
from gtsam.symbol_shorthand import B, V, X, L

from gtsam import (
    Cal3_S2,
    LevenbergMarquardtOptimizer,
    GenericProjectionFactorCal3_S2,
    NonlinearFactorGraph,
    PriorFactorPoint3,
    PriorFactorPose3,
    Values,
    Pose3,
)

from lac.utils.geometry import in_bbox
from lac.params import FL_X, FL_Y, IMG_HEIGHT, IMG_WIDTH, SCENE_BBOX
from lac.utils.frames import get_cam_pose_rover, CAMERA_TO_OPENCV_PASSIVE

# 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
POSE_SIGMA = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
PIXEL_NOISE = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v
HUBER_PIXEL_NOISE = gtsam.noiseModel.Robust(gtsam.noiseModel.mEstimator.Huber(1.5), PIXEL_NOISE)
POINT_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 1e-2)

# Camera intrinsics
K = Cal3_S2(FL_X, FL_Y, 0.0, IMG_WIDTH / 2, IMG_HEIGHT / 2)

# Rover to camera transform
rover_T_cam = get_cam_pose_rover("FrontLeft")
rover_T_cam[:3, :3] = rover_T_cam[:3, :3] @ CAMERA_TO_OPENCV_PASSIVE
ROVER_T_CAM = Pose3(rover_T_cam)


class SLAM:
    def __init__(self):
        self.poses = {}

        self.projection_factors: dict[int, list] = {}
        self.pose_to_landmark_map: dict[int, np.ndarray] = {}
        self.landmarks = {}

        self.landmark_ids = set()

        self.optimizer_params = gtsam.LevenbergMarquardtParams()
        # self.optimizer_params = gtsam.GncLMParams()
        # self.optimizer_params.setVerbosity("TERMINATION")

    def add_pose(self, i: int, pose: np.ndarray):
        """Add a pose to the graph"""
        self.poses[i] = pose

    def add_vision_factors(self, i: int, points: np.ndarray, pixels: np.ndarray, ids: np.ndarray):
        """Add a group of vision factors"""
        self.pose_to_landmark_map[i] = []
        self.projection_factors[i] = []

        for j, id in enumerate(ids):
            if in_bbox(points[j], SCENE_BBOX):  # Don't add landmarks outside scene bbox
                self.pose_to_landmark_map[i].append(id)
                self.projection_factors[i].append(
                    GenericProjectionFactorCal3_S2(
                        pixels[j], PIXEL_NOISE, X(i), L(id), K, ROVER_T_CAM
                    )
                )
                if id not in self.landmark_ids:
                    # if in_bbox(points[j], SCENE_BBOX):  # Don't add landmarks outside scene bbox
                    #     self.landmark_ids.add(id)
                    #     self.landmarks[id] = points[j]
                    self.landmark_ids.add(id)
                    self.landmarks[id] = points[j]

    def optimize(self, window: list, verbose: bool = False):
        """Optimize over window of poses"""
        # Build the graph
        graph = NonlinearFactorGraph()
        values = Values()
        active_landmarks = set()
        for i in window:
            values.insert(X(i), Pose3(self.poses[i]))
            # Fix first pose
            if i == window[0]:
                graph.add(gtsam.NonlinearEqualityPose3(X(i), Pose3(self.poses[i])))
            for factor in self.projection_factors[i]:
                graph.push_back(factor)
            active_landmarks.update(self.pose_to_landmark_map[i])

        for id in active_landmarks:
            values.insert(L(id), self.landmarks[id])
            # if window[0] == 0:
            #     # Constrain initial landmarks
            #     graph.push_back(PriorFactorPoint3(L(id), self.landmarks[id], POINT_NOISE))

        # Optimize
        optimizer = LevenbergMarquardtOptimizer(graph, values, self.optimizer_params)
        # optimizer = gtsam.GncLMOptimizer(graph, values, self.optimizer_params)
        result = optimizer.optimize()
        if verbose:
            print("initial error = {}".format(graph.error(values)))
            print("final error = {}".format(graph.error(result)))

        # Update the initial poses
        for i in window:
            self.poses[i] = result.atPose3(X(i)).matrix()

        # Update the landmarks
        for id in active_landmarks:
            if in_bbox(result.atPoint3(L(id)), SCENE_BBOX):
                self.landmarks[id] = result.atPoint3(L(id))
            else:
                print(f"Landmark {id} optimized outside scene bbox")
                # # Remove landmarks outside scene bbox
                # del self.landmarks[id]
                # # TODO: Remove associated factors
                # for key in self.pose_to_landmark_map:
                #     self.pose_to_landmark_map[key] = [
                #         x for x in self.pose_to_landmark_map[key] if x != id
                #     ]

        return result


class RockSLAM:
    def __init__(self):
        self.poses = {}
        self.rocks_positions = {}  # id -> point (3D centroid)
        self.rock_features = {}
        self.landmark_ids = set()

        self.optimizer_params = gtsam.LevenbergMarquardtParams()
        # self.optimizer_params = gtsam.GncLMParams()
        # self.optimizer_params.setVerbosity("TERMINATION")
