import numpy as np
import gtsam
from gtsam.symbol_shorthand import B, V, X, L

from gtsam import (
    Cal3_S2,
    DoglegOptimizer,
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

        self.optimizer_params = gtsam.DoglegParams()

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
            # NOTE: We add a position prior with low noise as a way of fixing the landmark. Not sure
            # if GTSAM has a way of explicitly not optimizing the landmark positions
            if id not in self.landmark_ids:
                self.landmark_ids.add(id)
                self.initial_estimate.insert(L(id), points[j])
                self.graph.push_back(PriorFactorPoint3(L(id), points[j], POINT_NOISE))

    def optimize(self):
        """Optimize the graph"""
        optimizer = DoglegOptimizer(self.graph, self.initial_estimate, self.optimizer_params)
        result = optimizer.optimize()
        return result
