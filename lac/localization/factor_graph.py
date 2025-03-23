"""SymForce factor graph for visual-inertial SLAM

Step 0 is initialization. Poses start at index 1
The first run_step call starts at step 1. Odometry and measurements start at index 1.

"""

import numpy as np
import symforce.symbolic as sf
import symforce.typing as T
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer

from lac.localization.symforce_util import (
    make_pose,
    to_np_pose,
    copy_pose,
    flatten_list,
)
from lac.localization.symforce_residuals import (
    imu_gyro_residual,
    imu_accel_residual,
    reprojection_residual,
)
from lac.utils.frames import get_cam_pose_rover
from lac.params import DT, LUNAR_GRAVITY, FL_X, FL_Y, IMG_HEIGHT, IMG_WIDTH


def get_poses_from_values(values: Values) -> list[sf.Pose3]:
    poses = []
    for key in values.keys():
        if key.startswith("pose"):
            poses.append(values[key])
    return poses


def get_pose_idxs(values: Values) -> list[int]:
    return [int(key.split("_")[-1]) for key in values.keys() if key.startswith("pose")]


def cast_sf_pose(pose):
    """Cast a sym.pose3.Pose3 to sf.Pose3"""
    return sf.Pose3.from_storage(pose.to_storage())


class FactorGraph:
    def __init__(self):
        self.values = Values()
        self.values["imu_accel_sigma"] = 1e-5
        self.values["imu_gyro_sigma"] = 1e-5
        self.values["gravity"] = sf.V3(LUNAR_GRAVITY)
        self.values["dt"] = DT
        self.values["epsilon"] = sf.numeric_epsilon

        rover_to_cam = get_cam_pose_rover("FrontLeft")
        self.values["rover_T_cam"] = make_pose(rover_to_cam)
        self.values["reproj_sigma"] = 2.0
        self.values["camera_cal"] = sf.LinearCameraCal(
            focal_length=(FL_X, FL_Y),
            principal_point=(IMG_WIDTH / 2, IMG_HEIGHT / 2),
        )

        self.num_poses = 0
        self.factors: dict[int, list[Factor]] = {}

    def get_pose(self, i: int):
        assert i < self.num_poses, "Pose index out of range"
        return to_np_pose(self.values[f"pose_{i}"])

    def get_all_poses(self):
        return [to_np_pose(pose) for pose in get_poses_from_values(self.values)]

    def add_pose(self, i: int, pose: np.ndarray = None):
        """Add pose for step i"""
        self.factors[i] = []
        if pose is not None:
            self.values[f"pose_{i}"] = make_pose(pose)
        else:
            self.values[f"pose_{i}"] = copy_pose(self.values[f"pose_{i - 1}"])
        self.num_poses += 1

    def add_gyro_factor(self, i: int, angvel: np.ndarray):
        self.values[f"angvel_{i}"] = sf.V3(angvel)
        self.factors[i].append(
            Factor(
                residual=imu_gyro_residual,
                keys=[
                    f"pose_{i}",
                    f"pose_{i - 1}",
                    f"angvel_{i}",
                    "dt",
                    "imu_gyro_sigma",
                ],
            )
        )

    def add_accel_factor(self, i: int, accel: np.ndarray):
        self.values[f"accel_{i}"] = sf.V3(accel)
        self.factors[i].append(
            Factor(
                residual=imu_accel_residual,
                keys=[
                    f"pose_{i}",
                    f"pose_{i - 1}",
                    f"pose_{i - 2}",
                    f"accel_{i}",
                    "gravity",
                    "dt",
                    "imu_accel_sigma",
                ],
            )
        )

    def add_reprojection_factor(
        self, i: int, pixel: np.ndarray, world_point: np.ndarray, point_id: int
    ):
        if f"point_{point_id}" not in self.values:
            self.values[f"point_{point_id}"] = sf.V3(world_point)
        self.values[f"kp_{i}_{point_id}"] = sf.V2(pixel.astype(float))
        self.factors[i].append(
            Factor(
                residual=reprojection_residual,
                keys=[
                    f"point_{point_id}",
                    f"pose_{i}",
                    "rover_T_cam",
                    f"kp_{i}_{point_id}",
                    "camera_cal",
                    "reproj_sigma",
                    "epsilon",
                ],
            )
        )

    def add_reprojection_factors(
        self, i: int, pixels: np.ndarray, world_points: np.ndarray, point_ids: np.ndarray
    ):
        # TODO: add reprojection factors in batch
        pass

    def optimize(self, window: T.Tuple[int] = None, verbose: bool = False):
        """Optimize the graph

        window : tuple
            Start and end pose indices (inclusive) of the window to optimize

        """
        if window is None:
            # idxs = list(range(1, self.num_poses))  # Don't optimize the initial pose
            idxs = get_pose_idxs(self.values)
            idxs = set(idxs) - {0}
        else:
            idxs = list(range(window[0], window[1] + 1))
        optimized_keys = [f"pose_{i}" for i in idxs]
        factors_to_optimize = flatten_list([self.factors[i] for i in idxs])

        optimizer = Optimizer(
            factors=factors_to_optimize,
            optimized_keys=optimized_keys,
            params=Optimizer.Params(
                verbose=verbose,
                initial_lambda=1.0,  # Initial LM damping factor
                lambda_up_factor=4.0,  # damping factor increase upon failure to reduce cost
                lambda_down_factor=0.25,  # damping factor decrease upon cost reduction
                iterations=50,
            ),
        )
        result = optimizer.optimize(self.values)

        for key in self.values.keys():
            if key.startswith("pose_"):
                self.values[key] = cast_sf_pose(result.optimized_values[key])

        return result
