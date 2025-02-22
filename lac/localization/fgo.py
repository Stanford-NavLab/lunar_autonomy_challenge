"""Factor Graph Optimization with SymForce

Step -1 is initialization. Poses start at index -1
The first run_step call starts at step 0. Odometry and measurements start at index 0.

"""

import numpy as np
from tqdm import tqdm
import symforce.symbolic as sf
import symforce.typing as T
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer

from lac.localization.symforce_util import (
    odometry_residual,
    make_pose,
    to_np_pose,
    copy_pose,
    flatten_list,
    odometry_lander_relpose_fgo,
)


def get_poses_from_values(values):
    poses = []
    for key in values.keys():
        if key.startswith("pose_"):
            poses.append(values[key])
    return poses


def cast_sf_pose(pose):
    """Cast a sym.pose3.Pose3 to sf.Pose3"""
    return sf.Pose3.from_storage(pose.to_storage())


class FactorGraph:
    def __init__(self, initial_pose, odometry_sigma, fiducial_sigma):
        self.values = Values()
        self.values["identity_pose"] = sf.Pose3()
        self.values["odometry_sigma"] = sf.V6(odometry_sigma)
        self.values["fiducial_sigma"] = sf.V6(fiducial_sigma)
        self.values["epsilon"] = sf.numeric_epsilon

        self.values["pose_0"] = make_pose(initial_pose)
        self.num_poses = 1

        self.factors = {}
        self.factors[0] = []

    def get_poses(self):
        return [to_np_pose(pose) for pose in get_poses_from_values(self.values)]

    def add_pose(self, i: int, pose: np.ndarray = None):
        """Add pose for step i"""
        self.factors[i] = []
        if pose is not None:
            self.values[f"pose_{i}"] = make_pose(pose)
        else:
            self.values[f"pose_{i}"] = copy_pose(self.values[f"pose_{i - 1}"])
        self.num_poses += 1

    def add_odometry_factor(self, i: int, odometry: np.ndarray):
        self.values[f"odometry_{i}"] = make_pose(odometry)
        self.factors[i].append(
            Factor(
                residual=odometry_residual,
                keys=[
                    f"pose_{i - 1}",
                    f"pose_{i}",
                    f"odometry_{i}",
                    "odometry_sigma",
                    "epsilon",
                ],
            )
        )

    def add_pose_measurement_factor(self, i: int, pose_measurement: np.ndarray):
        self.values[f"pose_measurement_{i}"] = make_pose(pose_measurement)
        self.factors[i].append(
            Factor(
                residual=odometry_residual,
                keys=[
                    "identity_pose",
                    f"pose_{i}",
                    f"pose_measurement_{i}",
                    "fiducial_sigma",
                    "epsilon",
                ],
            )
        )

    def optimize(self, window: T.Tuple[int] = None):
        """

        window : tuple
            Start and end pose indices (inclusive) of the window to optimize

        """
        if window is None:
            idxs = list(range(1, self.num_poses))
        else:
            idxs = list(range(window[0], window[1] + 1))
        optimized_keys = [f"pose_{i}" for i in idxs]
        factors_to_optimize = flatten_list([self.factors[i] for i in idxs])

        optimizer = Optimizer(
            factors=factors_to_optimize,
            optimized_keys=optimized_keys,
            params=Optimizer.Params(
                verbose=False, initial_lambda=1e4, iterations=100, lambda_down_factor=0.5
            ),
        )
        result = optimizer.optimize(self.values)

        for key in self.values.keys():
            if key.startswith("pose_"):
                self.values[key] = cast_sf_pose(result.optimized_values[key])

        optimized_poses = get_poses_from_values(result.optimized_values)
        return [to_np_pose(pose) for pose in optimized_poses], result


def sliding_window_fgo(
    params, initial_poses, odometry_measurements, pose_measurements, lander_pose_world
):
    N = params["N"]
    N_WINDOW = params["N_WINDOW"]
    N_SHIFT = params["N_SHIFT"]
    ODOM_SIGMA = params["ODOM_SIGMA"]
    LANDER_RELPOSE_SIGMA = params["LANDER_RELPOSE_SIGMA"]

    init_poses = initial_poses[:N_WINDOW]
    fgo_poses = [None] * N
    k_max = (N - N_WINDOW) // N_SHIFT

    for k in tqdm(range(k_max)):
        window = slice(N_SHIFT * k, N_SHIFT * k + N_WINDOW)
        odometry = odometry_measurements[window][:-1]
        lander_measurements = pose_measurements[window]

        opt_poses, result = odometry_lander_relpose_fgo(
            init_poses,
            lander_pose_world,
            odometry,
            lander_measurements,
            ODOM_SIGMA,
            LANDER_RELPOSE_SIGMA,
            debug=False,
        )
        fgo_poses[N_SHIFT * k : N_SHIFT * (k + 1)] = opt_poses[:N_SHIFT]

        init_poses[:-N_SHIFT] = opt_poses[N_SHIFT:]
        if k != k_max - 1:
            pose = opt_poses[-1]
            for i in range(N_SHIFT):
                init_poses[-N_SHIFT + i] = pose @ odometry_measurements[window][-1]
                pose = init_poses[-N_SHIFT + i]

    return fgo_poses


def odometry_fiducial_fgo(
    init_poses: list,
    lander_pose: np.ndarray,
    odometry_measurements: list,
    fiducial_measurements: list,
    odometry_sigma: np.ndarray,
    fiducial_sigma: np.ndarray,
    fix_first_pose=True,
    debug=False,
):
    """

    Parameters
    ----------
    init_poses : list
        List of initial poses (4x4 np.array)
    lander_pose : 4x4 np.array
        Pose of the lander in world frame (4x4 np.array)
    odometry_measurements : list
        List of odometry measurements (4x4 np.array)
    fiducial_measurements : list
        List of absolute pose measurements from fiducials (4x4 np.array)
    odometry_sigma : np.array (6,)
        Odometry measurement standard deviation
    fiducial_sigma : np.array (6,)
        Fiducial pose measurement standard deviation

    """
    # Build values
    values = Values()
    values["poses"] = [make_pose(T) for T in init_poses]
    values["lander_pose"] = make_pose(lander_pose)
    values["odometry"] = [make_pose(T) for T in odometry_measurements]
    values["pose_measurements"] = [make_pose(T) for T in fiducial_measurements]
    values["odometry_sigma"] = sf.V6(odometry_sigma)
    values["fiducial_sigma"] = sf.V6(fiducial_sigma)
    values["epsilon"] = sf.numeric_epsilon

    start_idx = 1 if fix_first_pose else 0

    # Build factors
    factors = []
    for i in range(len(odometry_measurements)):
        factors.append(
            Factor(
                residual=odometry_residual,
                keys=[
                    f"poses[{i}]",
                    f"poses[{i + 1}]",
                    f"odometry[{i}]",
                    "odometry_sigma",
                    "epsilon",
                ],
            )
        )
    for i in range(start_idx, len(fiducial_measurements)):
        factors.append(
            Factor(
                residual=bearing_residual,
                keys=[
                    f"poses[{i}]",
                    "lander_pose",
                    f"lander_los_measurements[{i}]",
                    "lander_sigma",
                    "epsilon",
                ],
            )
        )

    optimized_keys = [f"poses[{i}]" for i in range(start_idx, len(init_poses))]
    optimizer = Optimizer(
        factors=factors,
        optimized_keys=optimized_keys,
        # Return problem stats for every iteration
        debug_stats=debug,
        # Customize optimizer behavior
        params=Optimizer.Params(
            verbose=debug, initial_lambda=1e4, iterations=100, lambda_down_factor=0.5
        ),
    )
    result = optimizer.optimize(values)
    optimized_poses = result.optimized_values["poses"]
    return [to_np_pose(pose) for pose in optimized_poses], result
