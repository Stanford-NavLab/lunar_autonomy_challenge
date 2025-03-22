"""Utility functions for Factor Graph Optimization with SymForce"""

import numpy as np
import symforce.symbolic as sf
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer

from lac.localization.symforce_residuals import odometry_residual, bearing_residual


def make_pose(T):
    """
    Utility for creating a symforce Pose3 object from a 4x4 pose matrix

    Parameters
    ----------
    T : np.array (4, 4)
        Pose matrix

    """
    return sf.Pose3(R=sf.Rot3.from_rotation_matrix(T[:3, :3]), t=sf.Vector3(T[:3, 3]))


def copy_pose(pose: sf.Pose3):
    return sf.Pose3(R=pose.R, t=pose.t)


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def to_np_pose(pose: sf.Pose3):
    R = pose.R.to_rotation_matrix().to_numpy()
    t = pose.t.to_numpy()
    return np.block([[R, t.reshape(3, 1)], [0, 0, 0, 1]])


def odometry_lander_LOS_fgo(
    init_poses,
    lander_pose,
    odometry_measurements,
    lander_LOS_measurements,
    odometry_sigma,
    lander_sigma,
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
    lander_LOS_measurements : list
        List of lander LOS measurements ((3,) np.array)
    odometry_sigma : np.array (6,)
        Odometry measurement standard deviation
    lander_sigma : float
        Lander LOS measurement standard deviation

    """
    # Build values
    values = Values()
    values["poses"] = [make_pose(T) for T in init_poses]
    values["lander_pose"] = make_pose(lander_pose)
    values["odometry"] = [make_pose(T) for T in odometry_measurements]
    values["lander_los_measurements"] = [sf.V3(m) for m in lander_LOS_measurements]
    values["odometry_sigma"] = sf.V6(odometry_sigma)
    values["lander_sigma"] = lander_sigma
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
    for i in range(start_idx, len(lander_LOS_measurements)):
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


def odometry_lander_relpose_fgo(
    init_poses,
    lander_pose,
    odometry_measurements,
    lander_relpose_measurements,
    odometry_sigma,
    lander_sigma,
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
    lander_relpose_measurements : list
        List of lander relative pose measurements (4x4 np.array)
    odometry_sigma : np.array (6,)
        Odometry measurement standard deviation
    lander_sigma : np.array (6,)
        Lander measurement standard deviation

    """
    # Build values
    values = Values()
    values["poses"] = [make_pose(T) for T in init_poses]
    values["lander_pose"] = make_pose(lander_pose)
    values["odometry"] = [make_pose(T) for T in odometry_measurements]
    values["lander_relative_poses"] = [make_pose(T) for T in lander_relpose_measurements]
    values["odometry_sigma"] = sf.V6(odometry_sigma)
    values["lander_sigma"] = sf.V6(lander_sigma)
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
    for i in range(start_idx, len(lander_relpose_measurements)):
        factors.append(
            Factor(
                residual=odometry_residual,
                keys=[
                    f"poses[{i}]",
                    "lander_pose",
                    f"lander_relative_poses[{i}]",
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
