"""Utility functions for Factor Graph Optimization with SymForce"""

import numpy as np
import symforce.symbolic as sf
import symforce.typing as T
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer

from lac.localization.symforce_util import odometry_residual, make_pose, to_np_pose


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
