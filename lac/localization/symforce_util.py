"""Utility functions for Factor Graph Optimization with SymForce"""

import numpy as np
import symforce.symbolic as sf
import symforce.typing as T
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer


def odometry_residual(
    world_T_a: sf.Pose3,
    world_T_b: sf.Pose3,
    a_T_b: sf.Pose3,
    diagonal_sigmas: sf.V6,
    epsilon: sf.Scalar,
) -> sf.V6:
    """
    Residual on the relative pose between two timesteps of the robot.
    Args:
        world_T_a: First pose in the world frame
        world_T_b: Second pose in the world frame
        a_T_b: Relative pose measurement between the poses
        diagonal_sigmas: Diagonal standard deviation of the tangent-space error
        epsilon: Small number for singularity handling
    """
    a_T_b_predicted = world_T_a.inverse() * world_T_b
    tangent_error = a_T_b_predicted.local_coordinates(a_T_b, epsilon=epsilon)
    return T.cast(sf.V6, sf.M.diag(diagonal_sigmas.to_flat_list()).inv() * sf.V6(tangent_error))


def imu_gyro_residual(
    T_curr: sf.Pose3,
    T_prev: sf.Pose3,
    angvel: sf.V3,
    dt: float,
    sigma: float,
) -> sf.V3:
    """
    Angular velocity constraint on 2 consecutive poses based on IMU equations.
    """
    R_curr = T_curr.R.to_rotation_matrix()
    R_prev = T_prev.R.to_rotation_matrix()
    rotmat_der = (R_curr - R_prev) / dt
    angvel_mat = rotmat_der * R_curr.T
    expected_angvel = sf.V3(angvel_mat[2, 1], angvel_mat[0, 2], angvel_mat[1, 0])
    return sf.V3(angvel - expected_angvel) / sigma


def imu_accel_residual(
    T_curr: sf.Pose3,
    T_prev: sf.Pose3,
    T_prev_prev: sf.Pose3,
    accel: sf.V3,
    gravity: sf.V3,
    dt: float,
    sigma: float,
) -> sf.V3:
    """
    Acceleration constraint on 3 consecutive poses based on IMU equations.
    """
    expected_accel = (T_curr.t + T_prev_prev.t - 2.0 * T_prev.t) / dt**2.0 + gravity
    expected_accel = T_curr.R.inverse() * expected_accel
    return sf.V3(accel - expected_accel) / sigma


def bearing_residual(
    robot_pose: sf.Pose3,
    landmark_pose: sf.Pose3,
    los_vector: sf.Vector3,
    sigma: sf.Scalar,
    epsilon: sf.Scalar,
) -> sf.V1:
    """
    Residual on the bearing measurement between the robot and a landmark.
    Args:
        robot_pose: Pose of the robot in the world frame
        landmark_pose: Pose of the landmark in the world frame
        los_vector: Unit vector pointing from the robot to the landmark
        sigma: Standard deviation of the bearing measurement
        epsilon: Small number for singularity handling
    """
    robot_to_landmark = (landmark_pose.t - robot_pose.t).normalized(epsilon=epsilon)
    angle_error = sf.acos_safe(robot_to_landmark.dot(los_vector), epsilon=epsilon)
    return sf.V1(angle_error / sigma)


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
