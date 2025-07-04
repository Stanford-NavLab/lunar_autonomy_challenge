"""Recover poses and pose deltas from IMU data.

Delta at step k is defined as the change in pose from k-1 to k.

"""

import numpy as np
import symforce

try:
    symforce.set_epsilon_to_symbol()
except symforce.AlreadyUsedEpsilon:
    print("Already set symforce epsilon")
    pass

import symforce.symbolic as sf
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer

from lac.utils.geometry import interpolate_rotation_matrix
from lac.util import skew_symmetric, normalize_rotation_matrix
from lac.utils.frames import make_transform_mat, invert_transform_mat
from lac.params import LUNAR_GRAVITY, DT


def recover_rotation(R_prev, omega, dt):
    Omega = skew_symmetric(omega)
    R_curr = (np.eye(3) - Omega * dt).T @ R_prev
    return normalize_rotation_matrix(R_curr)


def recover_rotation_delta(omega, dt):
    # NOTE: this is not exact since Omega (R_k - R_{k-1}) is not a perfect skew symmetric matrix
    Omega = skew_symmetric(omega)
    return (np.eye(3) - Omega * dt).T


def rotation_residual(R_prev: sf.Rot3, R_curr: sf.Rot3, omega: sf.V3, dt: float) -> sf.V3:
    Omega = (sf.M33.eye() - (R_prev * R_curr.inverse()).to_rotation_matrix()) / dt
    return sf.V3(omega[0] - Omega[2, 1], omega[1] - Omega[0, 2], omega[2] - Omega[1, 0])


def recover_rotation_exact(R_prev, omega, dt):
    """
    R_prev : np.ndarray (3, 3) - Previous rotation matrix
    omega : np.ndarray (3,) - Angular velocity from IMU
    dt : float - Time step in seconds

    """
    values = Values(
        R_prev=sf.Rot3.from_rotation_matrix(R_prev),
        R_curr=sf.Rot3.from_rotation_matrix(R_prev),  # Initialize guess with previous rotation
        omega=sf.V3(omega),
        dt=dt,
    )

    factors = [Factor(residual=rotation_residual, keys=["R_prev", "R_curr", "omega", "dt"])]

    optimizer = Optimizer(
        factors=factors,
        optimized_keys=["R_curr"],
        debug_stats=False,
        params=Optimizer.Params(verbose=False),
    )
    result = optimizer.optimize(values)
    return result.optimized_values["R_curr"].to_rotation_matrix()


def recover_translation(t_prev_prev, t_prev, R_curr, a, dt):
    """
    t_prev_prev : np.ndarray (3,) - Translation at k-2
    t_prev : np.ndarray (3,) - Translation at k-1
    a : np.ndarray (3,) - Acceleration from IMU
    R_curr : np.ndarray (3, 3) - Current rotation (at k)
    dt : float - Time step in seconds

    """
    t_curr = R_curr @ a * dt**2 - t_prev_prev + 2 * t_prev - LUNAR_GRAVITY * dt**2
    return t_curr


def recover_translation_delta(a, R_curr, v_prev, dt):
    return dt * (LUNAR_GRAVITY * dt - v_prev - R_curr @ a * dt)


def estimate_imu_odometry(a, omega, R_curr, v_prev, dt=DT):
    """
    a : np.ndarray (3,) - Acceleration from IMU
    omega : np.ndarray (3,) - Angular velocity from IMU
    R_curr : np.ndarray (3, 3) - Current rotation (at k)
    v_prev : np.ndarray (3,) - Velocity at k-1
    dt : float - Time step in seconds

    """
    R_delta = recover_rotation_delta(omega, dt)
    t_delta = recover_translation_delta(a, R_curr, v_prev, dt)
    return make_transform_mat(R_delta, t_delta)


class ImuEstimator:
    def __init__(self, initial_pose: np.ndarray, dt: float = DT):
        self.dt = dt
        self.R_prev = initial_pose[:3, :3]
        self.t_prev_prev = initial_pose[:3, 3]
        self.t_prev = initial_pose[:3, 3]
        self.R_curr = self.R_prev
        self.t_curr = self.t_prev

    def reset(self, initial_pose: np.ndarray) -> None:
        self.R_prev = initial_pose[:3, :3]
        self.t_prev_prev = initial_pose[:3, 3]
        self.t_prev = initial_pose[:3, 3]

    def update_pose(self, pose: np.ndarray) -> None:
        self.R_prev = self.R_curr
        self.t_prev = self.t_curr
        self.t_prev_prev = self.t_prev
        self.R_curr = pose[:3, :3]
        self.t_curr = pose[:3, 3]

    def update_pose_from_vo(self, pose: np.ndarray) -> None:
        """Interpolate assuming constant motion (since VO is 10 Hz and IMU is 20 Hz)"""
        self.R_prev = interpolate_rotation_matrix(self.R_curr, pose[:3, :3], alpha=0.5)
        self.t_prev = self.t_curr + 0.5 * (pose[:3, 3] - self.t_curr)
        self.t_prev_prev = self.t_curr
        self.R_curr = pose[:3, :3]
        self.t_curr = pose[:3, 3]

    def update(self, imu_data: np.ndarray, exact: bool = True) -> None:
        a = imu_data[:3]
        omega = imu_data[3:]

        self.R_prev = self.R_curr
        self.t_prev_prev = self.t_prev
        self.t_prev = self.t_curr

        if exact:
            self.R_curr = recover_rotation_exact(self.R_prev, omega, self.dt)
        else:
            self.R_curr = recover_rotation(self.R_prev, omega, self.dt)
        self.t_curr = recover_translation(self.t_prev_prev, self.t_prev, self.R_curr, a, self.dt)

    def get_pose(self) -> np.ndarray:
        return make_transform_mat(self.R_curr, self.t_curr)

    def get_pose_delta(self) -> np.ndarray:
        T_curr = make_transform_mat(self.R_curr, self.t_curr)
        T_prev = make_transform_mat(self.R_prev, self.t_prev)
        # Odometry is intrinsic (right multiply): T_curr = T_prev @ delta
        return invert_transform_mat(T_prev) @ T_curr
