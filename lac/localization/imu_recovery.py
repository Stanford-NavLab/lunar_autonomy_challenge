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


def rotation_residual(R_prev: sf.Rot3, R_curr: sf.Rot3, omega: sf.V3, dt: float) -> sf.V3:
    Omega = (sf.M33.eye() - (R_prev * R_curr.inverse()).to_rotation_matrix()) / dt
    return sf.V3(omega[0] - Omega[2, 1], omega[1] - Omega[0, 2], omega[2] - Omega[1, 0])


def recover_rotation(R_prev, omega, dt):
    """
    R_prev: np.ndarray (3, 3) - Previous rotation matrix
    omega: np.ndarray (3,) - Angular velocity from IMU

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


def recover_translation():
    """ """
    pass
