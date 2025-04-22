import numpy as np
from scipy.spatial.transform import Rotation

from lac.util import skew_symmetric, normalize_rotation_matrix


# Uses the right handed coordinate system given to the agent, with counterclockwise rotations
# alpha around z axis (yaw), beta around y axis (pitch), gamma around x axis (roll)
def rot_matrix(alpha, beta, gamma):

    row1 = [
        np.cos(alpha) * np.cos(beta),
        np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma),
        np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma),
    ]
    row2 = [
        np.sin(alpha) * np.cos(beta),
        np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
        np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma),
    ]
    row3 = [-np.sin(beta), np.cos(beta) * np.sin(gamma), np.cos(beta) * np.cos(gamma)]

    matrix = np.array([row1, row2, row3])
    return matrix


def dR_dalpha(alpha, beta, gamma):
    row1 = [
        -np.sin(alpha) * np.cos(beta),
        -np.sin(alpha) * np.sin(beta) * np.sin(gamma) - np.cos(alpha) * np.cos(gamma),
        -np.sin(alpha) * np.sin(beta) * np.cos(gamma) + np.cos(alpha) * np.sin(gamma),
    ]
    row2 = [
        np.cos(alpha) * np.cos(beta),
        np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma),
        np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma),
    ]
    row3 = [0, 0, 0]
    return np.array([row1, row2, row3])


def dR_dbeta(alpha, beta, gamma):
    row1 = [
        -np.cos(alpha) * np.sin(beta),
        np.cos(alpha) * np.cos(beta) * np.sin(gamma),
        np.cos(alpha) * np.cos(beta) * np.cos(gamma),
    ]
    row2 = [
        -np.sin(alpha) * np.sin(beta),
        np.sin(alpha) * np.cos(beta) * np.sin(gamma),
        np.sin(alpha) * np.cos(beta) * np.cos(gamma),
    ]
    row3 = [-np.cos(beta), -np.sin(beta) * np.sin(gamma), -np.sin(beta) * np.cos(gamma)]
    return np.array([row1, row2, row3])


def dR_dgamma(alpha, beta, gamma):
    row1 = [
        0,
        np.cos(alpha) * np.sin(beta) * np.cos(gamma) - np.sin(alpha) * np.sin(gamma),
        -np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma),
    ]
    row2 = [
        0,
        np.sin(alpha) * np.sin(beta) * np.cos(gamma) + np.cos(alpha) * np.sin(gamma),
        -np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
    ]
    row3 = [0, np.cos(beta) * np.cos(gamma), -np.cos(beta) * np.sin(gamma)]
    return np.array([row1, row2, row3])


def rot_to_ypr(R):
    r = Rotation.from_matrix(R)
    roll, pitch, yaw = r.as_euler("xyz")
    return yaw, pitch, roll


def dyaw_dR(R):
    r = Rotation.from_matrix(R)
    return r.as_dcm_dR()


def propagate_state(x_prev, a_k, omega_k, dt, with_stm=False, use_numdiff=False):

    gravity = np.array([0.0, 0.0, 1.6220])  # m/s^22

    r_prev = x_prev[:3]
    v_prev = x_prev[3:6]
    roll_prev = x_prev[6]
    pitch_prev = x_prev[7]
    yaw_prev = x_prev[8]

    # angles update
    R_prev = rot_matrix(yaw_prev, pitch_prev, roll_prev)
    w_hat = skew_symmetric(omega_k)
    R_curr = (np.eye(3) - w_hat * dt).T @ R_prev
    R_curr = normalize_rotation_matrix(R_curr)
    yaw_curr, pitch_curr, roll_curr = rot_to_ypr(R_curr)

    # position update
    r_curr = r_prev + v_prev * dt
    v_curr = v_prev + (R_curr @ a_k - gravity) * dt

    # assign variables
    x_k = np.zeros(9)
    x_k[:3] = r_curr
    x_k[3:6] = v_curr
    x_k[6] = roll_curr
    x_k[7] = pitch_curr
    x_k[8] = yaw_curr

    if with_stm:
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * dt  # dr/dv

        if use_numdiff:
            # for derivatives w,r,t to yaw, pitch, roll, we use numerical differentiation
            eps = 1e-8
            # using
            for i in range(3):
                x_prev_plus = x_prev.copy()
                x_prev_plus[6 + i] += eps
                x_curr_plus, _ = propagate_state(x_prev_plus, a_k, omega_k, dt, False)

                x_prev_minus = x_prev.copy()
                x_prev_minus[6 + i] -= eps
                x_curr_minus, _ = propagate_state(x_prev_minus, a_k, omega_k, dt, False)

                F[:, 6 + i] = (x_curr_plus - x_curr_minus) / (2 * eps)
        else:
            # alpha: yaw(8), beta: pitch(7), gamma: roll(6)
            dR_dalpha_curr = dR_dalpha(yaw_curr, pitch_curr, roll_curr)
            dR_dbeta_curr = dR_dbeta(yaw_curr, pitch_curr, roll_curr)
            dR_dgamma_curr = dR_dgamma(yaw_curr, pitch_curr, roll_curr)

            F[3:6, 6] = dR_dgamma_curr @ a_k * dt
            F[3:6, 7] = dR_dbeta_curr @ a_k * dt
            F[3:6, 8] = dR_dalpha_curr @ a_k * dt

        return x_k, F
    else:
        return x_k, False
