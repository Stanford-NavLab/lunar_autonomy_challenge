import numpy as np
import copy

from lac.params import EKF_P0


class EKF:
    """
    State: (x, y, z, vx, vy, vz, roll, pitch, yaw)
    """

    def __init__(self, x0, P0, store=False):
        """
        Initialize the EKF filter
        x0: initial state
        P0: initial covariance
        store: store the filter history (required for smoothing)
        """
        self.x = x0
        self.P = P0
        self.store = False

        if store:
            self.store = True
            self.Phi_store = {0: np.eye(P0.shape[0])}
            self.xbar_store = {0: x0}
            self.xhat_store = {0: x0}
            self.Pbar_store = {0: P0}
            self.Phat_store = {0: P0}
            self.z_store = {}
            self.xhat_store_smooth = {}
            self.Phat_store_smooth = {}
            self.lent = 1
            self.tidx = 0
            self.prev_meas_tidx = 0
            self.smoothed = False

    def predict(self, tidx, dyn_func, Q):
        """
        Perform the prediction step of the EKF

        dyn_func: dynamics function
        Q: process noise covariance
        """
        x_new, F = dyn_func(self.x)
        self.P = F @ self.P @ F.T + Q
        self.x = x_new

        if self.store:
            self.xbar_store[tidx] = self.x
            self.Pbar_store[tidx] = self.P
            self.xhat_store[tidx] = self.x
            self.Phat_store[tidx] = self.P
            self.Phi_store[tidx] = F
            self.tidx = tidx

    def update(self, tidx, z, meas_func, called_from_smooth=False):
        """
        Perform the update step of the EKF
        (Alwas)

        z: measurement
        meas_func: measurement function
        """
        # add to storage here

        # measurement update
        z_hat, H, R = meas_func(self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dz = z - z_hat
        self.x = self.x + K @ dz

        # Joseph form update
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.z_store[tidx] = z

        for k in range(self.prev_meas_tidx + 1, tidx):
            self.xhat_store[k] = copy.deepcopy(self.xbar_store[k])
            self.Phat_store[k] = copy.deepcopy(self.Pbar_store[k])

        self.xhat_store[tidx] = self.x
        self.Phat_store[tidx] = self.P

        self.prev_meas_tidx = tidx
        self.tidx = tidx

    def smooth(self):
        """
        Fixed-interval smoothing

        n_iter: number of iterations
        """
        self.xhat_store_smooth = {}
        self.Phat_store_smooth = {}
        self.lent = self.tidx

        # initialize
        self.Phat_store_smooth[self.lent] = self.P
        self.xhat_store_smooth[self.lent] = self.x

        # backward pass
        Phat = self.P
        xhat = self.x

        for tj in range(self.lent):  # 0, 1, ..., lent-1
            k = (self.lent - 1) - tj  # lent-1, lent-3, ..., 0
            Phi_k_k1 = self.Phi_store[k + 1]
            Phat_k = self.Phat_store[k]
            Pbar_k1 = self.Pbar_store[k + 1]
            xbar_k1 = self.xbar_store[k + 1]

            # update
            # S = Phat_k * Phi_k_k1.T * pinv(Pbar_k1)
            S = Phat_k @ Phi_k_k1.T @ np.linalg.inv(Pbar_k1)
            Phat = Phat_k + S @ (Phat - Pbar_k1) @ S.T

            xhat = self.xhat_store[k] + S @ (xhat - xbar_k1)
            self.Phat_store_smooth[k] = Phat
            self.xhat_store_smooth[k] = xhat

        self.smoothed = True

    def zero_velocity_update(self, tidx):
        """
        Zero velocity update
        """
        self.x[3:6] = 0
        self.P[3:6, 3:6] = EKF_P0[3:6, 3:6]
        self.P[3:6, :3] = 0
        self.P[:3, 3:6] = 0
        self.P[3:6, 6:] = 0
        self.P[6:, 3:6] = 0
        self.xhat_store[tidx] = self.x
        self.Phat_store[tidx] = self.P

    def get_results(self):
        """
        Convert the dictionary to array
        """
        result = {
            "xhat": np.zeros((self.lent + 1, self.x.shape[0])),
            "Phat": np.zeros((self.lent + 1, self.P.shape[0], self.P.shape[1])),
            "xhat_smooth": np.zeros((self.lent + 1, self.x.shape[0])),
            "Phat_smooth": np.zeros((self.lent + 1, self.P.shape[0], self.P.shape[1])),
        }

        for k in range(self.lent + 1):
            result["xhat"][k] = self.xhat_store[k]
            result["Phat"][k] = self.Phat_store[k]
            if self.smoothed:
                result["xhat_smooth"][k] = self.xhat_store_smooth[k]
                result["Phat_smooth"][k] = self.Phat_store_smooth[k]

        return result


def create_Q(dt, sigma_a, sigma_angle):
    dt2 = dt * dt
    dt3 = dt2 * dt
    Q_rvm = np.block(
        [[np.eye(3) * dt3 / 3.0, np.eye(3) * dt2 / 2.0], [np.eye(3) * dt2 / 2.0, np.eye(3) * dt]]
    ) * (sigma_a**2)

    Q_angle = np.eye(3) * sigma_angle**2

    Q_meas = np.block([[Q_rvm, np.zeros((6, 3))], [np.zeros((3, 6)), Q_angle]])

    return Q_meas


def get_pose_measurement_tag(x, nmeas):
    """
    Generate a measurement vector and the corresponding Jacobian and covariance matrix

    Args:
    x: state vector (9x1)  [x, y, z, vx, vy, vz, roll, pitch, yaw]
    nmeas: number of measurements (detected tags)
    """

    H = np.zeros((0, 9))
    Rdiag = np.zeros((0))

    meas = np.zeros(6 * nmeas)

    for j in range(nmeas):
        meas[6 * j : 6 * j + 3] = x[:3]
        meas[6 * j + 3 : 6 * j + 6] = x[6:]

        # Jacobian of the measurement function
        Htmp = np.zeros((6, 9))
        Htmp[:3, :3] = np.eye(3)
        Htmp[3:, 6:] = np.eye(3)

        H = np.vstack([H, Htmp])

        # TODO: move these params
        std_x = 0.25
        std_y = 0.25
        std_z = 0.25
        std_roll = 0.05
        std_pitch = 0.05
        std_yaw = 0.2

        Rtmp = np.array([std_x**2, std_y**2, std_z**2, std_roll**2, std_pitch**2, std_yaw**2])
        Rdiag = np.concatenate([Rdiag, Rtmp])

    R = np.diag(Rdiag)

    return meas, H, R
