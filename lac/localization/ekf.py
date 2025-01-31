import numpy as np


class EKF:
    def __init__(self, x0, P0):
        self.x = x0
        self.P = P0

    def predict(self, dyn_func, Q):
        x_new, F = dyn_func(self.x)
        self.P = F @ self.P @ F.T + Q
        self.x = x_new

    def update(self, z, meas_func):
        z_hat, H, R = meas_func(self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dz = z - z_hat
        self.x = self.x + K @ dz

        # Joseph form update
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T



