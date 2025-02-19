import numpy as np


class EKF:
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
            self.Phi_store = [np.eye(P0.shape[0])]
            self.xbar_store = [x0]
            self.xhat_store = [x0]
            self.Pbar_store = [P0]
            self.Phat_store = [P0]
            self.dyn_funcs = [None]
            self.meas_funcs = [None]
            self.z_store = [None]
            self.xhat_store_smooth = []
            self.Phat_store_smooth = []
            self.lent = 1

    def reset_for_next_iter(self):
        """
        Reset the EKF filter
        """
        x0 = self.xhat_store_smooth[0]
        P0 = self.Pbar_store[0]

        self.x = x0
        self.P = P0   # use the original P0

        self.Phi_store = [np.eye(P0.shape[0])]
        self.xbar_store = [x0]
        self.xhat_store = [x0]
        self.Pbar_store = [P0]
        self.Phat_store = [P0]
        self.Q_store = []       # this has size one smaller than x, P store
        self.dyn_funcs = []     # this has size one smaller than x, P store
        self.meas_funcs = []    # this has size one smaller than x, P store
        self.z_store = []       # this has size one smaller than x, P store
        self.xhat_store_smooth = []
        self.Phat_store_smooth = []
        self.lent = 0

    def predict(self, dyn_func, Q):
        """
        Perform the prediction step of the EKF

        dyn_func: dynamics function
        Q: process noise covariance
        """
        x_new, F = dyn_func(self.x)
        self.P = F @ self.P @ F.T + Q
        self.x = x_new

        if self.store:
            self.lent += 1
            self.dyn_funcs.append(dyn_func)
            self.Phi_store.append(F)
            self.xbar_store.append(x_new)
            self.Pbar_store.append(self.P)

    def update(self, z, meas_func):
        """
        Perform the update step of the EKF
        (Alwas)

        z: measurement
        meas_func: measurement function
        """
        z_hat, H, R = meas_func(self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dz = z - z_hat
        self.x = self.x + K @ dz

        # Joseph form update
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        if self.store:
            self.xhat_store.append(self.x)
            self.Phat_store.append(self.P)
            self.meas_funcs.append(meas_func)
            self.z_store.append(z)

    def smooth(self, n_iter=1):
        """
        Fixed-interval smoothing

        n_iter: number of iterations
        """
        self.xhat_store_smooth = [None] * self.lent
        self.Phat_store_smooth = [None] * self.lent

        # initialize
        self.Phat_store_smooth[-1] = self.P
        self.xhat_store_smooth[-1] = self.x
        
        # backward pass
        for tj in range(self.lent-1):   # 0, 1, ..., lent-2
            k = (self.lent - 2) - tj  # lent-2, lent-3, ..., 0
            Phi_k_k1 = self.Phi_store[k+1]
            Phat_k = self.Phat_store[k]
            Pbar_k1 = self.Pbar_store[k+1]
            xbar_k1 = self.xbar_store[k+1]
            
            # update
            # S = Phat_k * Phi_k_k1.T * pinv(Pbar_k1)
            S = Phat_k * Phi_k_k1.T * np.inv(Pbar_k1);  
            Phat= Phat_k + S * (Phat - Pbar_k1) * S.T
           
            xhat = self.xhat_store[k] + S * (xhat - xbar_k1)
            self.Phat_store_smooth[k] = Phat
            self.xhat_store_smooth[k] = xhat

        if n_iter >=2:
            # reset the log
            lent = self.lent   # store the original lent
            self.reset_for_next_iter()

            # re-run the filter
            for k in range(lent-1):  # lent includes the intial state
                self.predict(self.dyn_funcs[k], self.Q_store[k])
                self.update(self.z_store[k], self.meas_funcs[k])

            # re-run the smoother
            self.smooth(n_iter-1)


