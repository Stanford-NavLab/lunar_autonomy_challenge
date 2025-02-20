import numpy as np
import copy

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
            self.Phi_store  = {0: np.eye(P0.shape[0])}
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

        for k in range(self.prev_meas_tidx+1, tidx):
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

        for tj in range(self.lent):   # 0, 1, ..., lent-1
            k = (self.lent - 1) - tj  # lent-1, lent-3, ..., 0
            Phi_k_k1 = self.Phi_store[k+1]
            Phat_k = self.Phat_store[k]
            Pbar_k1 = self.Pbar_store[k+1]
            xbar_k1 = self.xbar_store[k+1]
            
            # update
            # S = Phat_k * Phi_k_k1.T * pinv(Pbar_k1)
            S = Phat_k @ Phi_k_k1.T @ np.linalg.inv(Pbar_k1);  
            Phat= Phat_k + S @ (Phat - Pbar_k1) @ S.T
           
            xhat = self.xhat_store[k] + S @ (xhat - xbar_k1)
            self.Phat_store_smooth[k] = Phat
            self.xhat_store_smooth[k] = xhat

        self.smoothed = True

    def get_array(self):
        """
        Convert the dictionary to array
        """
        result = {
            "xhat": np.zeros((self.lent +1, self.x.shape[0])),
            "Phat": np.zeros((self.lent +1, self.P.shape[0], self.P.shape[1])),
            "xhat_smooth": np.zeros((self.lent +1, self.x.shape[0])),
            "Phat_smooth": np.zeros((self.lent+1, self.P.shape[0], self.P.shape[1])),
        }
        
        for k in range(self.lent + 1):
            result["xhat"][k] = self.xhat_store[k]
            result["Phat"][k] = self.Phat_store[k]
            if self.smoothed:
                result["xhat_smooth"][k] = self.xhat_store_smooth[k]
                result["Phat_smooth"][k] = self.Phat_store_smooth[k]

        return result
