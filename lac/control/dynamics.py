"""Dynamics utils"""

import numpy as np


def arc(x0, u, N, dt):
    """Generates an arc trajectory given an initial state and control input"""

    def dx(v, w, t, eps=1e-15):
        return v * np.sin((w + eps) * t) / (w + eps)

    def dy(v, w, t, eps=1e-15):
        return -v * (1 - np.cos((w + eps) * t)) / (w + eps)

    traj = np.zeros((N, 3))
    traj[:, 0] = [x0[0] + dx(u[0], u[1], i * dt) for i in range(N)]
    traj[:, 1] = [x0[1] + dy(u[0], u[1], i * dt) for i in range(N)]
    traj[:, 2] = x0[2] + u[1] * np.arange(N) * dt

    return traj
