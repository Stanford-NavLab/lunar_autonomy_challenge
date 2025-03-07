"""Dynamics utils"""

import numpy as np
def dubins_step(x, u, dt):
    """Run one step of dynamics

    Parameters
    ----------
    x : np.array
        State vector (x, y, theta)
    u : np.array
        Control vector (v, w)
    dt : float
        Time step

    Returns
    -------
    np.array
        Updated state vector (x, y, theta)

    """
    x_dot = u[0] * np.cos(x[2])
    y_dot = u[0] * np.sin(x[2])
    theta_dot = u[1]
    x_new = x + np.array([x_dot, y_dot, theta_dot]) * dt
    return x_new

def dubins_traj(x0, u, N, dt):
    """Compute dubins trajectory from a sequence of controls
    
    Parameters
    ----------
    x0 : np.array
        Initial state vector (x, y, theta)
    u : np.array (2)
        Control input (v, w)
    N : int
        Number of steps
    dt : float
        Time step
    
    Returns
    -------
    np.array
        Trajectory (x, y, theta)
    
    """
    traj = np.zeros((N, 3))
    traj[0] = x0
    for i in range(1, N):
        traj[i] = dubins_step(traj[i-1], u, dt)
    return traj

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
