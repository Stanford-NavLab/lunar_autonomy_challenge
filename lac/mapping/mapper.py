"""Mapper class"""

import numpy as np
from scipy.interpolate import griddata

from lac.utils.frames import apply_transform
from lac.util import pos_rpy_to_pose
from lac.params import WHEEL_RIG_POINTS


def grid_to_points(grid: np.ndarray, remove_missing=True) -> np.ndarray:
    """Convert grid to set of points.

    grid: np.ndarray
        Array with shape (N, N, 3) where the last dimension is (x, y, z).

    """
    points = grid[:, :, :3].reshape(-1, 3)
    points = points[points[:, 2] != -np.inf]
    return np.array(points)


def interpolate_heights(height_array: np.ndarray) -> np.ndarray:
    """Interpolate the heights of the height array using cubic interpolation.

    height_array: np.ndarray
        The height array with shape (N, N, 3) where the last dimension is (x, y, z).

    """
    N = height_array.shape[0]
    valid_mask = height_array[..., 2] != -np.inf
    x_known = height_array[..., 0][valid_mask]
    y_known = height_array[..., 1][valid_mask]
    z_known = height_array[..., 2][valid_mask]

    # Extract all (x, y) grid points
    x_all = height_array[..., 0].ravel()
    y_all = height_array[..., 1].ravel()

    # Cubic interpolation
    z_interp = griddata((x_known, y_known), z_known, (x_all, y_all), method="cubic")

    # Fill NaNs using nearest neighbor
    z_nn = griddata((x_known, y_known), z_known, (x_all, y_all), method="nearest")
    z_interp[np.isnan(z_interp)] = z_nn[np.isnan(z_interp)]

    height_array_interpolated = np.zeros_like(height_array)
    height_array_interpolated[..., 0] = x_all.reshape(N, N)
    height_array_interpolated[..., 1] = y_all.reshape(N, N)
    height_array_interpolated[..., 2] = z_interp.reshape(N, N)
    return height_array_interpolated


class Mapper:
    def __init__(self):
        self.rock_detections_samples = []

    def wheel_contact_update(self, g_map, ekf_result) -> None:
        """Update the geometric map with the wheel contact points based on EKF trajectory."""
        ekf_trajectory = ekf_result["xhat_smooth"]
        for state in ekf_trajectory:
            pose = pos_rpy_to_pose(state[:3], state[-3:])
            wheel_contact_points = apply_transform(pose, WHEEL_RIG_POINTS)
            for point in wheel_contact_points:
                current_height = g_map.get_height(point[0], point[1])
                if current_height is None:  # Out of bounds
                    continue
                if (current_height == -np.inf) or (current_height > point[2]):
                    g_map.set_height(point[0], point[1], point[2])

    def finalize(self, g_map) -> None:
        height_array = g_map.get_map_array()
        height_array = interpolate_heights(height_array)
