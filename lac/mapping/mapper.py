"""Mapper class"""

import numpy as np
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
from scipy.ndimage import median_filter

from lac.slam.backend import SemanticPointCloud
from lac.perception.segmentation import SemanticClasses
from lac.perception.depth import project_rock_depths_to_world
from lac.utils.frames import apply_transform
from lac.params import WHEEL_RIG_POINTS, MAP_EXTENT, MAP_SIZE


def grid_to_points(grid: np.ndarray, remove_missing=True) -> np.ndarray:
    """Convert grid to set of points.

    grid: np.ndarray
        Array with shape (N, N, 3) where the last dimension is (x, y, z).

    """
    points = grid[:, :, :3].reshape(-1, 3)
    points = points[points[:, 2] != -np.inf]
    return np.array(points)


def robust_mean(values):
    if len(values) == 0:
        return np.nan
    lower = np.percentile(values, 10)
    upper = np.percentile(values, 90)
    clipped = values[(values >= lower) & (values <= upper)]
    return np.mean(clipped) if len(clipped) > 0 else np.nan


def nanmedian_filter(grid, size=3):
    mask = ~np.isnan(grid)
    filled = np.where(mask, grid, 0)
    counts = median_filter(mask.astype(int), size=size)
    smoothed = median_filter(filled, size=size)
    return np.where(counts > 0, smoothed, np.nan)


def bin_points_to_grid(points: np.ndarray, statistic="median") -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_min, x_max = -MAP_EXTENT, MAP_EXTENT
    y_min, y_max = -MAP_EXTENT, MAP_EXTENT

    if statistic == "robust_mean":
        statistic = robust_mean

    grid_medians, x_edges, y_edges, _ = binned_statistic_2d(
        x, y, z, statistic=statistic, bins=MAP_SIZE, range=[[x_min, x_max], [y_min, y_max]]
    )
    # Set Nans to -np.inf
    grid_medians[np.isnan(grid_medians)] = -np.inf
    return grid_medians


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

    height_array_interpolated = height_array.copy()
    height_array_interpolated[..., 0] = x_all.reshape(N, N)
    height_array_interpolated[..., 1] = y_all.reshape(N, N)
    height_array_interpolated[..., 2] = z_interp.reshape(N, N)
    return height_array_interpolated


def process_map(semantic_points: SemanticPointCloud, agent_map: np.ndarray) -> np.ndarray:
    # Height map
    ground_points = semantic_points.points[semantic_points.labels == SemanticClasses.GROUND.value]
    ground_grid = bin_points_to_grid(ground_points)
    agent_map[:, :, 2] = ground_grid
    agent_map[:] = interpolate_heights(agent_map)

    # Rock map
    rock_points = semantic_points.points[semantic_points.labels == SemanticClasses.ROCK.value]
    x_edges = np.linspace(-MAP_EXTENT, MAP_EXTENT, MAP_SIZE + 1)
    y_edges = np.linspace(-MAP_EXTENT, MAP_EXTENT, MAP_SIZE + 1)
    ROCK_COUNT_THRESH = 10
    rock_counts, _, _ = np.histogram2d(
        rock_points[:, 0], rock_points[:, 1], bins=[x_edges, y_edges]
    )
    agent_map[:, :, 3] = np.where(rock_counts > ROCK_COUNT_THRESH, 1, 0)

    return agent_map


class Mapper:
    def __init__(self, geometric_map):
        self.g_map = geometric_map
        map_array = self.g_map.get_map_array()
        map_array[:, :, 3] = 0.0  # initialize rocks
        self.rock_detections = {}
        self.rock_detections_serialized = {}

    def wheel_contact_update(self, poses: list[np.ndarray]) -> None:
        """Update the geometric map with the wheel contact points based on trajectory of poses."""
        for pose in poses:
            wheel_contact_points = apply_transform(pose, WHEEL_RIG_POINTS)
            for point in wheel_contact_points:
                current_height = self.g_map.get_height(point[0], point[1])
                if current_height is None:  # Out of bounds
                    continue
                if (current_height == -np.inf) or (current_height > point[2]):
                    self.g_map.set_height(point[0], point[1], point[2])

    def add_rock_detections(self, step: int, depth_results: list[dict]) -> None:
        self.rock_detections[step] = depth_results
        out_list = []
        for result in depth_results:
            out_list.append(
                [result["left_centroid"][0], result["left_centroid"][1], result["depth"]]
            )
        self.rock_detections_serialized[step] = out_list

    def rock_projection_update(self, poses: list[np.ndarray], camera_config: dict) -> None:
        """Update the geometric map with the projected rock detections based on trajectory of poses."""
        for step, detections in self.rock_detections.items():
            if step >= len(poses):
                continue
            pose = poses[step]
            rock_points_world = project_rock_depths_to_world(
                detections, pose, "FrontLeft", camera_config
            )
            for point in rock_points_world:
                self.g_map.set_rock(point[0], point[1], True)

    def finalize_heights(self) -> None:
        map_array = self.g_map.get_map_array()
        if np.sum(map_array[..., 2] != -np.inf) == 0:
            # No height values have been set
            map_array[..., 2] = 0.0
        else:
            map_array[:] = interpolate_heights(map_array)

    def get_map(self):
        return self.g_map
