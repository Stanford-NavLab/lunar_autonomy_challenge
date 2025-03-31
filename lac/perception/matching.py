import numpy as np
from numpy.linalg import inv
import cv2

from lac.utils.frames import CAMERA_TO_OPENCV_PASSIVE
from lac.params import FL_X, STEREO_BASELINE
from lac.perception.depth import project_pixel_to_world
from lac.utils.frames import get_cam_pose_rover
from lac.perception.vision import get_camera_intrinsics


def insert_submask(mask: np.ndarray, submask: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    """
    Insert a submask into a mask at a specified center position.

    Inputs:
    -----------
    mask : np.ndarray (H, W) - The original mask
    submask : np.ndarray (h, w) - The submask to insert
    center : tuple (cx, cy) - The center position to insert the submask

    Returns:
    -----------
    np.ndarray (H, W) - The mask with the submask inserted
    """
    cx, cy = center
    h, w = submask.shape
    H, W = mask.shape

    x0 = cx - w // 2
    y0 = cy - h // 2
    x1 = x0 + w
    y1 = y0 + h

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - W)
    pad_bottom = max(0, y1 - H)

    x0_clipped = max(0, x0)
    y0_clipped = max(0, y0)
    x1_clipped = min(W, x1)
    y1_clipped = min(H, y1)

    sx0 = pad_top
    sy0 = pad_left
    sx1 = h - pad_bottom
    sy1 = w - pad_right

    region_mask = mask[y0_clipped:y1_clipped, x0_clipped:x1_clipped]
    region_submask = submask[sx0:sx1, sy0:sy1]

    assert (
        region_mask.shape == region_submask.shape
    ), f"Shape mismatch: mask {region_mask.shape}, submask {region_submask.shape}"

    new_mask = mask.copy()
    new_mask[y0_clipped:y1_clipped, x0_clipped:x1_clipped] = region_submask
    return new_mask


def get_submask(mask: np.ndarray, size: tuple[int, int], center: tuple[int, int], pad_value: int = 0) -> np.ndarray:
    """ "
    Get a submask from a mask at a specified center position.

    Inputs:
    -------
    mask : np.ndarray (H, W) - The original mask
    size : tuple (w, h) - The size of the submask
    center : tuple (cx, cy) - The center position to extract the submask
    pad_value : int - Value to pad the submask with (default: 0)

    Returns:
    --------
    np.ndarray (h, w) - The extracted submask
    """
    cx, cy = center
    w, h = size
    H, W = mask.shape

    x0 = cx - w // 2
    y0 = cy - h // 2
    x1 = x0 + w
    y1 = y0 + h

    px0 = max(0, -x0)
    py0 = max(0, -y0)
    px1 = max(0, x1 - W)
    py1 = max(0, y1 - H)

    x0_clipped = max(x0, 0)
    y0_clipped = max(y0, 0)
    x1_clipped = min(x1, W)
    y1_clipped = min(y1, H)

    cropped = mask[y0_clipped:y1_clipped, x0_clipped:x1_clipped]
    if any([py0, py1, px0, px1]):
        cropped = np.pad(cropped, ((py0, py1), (px0, px1)), mode="constant", constant_values=pad_value)

    return cropped


def get_depths_mono(
    cam_name: str,
    cam_config: dict,
    points0: np.ndarray,
    points1: np.ndarray,
    prev_pose: np.ndarray,
    curr_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get depths from stereo camera images using triangulation.

    Inputs:
    -----------
    cam_name : str - Name of the camera
    cam_config : dict - Camera configuration dictionary
    points0 : np.ndarray (N, 2) - Points in the first image
    points1 : np.ndarray (N, 2) - Points in the second image
    prev_pose : np.ndarray (4, 4) - Previous pose of the camera
    curr_pose : np.ndarray (4, 4) - Current pose of the camera

    Returns:
    -----------
    tuple[np.ndarray, np.ndarray] -
        np.ndarray (N, 3) - 3D points in camera frame
        np.ndarray (N,) - Depths of the points
    """

    # Camera intrinsics and extrinsics
    K = get_camera_intrinsics(cam_name, cam_config)
    rover_T_cam = get_cam_pose_rover(cam_name)
    rover_T_cam_ocv = rover_T_cam.copy()
    rover_T_cam_ocv[:3, :3] = rover_T_cam_ocv[:3, :3] @ CAMERA_TO_OPENCV_PASSIVE

    # Projection matrices
    cam_T_world_0 = inv(prev_pose @ rover_T_cam_ocv)
    cam_T_world_1 = inv(curr_pose @ rover_T_cam_ocv)

    P0 = K @ cam_T_world_0[:3]
    P1 = K @ cam_T_world_1[:3]

    # Triangulate
    points_4d_h = cv2.triangulatePoints(P0, P1, points0.T, points1.T)
    points_3d_est = (points_4d_h[:3] / points_4d_h[3]).T
    depths_est = (cam_T_world_1[:3, :3] @ points_3d_est.T + cam_T_world_1[:3, 3:4]).T[:, 2]

    return points_3d_est, depths_est


ROCK_MIN_SCORE = 0.0
ROCK_MAX_SCORE = 300.0
ROCK_UNCOMPLETED_VALUE = -np.inf


def get_rocks_score(ground_map, agent_map):
    """
    Compare the number of rocks found vs the real ones using an F1 score. Uncompleted values
    will be supposed False, increasing the amount of false negatives.
    """
    if agent_map is None or ground_map is None:
        return ROCK_MIN_SCORE

    true_rocks = ground_map[:, :, 3]
    if np.sum(true_rocks) == 0:
        # Special case, preset has no rocks, disable the
        return ROCK_MIN_SCORE

    agent_rocks = np.copy(agent_map[:, :, 3])
    agent_rocks[agent_rocks == ROCK_UNCOMPLETED_VALUE] = False  # Uncompleted will

    tp = np.sum(np.logical_and(agent_rocks == True, true_rocks == True))
    fp = np.sum(np.logical_and(agent_rocks == True, true_rocks == False))
    fn = np.sum(np.logical_and(agent_rocks == False, true_rocks == True))

    score_rate = (2 * tp) / (2 * tp + fp + fn)
    return ROCK_MAX_SCORE * score_rate
