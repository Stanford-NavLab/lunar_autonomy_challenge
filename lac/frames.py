# Functions for converting between coordinate frames and dealing with transformations

import numpy as np
from scipy.spatial.transform import Rotation as R
import typing as T

def invert_transform_mat(t_mat: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 transformation matrix.
    """
    inv_transf = np.eye(4)
    inv_transf[:3, :3] = t_mat[:3, :3].T
    inv_transf[:3, 3] = -t_mat[:3, :3].T @ t_mat[:3, 3]
    return inv_transf

def get_cam_pose_rover(cam_name: str, cam_geoms: T.Dict[str, T.Any]) -> np.ndarray:
    """
    Returns the camera pose relative to the rover from known camera geometries. The resulting 4x4 transformation matrix is the active transformation from the rover body-fixed to the camera frame.
    
    Inputs:
    -------
    cam_name: str
        Name of the camera. e.g., "FrontLeft"
    cam_geoms: dict
        Dictionary containing camera geometries. The keys are camera names and the values are dictionaries with (at least) the following keys:
            - "name": camera name.
            - "x", "y", "z": camera position in the rover body-fixed frame.
            - "roll", "pitch", "yaw": camera orientation relative to the rover body-fixed frame.
    
    Returns:
    --------
    cam_pose: np.ndarray
        4x4 transformation matrix representing the camera pose relative to the rover body-fixed frame.
    """

    CAM_POS_KEYS = ["x", "y", "z"]
    CAM_ANG_KEYS = ["roll", "pitch", "yaw"]

    cam = cam_geoms[cam_name]
    rover_t_cam = np.array([cam[key] for key in CAM_POS_KEYS])
    rover_r_cam = R.from_euler("xyz", [cam[key] for key in CAM_ANG_KEYS], degrees=False).as_matrix()
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = rover_r_cam
    cam_pose[:3, 3] = rover_t_cam

    return cam_pose