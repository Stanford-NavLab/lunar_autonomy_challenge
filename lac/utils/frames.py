# Functions for converting between coordinate frames and dealing with transformations

import typing as T

import numpy as np
from scipy.spatial.transform import Rotation as R

from lac.params import GEOMETRY_DICT

# OpenCV uses Z-forward, Y-down, X-right convention
# Rover/Camera uses X-forward, Y-left, Z-up convention
OPENCV_TO_CAMERA_ACTIVE = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
CAMERA_TO_OPENCV_ACTIVE = OPENCV_TO_CAMERA_ACTIVE.T
OPENCV_TO_CAMERA_PASSIVE = OPENCV_TO_CAMERA_ACTIVE.T
CAMERA_TO_OPENCV_PASSIVE = OPENCV_TO_CAMERA_ACTIVE

OPENGL_T_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
OPENCV_T_OPENGL = np.linalg.inv(OPENGL_T_OPENCV)


def make_transform_mat(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 transformation matrix from a translation vector and a rotation matrix.
    """
    transf = np.eye(4)
    transf[:3, :3] = R
    transf[:3, 3] = t
    return transf


def invert_transform_mat(t_mat: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 transformation matrix.
    """
    inv_transf = np.eye(4)
    inv_transf[:3, :3] = t_mat[:3, :3].T
    inv_transf[:3, 3] = -t_mat[:3, :3].T @ t_mat[:3, 3]
    return inv_transf


def get_cam_pose_rover(cam_name: str) -> np.ndarray:
    """
    Returns the camera pose relative to the rover from known camera geometries. The resulting
    4x4 transformation matrix is the active transformation from the rover body-fixed to the camera frame.
    Intrinsic rotation, apply on the right.

    Inputs:
    -------
    cam_name: str
        Name of the camera. e.g., "FrontLeft"

    Returns:
    --------
    rover_T_cam: np.ndarray
        4x4 transformation matrix representing the camera pose relative to the rover body-fixed frame. (Active transformation)

    """
    CAM_POS_KEYS = ["x", "y", "z"]
    CAM_ANG_KEYS = ["roll", "pitch", "yaw"]

    cam = GEOMETRY_DICT["rover"]["cameras"][cam_name]
    rover_t_cam = np.array([cam[key] for key in CAM_POS_KEYS])
    rover_R_cam = R.from_euler("xyz", [cam[key] for key in CAM_ANG_KEYS], degrees=False).as_matrix()
    rover_T_cam = np.eye(4)
    rover_T_cam[:3, :3] = rover_R_cam
    rover_T_cam[:3, 3] = rover_t_cam

    return rover_T_cam


def get_opencv_pose_rover(camera_name: str, cam_geoms: T.Dict[str, T.Any]) -> np.ndarray:
    """
    Defines the specified camera pose in OpenCV convention relative to the rover-fixed frame.
    OpenCV uses Z-forward, Y-down, X-right convention. Assumes that the rover's camera axes follow
    the rover's body-fixed axes (X-forward, Y-left, Z-up). Intrinsic rotation, apply on the right.

    Inputs:
    -------
    camera_name: str
        Name of the camera. e.g., "FrontLeft"
    cam_geoms: dict
        Dictionary containing camera geometries. The keys are camera names and the values are dictionaries with (at least) the following keys:
            - "name": camera name.
            - "x", "y", "z": camera position in the rover body-fixed frame.
            - "roll", "pitch", "yaw": camera orientation relative to the rover body-fixed frame.

    Returns:
    --------
    opencv_cam_pose: np.ndarray
        4x4 transformation matrix representing the camera pose in OpenCV convention relative to the frame of the specified rover camera. (Active transformation)

    """
    cam_T_opencv = np.eye(4)  # Transformation Camera to OpenCV
    cam_T_opencv[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).T

    rover_T_cam = get_cam_pose_rover(camera_name)
    rover_T_opencv = rover_T_cam @ cam_T_opencv  # Active transformation, intrinsic rotation

    return rover_T_opencv


def opencv_to_camera(points: np.ndarray) -> np.ndarray:
    """Convert points from OpenCV frame to camera frame."""
    return points @ OPENCV_TO_CAMERA_ACTIVE.T


def camera_to_opencv(points: np.ndarray) -> np.ndarray:
    """Convert points from camera frame to OpenCV frame."""
    return points @ CAMERA_TO_OPENCV_ACTIVE.T


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transformation matrix to an Nx3 set of points"""
    return points @ T[:3, :3].T + T[:3, 3]


class Trajectory:
    """Sequence of poses with timestamps"""

    def __init__(self):
        self.poses = []
        self.timestamps = []

    def get_positions(self):
        return np.array([pose[:3, 3] for pose in self.poses])

    def get_orientations(self):
        return np.array([pose[:3, :3] for pose in self.poses])


def positions_rmse(traj_a: Trajectory, traj_b: Trajectory):
    """
    Compute the root mean squared error (RMSE) between two trajectories.
    """
    pos_a = traj_a.get_positions()
    pos_b = traj_b.get_positions()
    return np.sqrt(np.mean(np.linalg.norm(pos_a - pos_b, axis=1) ** 2))


def cam_to_world(world_T_rover: np.array, cam_name: str) -> np.ndarray:
    rover_T_cam = get_cam_pose_rover(cam_name)
    rover_T_ocv = rover_T_cam.copy()
    rover_T_ocv[:3, :3] = rover_T_ocv[:3, :3] @ CAMERA_TO_OPENCV_PASSIVE
    ocv_T_world = np.linalg.inv(world_T_rover @ rover_T_ocv)
    return ocv_T_world
