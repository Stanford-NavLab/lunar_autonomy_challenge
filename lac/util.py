import json

import cv2 as cv
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def load_data(data_path):
    """Load data from data log file."""
    json_data = json.load(open(f"{data_path}/data_log.json"))
    initial_pose = np.array(json_data["initial_pose"])
    lander_pose = np.array(json_data["lander_pose_world"])
    cam_config = json_data["cameras"]

    poses = [initial_pose]
    imu_data = []
    for frame in json_data["frames"]:
        poses.append(np.array(frame["pose"]))
        imu_data.append(np.array(frame["imu"]))
    imu_data = np.array(imu_data)

    return initial_pose, lander_pose, poses, imu_data, cam_config


def transform_to_pos_rpy(transform):
    """Convert a Transform object to roll-pitch-yaw euler angles and position.
    Euler angles are in radians.

    """
    t = np.array([transform.location.x, transform.location.y, transform.location.z])
    rpy = np.array([transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw])
    return t, rpy


def transform_to_numpy(transform):
    """Convert a Transform object to a 4x4 pose matrix.

    The resulting pose matrix has +X forward, +Y left, +Z up.

    """
    t, euler_angles = transform_to_pos_rpy(transform)
    R = Rotation.from_euler("xyz", euler_angles).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pos_rpy_to_pose(pos, rpy):
    """Convert a position and rpy to a 4x4 pose matrix."""
    R = Rotation.from_euler("xyz", rpy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def to_blender_convention(pose):
    """Convert a camera pose matrix to Blender convention.

    The camera pose matrix is assumed to have above starting convention (+X forward, +Y left, +Z up)
    The Blender convention has -Z forward, +X right, +Y up.

    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    # Convert the rotation matrix to the Blender convention
    rx, ry, rz = R[:, 0], R[:, 1], R[:, 2]
    R_blender = np.array([-ry, rz, -rx]).T
    return np.block([[R_blender, t[:, None]], [0, 0, 0, 1]])


def pose_to_pos_rpy(pose):
    """Convert a camera pose matrix to LAC convention.

    The camera pose matrix is assumed to have above starting convention (+X forward, +Y left, +Z up)
    The LAC convention has +X forward, +Y left, +Z up.

    """
    R = pose[:3, :3]
    t = pose[:3, 3]

    # Calculate yaw, pitch, roll using scipy Rotation
    r = Rotation.from_matrix(R)
    roll, pitch, yaw = r.as_euler("xyz")

    pos = np.array([t[0], t[1], t[2]])
    rpy = np.array([roll, pitch, yaw])

    return pos, rpy


def skew_symmetric(v):
    """Convert a 3D vector to a skew-symmetric matrix."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def normalize_rotation_matrix(R):
    """Normalizes a rotation matrix using SVD."""
    U, _, Vt = np.linalg.svd(R)  # Singular Value Decomposition
    R_normalized = U @ Vt  # Reconstruct a valid rotation matrix
    return R_normalized


def wrap_angle(angle):
    """Wrap an angle in radians to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def rotation_matrix_error(R1, R2):
    R_error = R1 @ R2.T  # Compute relative rotation matrix
    trace_value = np.trace(R_error)

    # Ensure the input to arccos is within the valid range [-1, 1]
    cos_theta = np.clip((trace_value - 1) / 2, -1.0, 1.0)

    theta = np.arccos(cos_theta)  # Rotation angle in radians
    return np.degrees(theta)  # Convert to degrees


def rotations_rmse(R1, R2):
    """Compute the RMSE between two sets of rotation matrices."""
    errors = np.zeros(len(R1))
    for i in range(len(R1)):
        errors[i] = rotation_matrix_error(R1[i], R2[i])
    return np.sqrt(np.mean(errors**2))


def np_img_to_PIL_rgb(img_array):
    """Convert a numpy image array to a PIL image."""
    return Image.fromarray(img_array).convert("RGB")


def mask_centroid(mask: np.ndarray) -> tuple:
    """Compute the centroid of a binary mask."""
    M = cv.moments(mask)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def gen_square_spiral(max_val, min_val, step):
    """
    Generate an Nx2 numpy array of 2D coordinates following a square spiral.

    Parameters:
      max_val (float): The half-side length of the outermost square.
      min_val (float): The half-side length of the innermost square.
      step (float): The decrement between successive squares.

    Returns:
      np.array: An (N x 2) numpy array containing the 2D coordinates.
    """
    points = []
    r = max_val
    # Use a small tolerance to account for floating point comparisons.
    while r >= min_val - 1e-8:
        # Order: top-left, top-right, bottom-right, bottom-left.
        points.append([-r, r])  # top-left
        points.append([r, r])  # top-right
        points.append([r, -r])  # bottom-right
        points.append([-r, -r])  # bottom-left
        r -= step
    return np.array(points)


def get_positions_from_poses(poses):
    return np.array([pose[:3, 3] for pose in poses])


def get_rotations_from_poses(poses):
    return np.array([pose[:3, :3] for pose in poses])


def positions_rmse_from_poses(poses_a, poses_b):
    pos_a = get_positions_from_poses(poses_a)
    pos_b = get_positions_from_poses(poses_b)
    return np.sqrt(np.mean(np.linalg.norm(pos_a - pos_b, axis=1) ** 2))


def rotations_rmse_from_poses(poses_a, poses_b):
    rots_a = get_rotations_from_poses(poses_a)
    rots_b = get_rotations_from_poses(poses_b)
    return rotations_rmse(rots_a, rots_b)
