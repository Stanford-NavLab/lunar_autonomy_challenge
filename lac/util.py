import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import cv2 as cv


def transform_to_numpy(transform):
    """Convert a Transform object to a 4x4 pose matrix.

    The resulting pose matrix has +X forward, +Y left, +Z up.

    """
    t = np.array([transform.location.x, transform.location.y, transform.location.z])
    euler_angles = np.array(
        [transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw]
    )
    R = Rotation.from_euler("xyz", euler_angles).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
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


def pose_to_rpy_pos(pose):
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

    return rpy, pos


def skew_symmetric(v):
    """Convert a 3D vector to a skew-symmetric matrix."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def normalize_rotation_matrix(R):
    """Normalizes a rotation matrix using SVD."""
    U, _, Vt = np.linalg.svd(R)  # Singular Value Decomposition
    R_normalized = U @ Vt  # Reconstruct a valid rotation matrix
    return R_normalized


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def wrap_angle(angle):
    """Wrap an angle in radians to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


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
