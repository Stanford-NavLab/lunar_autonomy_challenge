import numpy as np
from scipy.spatial.transform import Rotation


def transform_to_numpy(transform):
    """Convert a Transform object to a 4x4 pose matrix.

    The resulting pose matrix has +X forward, +Y left, +Z up.

    """
    t = np.array([transform.location.x, transform.location.y, transform.location.z])
    euler_angles = np.array(
        [transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw]
    )
    # Negate the angles since they are for clockwise rotation
    R = Rotation.from_euler("xyz", -euler_angles).as_matrix()
    # print("determinant of R:", np.linalg.det(R))
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
    R_blender = np.array([-ry, rz, -rx])
    return np.block([[R_blender, t[:, None]], [0, 0, 0, 1]])
