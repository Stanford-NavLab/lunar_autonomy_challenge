import cv2
import numpy as np


def get_tag_corners_world(tag_center: np.ndarray, tag_size: float=0.339):
    """
    Prepares the corner points of the tag in the world frame.
    
    Inputs:
    -------
    tag_center: np.ndarray
        3D coordinates of the tag center on the lander.
    tag_size: float
        Size of the tag. (Assumed to be square)

    Returns:
    --------
    world_points: np.ndarray
        3D coordinates of the tag corners in the world frame. Ordering: 
        [center, top-left, top-right, bottom-right, bottom-left]

    """
    half_size = tag_size / 2
    world_points = np.array(
        [
            tag_center,
            tag_center + np.array([-half_size, half_size, 0]),
            tag_center + np.array([half_size, half_size, 0]),
            tag_center + np.array([half_size, -half_size, 0]),
            tag_center + np.array([-half_size, -half_size, 0]),
        ],
        dtype=np.float32,
    )
    return world_points


def solve_pnp(
    tag_corners_world: np.ndarray,
    detected_tag_corners: np.ndarray,
    tag_size: float,
    tag_center: np.ndarray,
    cam_intrinsics: np.ndarray,
    rot_init: np.ndarray = None,
    tvec_init: np.ndarray = None,
):
    """
    Solves the PnP problem to estimate the camera pose relative to the tag.

    Inputs:
    -------
    detected_tag_corners: np.ndarray
        2D coordinates of the detected tag corners in the image. Ordering:
        [center, top-left, top-right, bottom-right, bottom-left]
    tag_size: float
        Size of the tag. (Assumed to be square)
    tag_center: np.ndarray
        3D coordinates of the tag center on the lander. 
    cam_intrinsics: np.ndarray
        Camera intrinsics matrix.
    rot_init: np.ndarray
        Initial guess for the rotation matrix.
    tvec_init: np.ndarray
        Initial guess for the translation vector.

    Returns:
    --------
    cam_rot_world: np.ndarray
        Rotation from the world frame to the camera frame (assumed OpenCV convention).
    tvec: np.ndarray
        Translation vector representing the camera pose relative to the tag.

    """

    rvec_init, _ = (
        cv2.Rodrigues(rot_init) if rot_init is not None else (None, None)
    )
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=tag_corners_world,
        imagePoints=detected_tag_corners,
        cameraMatrix=cam_intrinsics,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        rvec=rvec_init,
        tvec=tvec_init,
    )

    if not success:
        raise RuntimeError(f"PnP Solution failed for frame {i}")

    # Convert rotation vector to rotation matrix
    cam_rot_world, _ = cv2.Rodrigues(rvec)

    return (
        np.array(cam_rot_world),
        np.array(tvec),
    )
