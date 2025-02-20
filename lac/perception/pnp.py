import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lac.params import TAG_LOCATIONS, TAG_CORNERS_LOCAL
from lac.utils.frames import apply_transform


def get_tag_corners_world(id: int, lander_pose: np.ndarray):
    """
    Prepares the corner points of the tag in the world frame.

    Inputs:
    -------
    tag_id : int
        ID of the tag.
    lander_pose : np.ndarray
        Pose of the lander in the world frame.

    Returns:
    --------
    tag_corners_world : np.ndarray
        3D coordinates of the tag corners in the world frame. Ordering:
        [center, top-left, top-right, bottom-right, bottom-left]

    """
    tag_center = TAG_LOCATIONS[id]["center"]
    tag_bearing = TAG_LOCATIONS[id]["bearing"]
    R_bearing = Rotation.from_euler("z", -tag_bearing, degrees=True)
    tag_corners_lander = R_bearing.apply(TAG_CORNERS_LOCAL) + tag_center
    tag_corners_world = apply_transform(lander_pose, tag_corners_lander)
    return tag_corners_world


def solve_tag_pnp(
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

    rvec_init, _ = cv2.Rodrigues(rot_init) if rot_init is not None else (None, None)
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
        raise RuntimeError("PnP Solution failed")

    # Convert rotation vector to rotation matrix
    cam_rot_world, _ = cv2.Rodrigues(rvec)

    return (
        np.array(cam_rot_world),
        np.array(tvec),
    )
