import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lac.params import TAG_LOCATIONS, CAMERA_INTRINSICS
from lac.utils.frames import apply_transform, invert_transform_mat, OPENCV_TO_CAMERA_PASSIVE


def get_tag_corners_local(size: float):
    """
    Prepares the corner points of the tag in the local frame.

    Inputs:
    -------
    size : float
        Size of the tag (side length dimension of the black area).

    Returns:
    --------
    tag_corners_local : np.ndarray
        3D coordinates of the tag corners in the local tag frame. Ordering:
        [top-left, top-right, bottom-right, bottom-left]

    """
    return np.array(
        [
            [size / 2, 0.0, size / 2],
            [-size / 2, 0.0, size / 2],
            [-size / 2, 0.0, -size / 2],
            [size / 2, 0.0, -size / 2],
        ]
    )


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
        [top-left, top-right, bottom-right, bottom-left]

    """
    tag_center = TAG_LOCATIONS[id]["center"]
    tag_bearing = TAG_LOCATIONS[id]["bearing"]
    tag_corners_local = get_tag_corners_local(TAG_LOCATIONS[id]["size"])
    R_bearing = Rotation.from_euler("z", -tag_bearing, degrees=True)
    tag_corners_lander = R_bearing.apply(tag_corners_local) + tag_center
    tag_corners_world = apply_transform(lander_pose, tag_corners_lander)
    return tag_corners_world


def solve_tag_pnp(detections: list, lander_pose: np.ndarray):
    """
    Solves the PnP problem to estimate the camera pose in world frame
    NOTE: currently solves for each individual tag detection in a group, rather than stacking them
          together and solving for one pose.

    Inputs:
    -------
    detections : list of apriltag.Detection - AprilTag detections

    Returns:
    --------
    transforms: list
        List of camera pose estimates in world frame

    """
    transforms = []

    for detection in detections:
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=get_tag_corners_world(detection.tag_id, lander_pose),
            imagePoints=detection.corners,
            cameraMatrix=CAMERA_INTRINSICS,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            print("PnP Solution failed on tag ID:", detection.tag_id)
            continue

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()  # world to opencv active

        w_T_c = invert_transform_mat(T)  # world to opencv passive
        w_T_c[:3, :3] = w_T_c[:3, :3] @ OPENCV_TO_CAMERA_PASSIVE  # world to camera passive
        transforms.append(w_T_c)

    return transforms
