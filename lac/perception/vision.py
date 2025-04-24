"""Computer vision utilities"""

import cv2
import numpy as np
import apriltag
import torch

from lightglue import LightGlue, SuperPoint, match_pair
from lightglue.utils import load_image, rbd

from lac.perception.pnp import solve_tag_pnp
from lac.utils.frames import (
    make_transform_mat,
    invert_transform_mat,
    get_cam_pose_rover,
    OPENCV_TO_CAMERA_PASSIVE,
)
from lac.params import IMG_FOV_RAD, CAMERA_INTRINSICS


def grayscale_to_3ch_tensor(np_image):
    # Ensure the input is float32 (or float64 if needed)
    np_image = np_image.astype(np.float32) / 255.0 if np_image.max() > 1 else np_image
    # Add channel dimension and repeat across 3 channels
    torch_tensor = torch.from_numpy(np_image).unsqueeze(0).repeat(3, 1, 1)
    return torch_tensor


def get_camera_intrinsics(cam_name: str, camera_config: dict):
    """
    Get camera intrinsics matrix from camera configuration

    cam_name : str - Name of the camera
    camera_config : dict - Camera configuration dictionary

    Returns:
    np.ndarray (3, 3) - Camera intrinsics matrix
    """
    w, h = camera_config[cam_name]["width"], camera_config[cam_name]["height"]
    return calc_camera_intrinsics(w, h)


def calc_camera_intrinsics(w: int, h: int):
    """
    Get camera intrinsics matrix from image dimensions

    w : int - Image width
    h : int - Image height

    Returns:
    np.ndarray (3, 3) - Camera intrinsics matrix
    """
    fx = w / (2 * np.tan(IMG_FOV_RAD / 2))
    fy = fx  # Assuming square pixels
    cx = w / 2
    cy = h / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def project_pixel_to_3D(pixel, depth, K):
    """
    Project a pixel to 3D using the depth map and camera intrinsics

    pixel : tuple - (x, y) pixel coordinates
    depth : float - depth value at the pixel
    K : np.ndarray (3, 3) - Camera intrinsics matrix

    Returns:
    np.ndarray (3,) - 3D point in camera frame
    """
    x, y = pixel
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])


def project_pixels_to_3D(pixels, depths, K):
    """
    Batch version of projecting pixels to 3D using depth and camera intrinsics.

    pixels : np.ndarray (N, 2) - Array of pixel coordinates (x, y)
    depths : np.ndarray (N,) - Array of depth values corresponding to each pixel
    K : np.ndarray (3, 3) - Camera intrinsics matrix

    Returns:
    np.ndarray (N, 3) - 3D points in camera frame for each pixel
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pixel_offset = pixels - np.array([cx, cy])
    X = pixel_offset[:, 0] * depths / fx
    Y = pixel_offset[:, 1] * depths / fy
    Z = depths
    return np.column_stack((X, Y, Z))


PNP_REPROJECTION_ERROR_THRESHOLD = 1.0  # pixels


def solve_vision_pnp(
    points3D: np.ndarray,
    points2D: np.ndarray,
    K: np.ndarray = CAMERA_INTRINSICS,
    cam_name: str = "FrontLeft",
) -> np.ndarray | None:
    """
    Solve the PnP problem to estimate the camera pose from 3D-2D point correspondences.

    NOTE: we apply to cam to rover transform to return rover pose by default

    points3D : np.ndarray (N, 3) - 3D points in world/local frame
    points2D : np.ndarray (N, 2) - 2D points in image frame
    K : np.ndarray (3, 3) - Camera intrinsics matrix

    Returns:
    np.ndarray (4, 4) - Estimated rover pose in world/local frame
    """
    if len(points3D) < 4:
        print("Not enough points to solve PnP.")
        return None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=points3D,
        imagePoints=points2D,
        cameraMatrix=K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=PNP_REPROJECTION_ERROR_THRESHOLD,
        iterationsCount=500,
        confidence=0.99,
    )
    if success:
        R, _ = cv2.Rodrigues(rvec)
        w_T_c = invert_transform_mat(make_transform_mat(R, tvec))
        w_T_c[:3, :3] = w_T_c[:3, :3] @ OPENCV_TO_CAMERA_PASSIVE
        rover_pose = w_T_c @ invert_transform_mat(get_cam_pose_rover(cam_name))
        return rover_pose
    else:
        print("PnP solve failed.")
        return None


class FiducialLocalizer:
    def __init__(self, camera_config: dict):
        self.camera_config = camera_config
        options = apriltag.DetectorOptions(families="tag36h11")
        self.detector = apriltag.Detector(options)

    def detect(self, img: np.ndarray):
        """
        Inputs:
        -------
        img : np.ndarray (H, W)
            Grayscale image

        Returns:
        --------
        detections : list of apriltag.Detection
            AprilTag detections
        """
        return self.detector.detect(img)

    def estimate_rover_pose(self, img: np.ndarray, cam_name: str, lander_pose: np.ndarray):
        """
        Given a camera image, compute a rover pose estimate from each fiducial detection

        Inputs:
        -------
        img : np.ndarray (H, W)
            Grayscale image
        cam_name : str
            Name of the camera
        lander_pose : np.ndarray (4, 4)
            Pose of the lander in the world frame

        Returns:
        --------
        rover_pose_estimates : list of np.ndarray (4, 4)
            List of estimated rover poses in the world frame
        detections : list of apriltag detections
        """
        detections = self.detect(img)
        cam_intrisics = get_camera_intrinsics(cam_name, self.camera_config)
        cam_poses = solve_tag_pnp(detections, cam_intrisics, lander_pose)
        rover_to_cam = get_cam_pose_rover(cam_name)
        cam_to_rover = invert_transform_mat(rover_to_cam)
        rover_pose_estimates = {
            tag_id: cam_pose @ cam_to_rover for tag_id, cam_pose in cam_poses.items()
        }

        return rover_pose_estimates, detections
