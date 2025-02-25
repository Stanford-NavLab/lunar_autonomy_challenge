"""Computer vision utilities"""

import cv2 as cv
import numpy as np
import apriltag

from lac.perception.pnp import solve_tag_pnp
from lac.utils.frames import invert_transform_mat, get_cam_pose_rover
from lac.params import IMG_FOV_RAD


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
        """
        detections = self.detect(img)
        cam_intrisics = get_camera_intrinsics(cam_name, self.camera_config)
        cam_poses = solve_tag_pnp(detections, cam_intrisics, lander_pose)
        rover_to_cam = get_cam_pose_rover(cam_name)
        cam_to_rover = invert_transform_mat(rover_to_cam)

        return {tag_id: cam_pose @ cam_to_rover for tag_id, cam_pose in cam_poses.items()}
