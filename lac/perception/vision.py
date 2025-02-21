"""Computer vision utilities"""

import cv2 as cv
import numpy as np
import apriltag

from lac.perception.pnp import solve_tag_pnp
from lac.utils.frames import invert_transform_mat, get_cam_pose_rover


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
    def __init__(self):
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

        cam_poses = solve_tag_pnp(detections, lander_pose)
        rover_to_cam = get_cam_pose_rover(cam_name)
        cam_to_rover = invert_transform_mat(rover_to_cam)

        return [cam_pose @ cam_to_rover for cam_pose in cam_poses]
