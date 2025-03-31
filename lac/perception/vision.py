"""Computer vision utilities"""

import cv2
import numpy as np
import apriltag
import torch

from lightglue import LightGlue, SuperPoint, match_pair
from lightglue.utils import load_image, rbd

from lac.perception.pnp import solve_tag_pnp
from lac.utils.frames import invert_transform_mat, get_cam_pose_rover
from lac.params import IMG_FOV_RAD


def grayscale_to_3ch_tensor(np_image):
    # Ensure the input is float32 (or float64 if needed)
    np_image = np_image.astype(np.float32) / 255.0 if np_image.max() > 1 else np_image
    # Add channel dimension and repeat across 3 channels
    torch_tensor = torch.from_numpy(np_image).unsqueeze(0).repeat(3, 1, 1)
    return torch_tensor


class LightGlueMatcher:
    def __init__(self):
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features="superpoint").eval().cuda()  # load the matcher

    def match(self, img0, img1):
        """
        Match keypoints between two images
        """
        image0 = grayscale_to_3ch_tensor(img0).cuda()
        image1 = grayscale_to_3ch_tensor(img1).cuda()
        feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, image0, image1)
        return feats0, feats1, matches01


class StereoVIO:
    def __init__(self, fl_x, baseline):
        self.matcher = LightGlueMatcher()
        self.fl_x = fl_x  # Focal length in x direction
        self.baseline = baseline  # Stereo baseline
        self.prev_feats = None
        self.prev_depths = None

    def process_stereo_pair(self, left_img, right_img):
        # Match the stereo pair to get depths
        feats0, feats1, matches01 = self.matcher.match(left_img, right_img)

        # Extract matched points
        points0 = feats0["keypoints"][matches01["matches"][:, 0]]
        points1 = feats1["keypoints"][matches01["matches"][:, 1]]

        # Calculate disparities and depths
        disparities = (points0 - points1)[:, 0]  # X-coordinates only
        depths = self.fl_x * self.baseline / disparities

        # Store features and depths for frame-to-frame tracking
        self.prev_feats = points0
        self.prev_depths = depths

    def track_frame(self, new_left_img, K):
        if self.prev_feats is None:
            raise ValueError("No previous frame to track!")

        # Match previous left frame with current left frame
        new_feats, _, matches = self.matcher.match(self.prev_feats, new_left_img)

        # Extract matched keypoints
        prev_pts = self.prev_feats[matches["matches"][:, 0]]
        new_pts = new_feats["keypoints"][matches["matches"][:, 1]]

        # Convert prev_pts to 3D points using the depths
        prev_depths = self.prev_depths[matches["matches"][:, 0]]
        prev_pts_h = np.hstack([prev_pts, np.ones((prev_pts.shape[0], 1))])
        prev_pts_3d = (np.linalg.inv(K) @ prev_pts_h.T).T * prev_depths[:, None]

        # Estimate pose with PnP
        _, rvec, tvec, inliers = cv2.solvePnPRansac(prev_pts_3d, new_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)

        return rvec, tvec


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
        rover_pose_estimates = {tag_id: cam_pose @ cam_to_rover for tag_id, cam_pose in cam_poses.items()}

        return rover_pose_estimates, detections
