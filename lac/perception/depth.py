"""Depth Estimation"""

import cv2
import numpy as np
from transformers import pipeline
import torch
from PIL import Image

from lac.perception.segmentation_util import (
    get_mask_centroids,
    centroid_matching,
)
from lac.perception.vision import project_pixel_to_3D, project_pixels_to_3D, get_camera_intrinsics
from lac.utils.frames import opencv_to_camera, get_cam_pose_rover, apply_transform
import lac.params as params

device = "cuda" if torch.cuda.is_available() else "cpu"


class DepthAnything:
    def __init__(self):
        checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
        self.pipe = pipeline(
            "depth-estimation",
            model=checkpoint,
            device=device,
            model_kwargs={"torch_dtype": torch.float32},
        )
        self.pipe.model.to(torch.float32)

    def predict_depth(self, image: Image):
        """
        image : RGB PIL Image
        """
        predictions = self.pipe(image)
        return predictions["depth"]


def stereo_depth_from_segmentation(left_seg_masks, right_seg_masks, baseline, focal_length_x):
    """
    left_seg_results : dict - Results from the segmentation model for the left image
    right_seg_results : dict - Results from the segmentation model for the right image
    baseline : float - Stereo baseline in meters
    focal_length_x : float - Horizontal focal length in pixels
    """
    left_rock_centroids = get_mask_centroids(left_seg_masks)
    right_rock_centroids = get_mask_centroids(right_seg_masks)

    if len(left_rock_centroids) == 0 or len(right_rock_centroids) == 0:
        return []

    matches = centroid_matching(left_rock_centroids, right_rock_centroids)

    # Since we compute disparity as x_left - x_right, the computed depth is with respect to the left camera
    disparities = [
        left_rock_centroids[match[0]][0] - right_rock_centroids[match[1]][0] for match in matches
    ]
    depths = (focal_length_x * baseline) / disparities

    results = []
    for i, match in enumerate(matches):
        if depths[i] > 0 and depths[i] < params.MAX_DEPTH:
            results.append(
                {
                    "left_centroid": left_rock_centroids[match[0]],
                    "left_mask": left_seg_masks[match[0]],
                    "right_centroid": right_rock_centroids[match[1]],
                    "right_mask": right_seg_masks[match[1]],
                    "disparity": disparities[i],
                    "depth": depths[i],
                }
            )
    return results


def compute_rock_coords_rover_frame(stereo_depth_results, cam_name, cam_config):
    """
    stereo_depth_results : list - List of dictionaries containing stereo depth results
    cam_name : str - Name of the camera
    cam_config : dict - Camera configuration dictionary

    Returns:
    list - List of rock points in the rover frame
    """
    rock_coords_rover_frame = []
    for result in stereo_depth_results:
        rock_point_rover_frame = project_pixel_to_rover(
            result["left_centroid"], result["depth"], cam_name, cam_config
        )
        rock_coords_rover_frame.append(rock_point_rover_frame)
    return rock_coords_rover_frame


def compute_rock_radii(stereo_depth_results):
    """
    Computes radii of detected rocks based on depth and bounding box width.

    Args:
        results (list): A list of dictionaries containing rock detection masks and depth values.
        params (object): An object containing camera parameters, specifically `FL_X` (focal length in pixels).

    Returns:
        list: A list of tuples [(x, y, radius), ...] representing the coordinates and estimated radius of each rock.
    """
    rock_radii = []

    for result in stereo_depth_results:
        # Convert mask to uint8 and get bounding box
        mask_uint8 = result["left_mask"].astype(np.uint8)
        x, y, w, h = cv2.boundingRect(mask_uint8)

        # Compute real-world width and radius using depth and focal length
        width_real = w * result["depth"] / params.FL_X
        radius_real = width_real / 2  # Approximate the radius

        rock_radii.append(radius_real)
    return rock_radii


def project_rock_depths_to_world(
    depth_results: list, rover_pose: np.ndarray, cam_name: str, cam_config: dict
) -> list:
    """
    depth_results : list - List of dictionaries containing depth results
    K : np.ndarray (3, 3) - Camera intrinsics matrix
    rover_pose : np.ndarray (4, 4) - Rover pose in the world frame

    Returns:
    list - List of world points

    TODO: vectorize this
    """
    rock_world_points = []
    CAM_TO_ROVER = get_cam_pose_rover(cam_name)
    K = get_camera_intrinsics(cam_name, cam_config)
    for result in depth_results:
        rock_point_opencv = project_pixel_to_3D(result["left_centroid"], result["depth"], K)
        # OpenCV to camera frame conversion
        rock_point_cam = opencv_to_camera(rock_point_opencv)
        # Apply camera to rover offset
        rock_point_rover = apply_transform(CAM_TO_ROVER, rock_point_cam)
        # Rover to world frame conversion
        rock_point_world = apply_transform(rover_pose, rock_point_rover)
        rock_world_points.append(rock_point_world)
    return rock_world_points


def project_pixel_to_rover(
    pixel: tuple | np.ndarray, depth: float, cam_name: str, cam_config: dict
):
    CAM_TO_ROVER = get_cam_pose_rover(cam_name)
    K = get_camera_intrinsics(cam_name, cam_config)
    point_opencv = project_pixel_to_3D(pixel, depth, K)
    point_cam = opencv_to_camera(point_opencv)
    point_rover = apply_transform(CAM_TO_ROVER, point_cam)
    return point_rover


def project_pixels_to_rover(
    pixels: np.ndarray, depths: np.ndarray, cam_name: str, cam_config: dict
):
    CAM_TO_ROVER = get_cam_pose_rover(cam_name)
    K = get_camera_intrinsics(cam_name, cam_config)
    points_opencv = project_pixels_to_3D(pixels, depths, K)
    points_cam = opencv_to_camera(points_opencv)
    points_rover = apply_transform(CAM_TO_ROVER, points_cam)
    return points_rover


def compute_stereo_depth(
    img_left: np.ndarray,
    img_right: np.ndarray,
    baseline: float,
    focal_length_x: float,
    semi_global: bool = False,
):
    """
    img_left: np.ndarray (H, W) - Grayscale left image
    img_right: np.ndarray (H, W) - Grayscale right image
    baseline: float - Stereo baseline in meters
    focal_length_x: float - Horizontal focal length in pixels
    """
    # Create a StereoBM object (you can also use StereoSGBM for better results)
    min_disparity = 0
    num_disparities = 64  # Should be divisible by 16
    block_size = 15

    if semi_global:
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )
    else:
        stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    # Compute the disparity map
    disparity = stereo.compute(img_left, img_right)

    # Normalize the disparity for visualization
    disparity_normalized = cv2.normalize(
        disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    disparity_normalized = np.uint8(disparity_normalized)

    # Convert disparity to depth (requires camera calibration parameters)
    # Assuming known focal length (f) and baseline (b) of the stereo setup
    disparity[disparity == 0] = 0.1  # Avoid division by zero
    depth = (focal_length_x * baseline) / disparity

    return disparity, depth
