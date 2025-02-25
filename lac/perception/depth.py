"""Depth Estimation"""

import cv2
import numpy as np
from transformers import pipeline
import torch
from PIL import Image

from lac.perception.segmentation import get_mask_centroids, centroid_matching
from lac.perception.vision import project_pixel_to_3D, get_camera_intrinsics
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
    disparities = [match[0][0] - match[1][0] for match in matches]
    depths = (focal_length_x * baseline) / disparities

    results = []
    for i, match in enumerate(matches):
        if depths[i] > 0 and depths[i] < params.MAX_DEPTH:
            results.append(
                {
                    "left_centroid": match[0],
                    "right_centroid": match[1],
                    "disparity": disparities[i],
                    "depth": depths[i],
                }
            )
    return results


def project_depths_to_world(
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
    world_points = []
    # offset from rover origin to front left camera in rover frame (TODO: load this)
    FL_OFFSET = np.array([0.28, 0.081, 0.131])
    CAM_TO_ROVER = get_cam_pose_rover(cam_name)
    K = get_camera_intrinsics(cam_name, cam_config)
    for result in depth_results:
        rock_point_opencv = project_pixel_to_3D(result["left_centroid"], result["depth"], K)
        # OpenCV to camera frame conversion
        rock_point_cam = opencv_to_camera(rock_point_opencv)
        # Apply camera to rover offset
        # rock_point_rover = rock_point_cam + FL_OFFSET  # TODO: generalize this
        rock_point_rover = apply_transform(CAM_TO_ROVER, rock_point_cam)
        # Rover to world frame conversion
        rock_point_world = (rover_pose @ np.concatenate((rock_point_rover, [1])))[:3]
        world_points.append(rock_point_world)
    return world_points


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
