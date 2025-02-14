import cv2
import numpy as np
from transformers import pipeline
import torch
from PIL import Image

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


def depth_from_stereo_segmentation(left_centroids, right_centroids):
    """
    left_centroids : np.ndarray (N, 2) - Centroids from the left image
    right_centroids : np.ndarray (N, 2) - Centroids from the right image
    """
    # Sort the centroids by y-coordinate
    left_centroids = left_centroids[np.argsort(left_centroids[:, 1])]
    right_centroids = right_centroids[np.argsort(right_centroids[:, 1])]

    # Compute the disparity between the centroids
    disparities = right_centroids - left_centroids

    # Compute the depth from the disparity
    baseline = 0.1  # meters
    focal_length_x = 500  # pixels
    disparities[disparities == 0] = 0.1  # Avoid division by zero
    depths = (focal_length_x * baseline) / disparities

    return depths


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
