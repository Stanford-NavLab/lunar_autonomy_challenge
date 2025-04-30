"""Depth Estimation"""

import cv2
import numpy as np
from transformers import pipeline
import torch
from PIL import Image
import open3d as o3d
from lac.perception.segmentation_util import (
    get_mask_centroids,
    centroid_matching,
)
from lac.perception.vision import project_pixel_to_3D, project_pixels_to_3D, get_camera_intrinsics
from lac.utils.frames import opencv_to_camera, get_cam_pose_rover, apply_transform
import lac.params as params
import plotly.graph_objects as go

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
        left_rock_centroids[match[0]][0] - right_rock_centroids[match[1]][0] + 1e-8
        for match in matches
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


def stereo_mask_depth_from_segmentation(left_seg_masks, right_seg_masks, baseline, focal_length_x):
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
        left_rock_centroids[match[0]][0] - right_rock_centroids[match[1]][0] + 1e-8
        for match in matches
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


def compute_rock_coords_rover_frame(stereo_depth_results, cam_config, cam_name="FrontLeft"):
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
    return np.array(rock_coords_rover_frame)


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


def project_pixel_to_world(
    rover_pose: np.ndarray, pixel: tuple | np.ndarray, depth: float, cam_name: str, cam_config: dict
):
    CAM_TO_ROVER = get_cam_pose_rover(cam_name)
    K = get_camera_intrinsics(cam_name, cam_config)
    point_opencv = project_pixel_to_3D(pixel, depth, K)
    point_cam = opencv_to_camera(point_opencv)
    point_rover = apply_transform(CAM_TO_ROVER, point_cam)
    point_world = apply_transform(rover_pose, point_rover)
    return point_world


def project_pixels_to_world(
    rover_pose: np.ndarray, pixels: np.ndarray, depths: np.ndarray, cam_name: str, cam_config: dict
):
    CAM_TO_ROVER = get_cam_pose_rover(cam_name)
    K = get_camera_intrinsics(cam_name, cam_config)
    points_opencv = project_pixels_to_3D(pixels, depths, K)
    points_cam = opencv_to_camera(points_opencv)
    points_rover = apply_transform(CAM_TO_ROVER, points_cam)
    points_world = apply_transform(rover_pose, points_rover)
    return points_world


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


RENDERERS = {}


from lac.params import IMG_WIDTH, IMG_HEIGHT, FL_X, FL_Y


def get_renderer():
    H, W = IMG_HEIGHT, IMG_WIDTH
    if RENDERERS.get((H, W)) is not None:
        renderer = RENDERERS[(H, W)]
    else:
        renderer = o3d.visualization.rendering.OffscreenRenderer(width=W, height=H)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.4, 0.4, 0.4, 1.0]  # gray rock color, RGBA
    material.base_roughness = 1.0  # high roughness for rock texture
    material.base_metallic = 0.0  # non-metallic surface
    material.base_reflectance = 0.0  # low reflectivity for natural rocks

    bg_color = np.array([0.0, 0.0, 0.0, 0.0])
    renderer.scene.set_background(bg_color)

    return renderer, material


def render_o3d(meshes, renderer, material, pose, d_light):
    renderer.scene.clear_geometry()
    for i, mesh in enumerate(meshes):
        renderer.scene.add_geometry(f"mesh_{i}", mesh, material)

    renderer.scene.set_lighting(renderer.scene.LightingProfile.HARD_SHADOWS, d_light)
    cam = o3d.camera.PinholeCameraParameters()
    cam.extrinsic = pose
    cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        IMG_WIDTH,
        IMG_HEIGHT,
        FL_X,
        FL_Y,
        IMG_WIDTH / 2,
        IMG_HEIGHT / 2,
    )
    renderer.setup_camera(cam.intrinsic, cam.extrinsic)
    img = np.asarray(renderer.render_to_image())
    depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
    return img, depth


def get_plotly_mesh(
    mesh: o3d.geometry.TriangleMesh, lighting=None, light_position=None, colorscale=None
) -> go.Mesh3d:
    if colorscale is None:
        colorscale = [0, "rgb(153, 153, 153)"], [1.0, "rgb(160,160,160)"]
    if light_position is None:
        light_position = dict(x=1000, y=500, z=2000)
    if lighting is None:
        lighting = dict(
            ambient=0.02,
            diffuse=0.8,
            specular=0.15,
            roughness=0.5,
            fresnel=0.2,
            facenormalsepsilon=1e-15,
            vertexnormalsepsilon=1e-15,
        )

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    mesh_plotly = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        flatshading=True,
        colorscale=colorscale,
        intensity=vertices[:, 0],
        lighting=lighting,
        lightposition=light_position,
        showlegend=False,
    )
    return mesh_plotly


def get_light_direction(az, el):
    return -np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])


def map_to_mesh(map_gt: np.ndarray) -> o3d.geometry.TriangleMesh:
    H, W = map_gt.shape[:2]
    vertices = map_gt[..., :3].reshape(-1, 3)

    triangles = []

    for i in range(H - 1):
        for j in range(W - 1):
            idx0 = i * W + j
            idx1 = idx0 + 1
            idx2 = idx0 + W
            idx3 = idx2 + 1

            # Triangle 1
            triangles.append([idx0, idx2, idx1])
            # Triangle 2
            triangles.append([idx1, idx2, idx3])

    triangles = np.array(triangles)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    return mesh


from thirdparty.raft_stereo.raft_stereo import RAFTStereo
from thirdparty.raft_stereo.utils.utils import InputPadder
from lac.params import LAC_BASE_PATH
import munch
import os
from typing import Union


class DepthEstimator:
    def __init__(
        self,
        **kwargs,
    ):
        config_dict = {
            "restore_ckpt": os.path.join(
                LAC_BASE_PATH, "lunar_autonomy_challenge/models/raftstereo-eth3d.pth"
            ),
            "save_numpy": False,
            "mixed_precision": False,
            "valid_iters": 32,
            "hidden_dims": [128, 128, 128],
            "corr_implementation": "reg",
            "shared_backbone": False,
            "corr_levels": 4,
            "corr_radius": 4,
            "n_downsample": 2,
            "context_norm": "batch",
            "slow_fast_gru": False,
            "n_gru_layers": 3,
            "device": "cuda",
        }
        config_dict.update(kwargs)
        self.config = munch.Munch.fromDict(config_dict)

        self.model = torch.nn.DataParallel(RAFTStereo(self.config), device_ids=[0])
        self.model.load_state_dict(torch.load(self.config.restore_ckpt, weights_only=True))

        self.model = self.model.module
        self.model.to(self.config.device)
        self.model.eval()

    def compute_disparity(
        self, image1: Union[np.ndarray, torch.Tensor], image2: Union[np.ndarray, torch.Tensor]
    ):
        # Convert to torch tensor [C, H, W]
        if isinstance(image1, np.ndarray):
            if image1.ndim == 2:
                image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
            if image2.ndim == 2:
                image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
            image1 = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).to(device)
            image2 = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).to(device)
        elif isinstance(image1, torch.Tensor):
            image1 = image1.unsqueeze(0).to(device)
            image2 = image2.unsqueeze(0).to(device)

        with torch.no_grad():
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = self.model(image1, image2, iters=self.config.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

        return -flow_up.cpu().numpy()


@torch.no_grad()
def align_scale_and_shift(prediction, target, weights):
    """
    weighted least squares problem to solve scale and shift:
        min sum{
                  weight[i,j] *
                  (prediction[i,j] * scale + shift - target[i,j])^2
               }

    prediction: [B,H,W]
    target: [B,H,W]
    weights: [B,H,W]
    """
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction).to(device)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).to(device)
    if isinstance(weights, np.ndarray):
        weights = torch.from_numpy(weights).to(device)

    if weights is None:
        weights = torch.ones_like(prediction).to(prediction.device)
    if len(prediction.shape) < 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        weights = weights.unsqueeze(0)
    a_00 = torch.nansum(weights * prediction * prediction, dim=[1, 2])
    a_01 = torch.nansum(weights * prediction, dim=[1, 2])
    a_11 = torch.nansum(weights, dim=[1, 2])
    # right hand side: b = [b_0, b_1]
    b_0 = torch.nansum(weights * prediction * target, dim=[1, 2])
    b_1 = torch.nansum(weights * target, dim=[1, 2])
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    det = a_00 * a_11 - a_01 * a_01
    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det
    error = (scale[:, None, None] * prediction + shift[:, None, None] - target).abs()
    masked_error = error * weights
    error_sum = masked_error.nansum(dim=[1, 2])
    error_num = weights.nansum(dim=[1, 2])
    avg_error = error_sum / error_num

    return scale, shift, avg_error
