import json
import torch
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from pathlib import Path
import os
from tqdm import tqdm
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing


def load_data(data_path: str | Path, dynamics=False):
    """Load data from data log file."""
    if dynamics:
        json_data = json.load(open(f"{data_path}"))
    else:
        json_data = json.load(open(f"{data_path}/data_log.json"))
    initial_pose = np.array(json_data["initial_pose"])
    lander_pose = np.array(json_data["lander_pose_world"])
    cam_config = json_data["cameras"]

    poses = [initial_pose]
    waypoints = []
    imu_data = []
    for frame in json_data["frames"]:
        poses.append(np.array(frame["pose"]))
        imu_data.append(np.array(frame["imu"]))
        if "waypoint" in frame:
            waypoints.append(np.array(frame["waypoint"]))
    imu_data = np.array(imu_data)

    if len(waypoints) > 0:
        return initial_pose, lander_pose, poses, imu_data, cam_config, waypoints
    return initial_pose, lander_pose, poses, imu_data, cam_config


def _load_image(img_path: str | Path, frame: int):
    return frame, cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)


def load_images(
    data_path: str | Path,
    cameras: list[str],
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    step: int = 1,
):
    """Load images from set of cameras in data log file.

    TODO: fix this

    """
    data_path = Path(data_path)
    all_frames = [int(img.split(".")[0]) for img in os.listdir(data_path / cameras[0])]

    if start_frame is None:
        start_frame = min(all_frames)
    if end_frame is None:
        end_frame = max(all_frames)

    camera_files = {
        cam: [
            img
            for img in os.listdir(data_path / cam)
            if int(img.split(".")[0]) % step == 0
            and start_frame <= int(img.split(".")[0]) <= end_frame
        ]
        for cam in cameras
    }

    with ThreadPoolExecutor() as executor:
        results = {}
        for cam in cameras:
            results[cam] = executor.map(
                lambda img: _load_image(data_path / cam / img, int(img.split(".")[0])),
                tqdm(camera_files[cam], desc=cam),
            )

    for cam in cameras:
        results[cam] = dict(results[cam])

    return results


def load_stereo_images(
    data_path: str | Path,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    step: int = 1,
):
    """Load stereo images from data log file."""
    data_path = Path(data_path)
    front_left_path = data_path / "FrontLeft"
    front_right_path = data_path / "FrontRight"

    all_frames = [int(img.split(".")[0]) for img in os.listdir(front_left_path)]

    if start_frame is None:
        start_frame = min(all_frames)
    if end_frame is None:
        end_frame = max(all_frames)

    front_left_files = [
        img
        for img in os.listdir(front_left_path)
        if int(img.split(".")[0]) % step == 0 and start_frame <= int(img.split(".")[0]) <= end_frame
    ]
    front_right_files = [
        img
        for img in os.listdir(front_right_path)
        if int(img.split(".")[0]) % step == 0 and start_frame <= int(img.split(".")[0]) <= end_frame
    ]

    num_workers = min(8, multiprocessing.cpu_count() // 2)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # front_left_results = executor.map(
        #     lambda img: _load_image(front_left_path / img, int(img.split(".")[0])),
        #     tqdm(front_left_files, desc="FrontLeft"),
        # )
        # front_right_results = executor.map(
        #     lambda img: _load_image(front_right_path / img, int(img.split(".")[0])),
        #     tqdm(front_right_files, desc="FrontRight"),
        # )
        futures_left = {
            executor.submit(_load_image, front_left_path / img, int(img.split(".")[0])): img
            for img in front_left_files
        }
        futures_right = {
            executor.submit(_load_image, front_right_path / img, int(img.split(".")[0])): img
            for img in front_right_files
        }

        front_left_imgs = {}
        for future in tqdm(as_completed(futures_left), total=len(futures_left), desc="FrontLeft"):
            frame_idx, img = future.result()
            front_left_imgs[frame_idx] = img

        front_right_imgs = {}
        for future in tqdm(
            as_completed(futures_right), total=len(futures_right), desc="FrontRight"
        ):
            frame_idx, img = future.result()
            front_right_imgs[frame_idx] = img

    # front_left_imgs = dict(front_left_results)
    # front_right_imgs = dict(front_right_results)

    assert len(front_left_imgs) == len(front_right_imgs)
    return front_left_imgs, front_right_imgs


def load_side_images(
    data_path: str | Path,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    step: int = 1,
):
    """Load side images from data log file."""
    data_path = Path(data_path)
    side_left_imgs_path = data_path / "Left"
    side_right_imgs_path = data_path / "Right"

    all_frames = [int(img.split(".")[0]) for img in os.listdir(side_left_imgs_path)]

    if start_frame is None:
        start_frame = min(all_frames)
    if end_frame is None:
        end_frame = max(all_frames)

    side_left_files = [
        img
        for img in os.listdir(side_left_imgs_path)
        if int(img.split(".")[0]) % step == 0 and start_frame <= int(img.split(".")[0]) <= end_frame
    ]
    side_right_files = [
        img
        for img in os.listdir(side_right_imgs_path)
        if int(img.split(".")[0]) % step == 0 and start_frame <= int(img.split(".")[0]) <= end_frame
    ]

    with ThreadPoolExecutor() as executor:
        side_left_results = executor.map(
            lambda img: _load_image(side_left_imgs_path / img, int(img.split(".")[0])),
            tqdm(side_left_files, desc="Left"),
        )
        side_right_results = executor.map(
            lambda img: _load_image(side_right_imgs_path / img, int(img.split(".")[0])),
            tqdm(side_right_files, desc="Right"),
        )
    side_left_imgs = dict(side_left_results)
    side_right_imgs = dict(side_right_results)
    assert len(side_left_imgs) == len(side_right_imgs)
    return side_left_imgs, side_right_imgs


def transform_to_pos_rpy(transform):
    """Convert a Transform object to roll-pitch-yaw euler angles and position.
    Euler angles are in radians.

    """
    t = np.array([transform.location.x, transform.location.y, transform.location.z])
    rpy = np.array([transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw])
    return t, rpy


def transform_to_numpy(transform):
    """Convert a Transform object to a 4x4 pose matrix.

    The resulting pose matrix has +X forward, +Y left, +Z up.

    """
    t, euler_angles = transform_to_pos_rpy(transform)
    R = Rotation.from_euler("xyz", euler_angles).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pos_rpy_to_pose(pos, rpy):
    """Convert a position and rpy to a 4x4 pose matrix."""
    R = Rotation.from_euler("xyz", rpy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def to_blender_convention(pose):
    """Convert a camera pose matrix to Blender convention.

    The camera pose matrix is assumed to have above starting convention (+X forward, +Y left, +Z up)
    The Blender convention has -Z forward, +X right, +Y up.

    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    # Convert the rotation matrix to the Blender convention
    rx, ry, rz = R[:, 0], R[:, 1], R[:, 2]
    R_blender = np.array([-ry, rz, -rx]).T
    return np.block([[R_blender, t[:, None]], [0, 0, 0, 1]])


def pose_to_pos_rpy(pose):
    """Convert a camera pose matrix to LAC convention.

    The camera pose matrix is assumed to have above starting convention (+X forward, +Y left, +Z up)
    The LAC convention has +X forward, +Y left, +Z up.

    """
    R = pose[:3, :3]
    t = pose[:3, 3]

    # Calculate yaw, pitch, roll using scipy Rotation
    r = Rotation.from_matrix(R)
    roll, pitch, yaw = r.as_euler("xyz")

    pos = np.array([t[0], t[1], t[2]])
    rpy = np.array([roll, pitch, yaw])

    return pos, rpy


def skew_symmetric(v):
    """Convert a 3D vector to a skew-symmetric matrix."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def normalize_rotation_matrix(R):
    """Normalizes a rotation matrix using SVD."""
    U, _, Vt = np.linalg.svd(R)  # Singular Value Decomposition
    R_normalized = U @ Vt  # Reconstruct a valid rotation matrix
    return R_normalized


def wrap_angle(angle):
    """Wrap an angle in radians to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def rotation_matrix_error(R1, R2):
    R_error = R1 @ R2.T  # Compute relative rotation matrix
    trace_value = np.trace(R_error)

    # Ensure the input to arccos is within the valid range [-1, 1]
    cos_theta = np.clip((trace_value - 1) / 2, -1.0, 1.0)

    theta = np.arccos(cos_theta)  # Rotation angle in radians
    return np.degrees(theta)  # Convert to degrees


def rotations_rmse(R1, R2):
    """Compute the RMSE between two sets of rotation matrices."""
    errors = np.zeros(len(R1))
    for i in range(len(R1)):
        errors[i] = rotation_matrix_error(R1[i], R2[i])
    return np.sqrt(np.mean(errors**2))


def np_img_to_PIL_rgb(img_array):
    """Convert a numpy image array to a PIL image."""
    return Image.fromarray(img_array).convert("RGB")


def mask_centroid(mask: np.ndarray) -> tuple | None:
    """Compute the centroid of a binary mask."""
    M = cv2.moments(mask)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def get_positions_from_poses(poses):
    return np.array([pose[:3, 3] for pose in poses])


def get_rotations_from_poses(poses):
    return np.array([pose[:3, :3] for pose in poses])


def positions_rmse_from_poses(poses_a, poses_b):
    pos_a = get_positions_from_poses(poses_a)
    pos_b = get_positions_from_poses(poses_b)
    return np.sqrt(np.mean(np.linalg.norm(pos_a - pos_b, axis=1) ** 2))


def rotations_rmse_from_poses(poses_a, poses_b):
    rots_a = get_rotations_from_poses(poses_a)
    rots_b = get_rotations_from_poses(poses_b)
    return rotations_rmse(rots_a, rots_b)


def grayscale_to_3ch_tensor(np_image):
    # Ensure the input is float32 (or float64 if needed)
    np_image = np_image.astype(np.float32) / 255.0 if np_image.max() > 1 else np_image
    # Add channel dimension and repeat across 3 channels
    torch_tensor = torch.from_numpy(np_image).unsqueeze(0).repeat(3, 1, 1)
    return torch_tensor
