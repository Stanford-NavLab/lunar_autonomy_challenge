import numpy as np

from lac.perception.vision import calc_camera_intrinsics
from lac.utils.frames import invert_transform_mat
from lac.params import IMG_WIDTH, IMG_HEIGHT, IMG_FOV_RAD


class Camera:
    """Pinhole camera with square pixels and no distortion"""

    def __init__(self, pose, width=IMG_WIDTH, height=IMG_HEIGHT):
        """

        width : int - Image width
        height : int - Image height
        pose : np.ndarray (4, 4) - Camera pose in world frame

        """
        self.width = width
        self.height = height
        self.cx = width / 2
        self.cy = height / 2
        self.fx = width / (2 * np.tan(IMG_FOV_RAD / 2))
        self.fy = self.fx
        self.pose = pose
        self.pose_inv = invert_transform_mat(pose)
        self.K = calc_camera_intrinsics(width, height)

    def project_world_points_to_camera(self, points):
        """Project 3D world points to camera frame

        points : np.ndarray (N, 3) - 3D points in world frame

        """
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        camera_points = self.pose_inv @ points_homogeneous.T
        return camera_points[:3].T

    def project_camera_points_to_uv(self, points_camera, colors=None):
        """Project 3D points to 2D pixel coordinates

        points_camera : np.ndarray (N, 3) - 3D points in camera frame
        colors : np.ndarray (N,) - monochrome colors of the points

        """
        uvw = points_camera[:, :3] @ self.K.T
        uv = uvw[:, :2] / (uvw[:, 2:] + 1e-8)
        depths = uvw[:, 2]

        out_of_frame_idxs = np.logical_or(
            np.logical_or(uv[:, 0] < 0, uv[:, 0] > self.width),
            np.logical_or(uv[:, 1] < 0, uv[:, 1] > self.height),
        )
        out_of_frame_idxs = np.logical_or(out_of_frame_idxs, depths < 0)
        out_of_frame_idxs = np.logical_or(
            out_of_frame_idxs, (~np.isfinite(uv[:, 0])) | (~np.isfinite(uv[:, 1]))
        )

        uv_inframe = uv[~out_of_frame_idxs]
        depths_inframe = depths[~out_of_frame_idxs]
        if colors is not None:
            colors_inframe = colors[~out_of_frame_idxs]
            return uv_inframe, depths_inframe, colors_inframe
        else:
            return uv, depths

    def project_world_points_to_uv(self, points, colors=None):
        """Project 3D world points to 2D pixel coordinates

        points : np.ndarray (N, 3) - 3D points in world frame
        colors : np.ndarray (N,) - monochrome colors of the points

        """
        points_camera = self.project_world_points_to_camera(points)
        return self.project_camera_points_to_uv(points_camera, colors)
