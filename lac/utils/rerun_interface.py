"""Interface for using Rerun as a visualization dashboard

Based on: https://github.com/luigifreda/pyslam/blob/master/viz/rerun_interface.py

"""

import numpy as np
import cv2
import rerun as rr
import rerun.blueprint as rrb
import math as math
import subprocess
import psutil
import time


class Rerun:
    # Static parameters
    blueprint = None
    img_compress = False  # set to true if you want to compress the data
    img_compress_jpeg_quality = 85
    camera_img_resize_factors = None  # [0.1, 0.1]
    current_camera_view_scale = 0.3
    camera_poses_view_size = 0.5
    is_initialized = False

    def __init__(self) -> None:
        self.init()

    # ===================================================================================
    # Init
    # ===================================================================================

    @staticmethod
    def init(img_compress: bool = False) -> None:
        Rerun.img_compress = img_compress

        if Rerun.blueprint:
            rr.init("lac_dashboard", spawn=True, default_blueprint=Rerun.blueprint)
        else:
            rr.init("lac_dashboard", spawn=True)
        # rr.connect()  # Connect to a remote viewer
        Rerun.is_initialized = True

    @staticmethod
    def init3d(img_compress: bool = False) -> None:
        Rerun.init(img_compress)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        Rerun.log_3d_grid_plane()

    @staticmethod
    def init_vo(img_compress: bool = False) -> None:
        # Setup the blueprint
        Rerun.blueprint = rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="3D", origin="/world"),
                rrb.Spatial2DView(name="Camera", origin="/world/camera/image"),
            ),
            rrb.Horizontal(
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/trajectory_error"),
                    rrb.TimeSeriesView(origin="/trajectory_stats"),
                    column_shares=[1, 1],
                ),
                rrb.Spatial2DView(name="Trajectory 2D", origin="/trajectory_img/2d"),
                column_shares=[3, 2],
            ),
            row_shares=[3, 2],  # 3 "parts" in the first Horizontal, 2 in the second
        )
        # Init rerun
        Rerun.init3d(img_compress)

    # ===================================================================================
    # Image logging
    # ===================================================================================

    @staticmethod
    def log_img(img: np.ndarray) -> None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if Rerun.img_compress:
            rr.log(
                "world/camera/image",
                rr.Image(rgb).compress(jpeg_quality=Rerun.img_compress_jpeg_quality),
            )
        else:
            rr.log("world/camera/image", rr.Image(rgb))

    # ===================================================================================
    # 3D logging
    # ===================================================================================

    @staticmethod
    def log_3d_grid_plane(num_divs: int = 20, div_size: int = 1) -> None:
        rr.set_time_sequence("frame_id", 0)
        # Plane parallel to x-y at z = 0 with normal +z
        minx = -num_divs * div_size
        miny = -num_divs * div_size
        maxx = num_divs * div_size
        maxy = num_divs * div_size

        lines = []
        for n in range(2 * num_divs):
            lines.append([[minx + div_size * n, miny, 0], [minx + div_size * n, maxy, 0]])
            lines.append([[minx, miny + div_size * n, 0], [maxx, miny + div_size * n, 0]])

        rr.log(
            "world/grid",
            rr.LineStrips3D(
                lines,
                radii=0.01,
                colors=[0.7 * 255, 0.7 * 255, 0.7 * 255],
            ),
        )

    @staticmethod
    def log_3d_trajectory(
        frame_id: int,
        points: np.ndarray,
        trajectory_string: str = "trajectory",
        color=[255, 0, 0],
        size=0.05,
    ) -> None:
        rr.set_time_sequence("frame_id", frame_id)
        points = np.array(points).reshape(-1, 3)
        rr.log(
            "/world/" + trajectory_string,
            rr.LineStrips3D(
                [points],
                # rr.Radius.ui_points produces radii that the viewer interprets as given in ui points.
                radii=size,
                colors=color,
            ),
        )

    @staticmethod
    def log_3d_points(points: np.ndarray, topic: str = "/world", color=[0, 0, 255]) -> None:
        points = np.array(points).reshape(-1, 3)
        rr.log(
            topic,
            rr.Points3D(
                points,
                radii=0.01,
                colors=color,
            ),
        )

    # ===================================================================================
    # 2D logging
    # ===================================================================================

    @staticmethod
    def log_2d_seq_scalar(topic: str, frame_id: int, scalar_data) -> None:
        rr.set_time_sequence("frame_id", frame_id)
        rr.log(topic, rr.Scalar(scalar_data))

    @staticmethod
    def log_2d_time_scalar(topic: str, frame_time_ns, scalar_data) -> None:
        rr.set_time_nanos("time", frame_time_ns)
        rr.log(topic, rr.Scalar(scalar_data))

    @staticmethod
    def log_img_seq(topic: str, frame_id: int, img, adjust_rgb=True) -> None:
        if adjust_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rr.set_time_sequence("frame_id", frame_id)
        if Rerun.img_compress:
            rr.log(topic, rr.Image(img).compress(jpeg_quality=Rerun.img_compress_jpeg_quality))
        else:
            rr.log(topic, rr.Image(img))

    @staticmethod
    def log_img_time(topic: str, frame_time_ns, img, adjust_rgb=True) -> None:
        if adjust_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rr.set_time_nanos("time", frame_time_ns)
        if Rerun.img_compress:
            rr.log(topic, rr.Image(img).compress(jpeg_quality=Rerun.img_compress_jpeg_quality))
        else:
            rr.log(topic, rr.Image(img))
