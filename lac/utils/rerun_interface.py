"""Interface for using Rerun as a visualization dashboard

Based on: https://github.com/luigifreda/pyslam/blob/master/viz/rerun_interface.py

"""

import numpy as np
import cv2
import rerun as rr
import rerun.blueprint as rrb
import math as math

from lac.perception.segmentation import SemanticClasses
from lac.slam.backend import SemanticPointCloud


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

    # @staticmethod
    def init(img_compress: bool = False) -> None:
        Rerun.img_compress = img_compress

        if Rerun.blueprint:
            rr.init("lac_dashboard", spawn=True, default_blueprint=Rerun.blueprint)
        else:
            rr.init("lac_dashboard", spawn=True)
        # rr.connect()  # Connect to a remote viewer
        Rerun.is_initialized = True

    # @staticmethod
    def init3d(img_compress: bool = False) -> None:
        Rerun.init(img_compress)
        rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        Rerun.log_3d_grid_plane()

    # @staticmethod
    def init_vo(img_compress: bool = False) -> None:
        # Setup the blueprint
        print("Setting rerun blueprint")
        Rerun.blueprint = rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="3D", origin="/world"),
                rrb.Vertical(
                    rrb.Spatial2DView(name="Camera", origin="/world/camera/image"),
                    rrb.Spatial2DView(
                        name="Local frame",
                        origin="/local",
                        background=[25, 25, 25],
                        visual_bounds=rrb.VisualBounds2D(
                            x_range=np.array([0, 5]), y_range=np.array([-5, 5])
                        ),
                    ),
                ),
            ),
            rrb.Horizontal(
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/trajectory_error"),
                    rrb.TimeSeriesView(origin="/scores"),
                    column_shares=[1, 1],
                ),
                # rrb.TensorView(
                #     name="Metrics",
                #     origin="/metrics",  # <--- ADD THIS
                # ),
                column_shares=[3, 2],
            ),
            row_shares=[3, 2],  # 3 "parts" in the first Horizontal, 2 in the second
        )
        # Init rerun
        Rerun.init3d(img_compress)
        Rerun.log_2d_grid()

    # ===================================================================================
    # Image logging
    # ===================================================================================

    @staticmethod
    def log_img(img: np.ndarray) -> None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if Rerun.img_compress:
            rr.log(
                "/world/camera/image",
                rr.Image(rgb).compress(jpeg_quality=Rerun.img_compress_jpeg_quality),
            )
        else:
            rr.log("/world/camera/image", rr.Image(rgb))

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
            "/world/grid",
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
        # rr.set_time_sequence("frame_id", frame_id)
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

    @staticmethod
    def log_3d_semantic_points(semantic_points: SemanticPointCloud) -> None:
        ground_points = semantic_points.points[
            semantic_points.labels == SemanticClasses.GROUND.value
        ]
        rock_points = semantic_points.points[semantic_points.labels == SemanticClasses.ROCK.value]
        lander_points = semantic_points.points[
            semantic_points.labels == SemanticClasses.LANDER.value
        ]
        Rerun.log_3d_points(ground_points, topic="/world/ground_points", color=[120, 120, 120])
        Rerun.log_3d_points(rock_points, topic="/world/rock_points", color=[255, 0, 0])
        Rerun.log_3d_points(lander_points, topic="/world/lander_points", color=[255, 215, 0])

    # ===================================================================================
    # 2D logging
    # ===================================================================================
    @staticmethod
    def log_2d_grid(num_divs: int = 20, div_size: int = 1) -> None:
        rr.set_time_sequence("frame_id", 0)
        # Plane parallel to x-y at z = 0 with normal +z
        minx = -num_divs * div_size
        miny = -num_divs * div_size
        maxx = num_divs * div_size
        maxy = num_divs * div_size

        lines = []
        for n in range(2 * num_divs):
            lines.append([[minx + div_size * n, miny], [minx + div_size * n, maxy]])
            lines.append([[minx, miny + div_size * n], [maxx, miny + div_size * n]])

        rr.log(
            "/local/grid",
            rr.LineStrips2D(
                lines,
                radii=0.01,
                colors=[0.7 * 255, 0.7 * 255, 0.7 * 255],
            ),
        )

    @staticmethod
    def log_2d_trajectory(
        frame_id: int, trajectory: np.ndarray, topic: str = "/local/path"
    ) -> None:
        # rr.set_time_sequence("frame_id", frame_id)
        # Swap x and y, and invert y
        trajectory = np.column_stack((-trajectory[:, 1], -trajectory[:, 0]))
        rr.log(
            topic,
            rr.LineStrips2D(
                [trajectory],
                radii=0.05,
                colors=[0, 0, 255],
            ),
        )

    @staticmethod
    def log_2d_obstacle_map(
        frame_id: int, centers: np.ndarray, radii: np.ndarray, topic: str = "/local/obstacles"
    ) -> None:
        # rr.set_time_sequence("frame_id", frame_id)
        # Swap x and y, and invert y
        centers = np.column_stack((-centers[:, 1], -centers[:, 0]))
        rr.log(
            topic,
            rr.Points2D(
                centers,
                radii=radii,
                colors=[255, 0, 0],
            ),
        )

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

    @staticmethod
    def log_scalar(topic: str, value: float):
        rr.log(topic, rr.Scalar(value))
