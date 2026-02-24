"""Interface for using Rerun as a visualization dashboard

Based on: https://github.com/luigifreda/pyslam/blob/master/viz/rerun_interface.py

"""

import numpy as np
import cv2
import rerun as rr
import rerun.blueprint as rrb
import math as math

import lac.params as params
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
    def init(img_compress: bool = False, save_path: str = None) -> None:
        Rerun.img_compress = img_compress

        if Rerun.blueprint:
            rr.init("lac_dashboard", spawn=True, default_blueprint=Rerun.blueprint)
        else:
            rr.init("lac_dashboard", spawn=True)
        # rr.connect()  # Connect to a remote viewer
        if save_path is not None:
            rr.save(save_path)
        Rerun.is_initialized = True

    # @staticmethod
    def init3d(img_compress: bool = False, save_path: str = None) -> None:
        Rerun.init(img_compress, save_path)
        rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        Rerun.log_3d_grid_plane()

    # @staticmethod
    def init_vo(img_compress: bool = False, save_path: str = None) -> None:
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
        Rerun.init3d(img_compress, save_path)
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
    def log_3d_mesh(mesh: rr.Mesh3D, topic: str = "/world/dem_mesh", static: bool = True) -> None:
        rr.log(topic, mesh, static=static)

    @staticmethod
    def log_3d_line_strips(
        lines: list[list[list[float]]],
        topic: str = "/world/lines",
        color: list[int] = [180, 180, 180],
        radius: float = 0.005,
        static: bool = True,
    ) -> None:
        rr.log(topic, rr.LineStrips3D(lines, colors=color, radii=radius), static=static)

    @staticmethod
    def log_3d_semantic_points(semantic_points: SemanticPointCloud, downsample: int = 10) -> None:
        ground_points = semantic_points.points[
            semantic_points.labels == SemanticClasses.GROUND.value
        ][::downsample]
        rock_points = semantic_points.points[semantic_points.labels == SemanticClasses.ROCK.value][
            ::downsample
        ]
        lander_points = semantic_points.points[
            semantic_points.labels == SemanticClasses.LANDER.value
        ][::downsample]
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

        axes = [
            [[0, miny], [0, maxy]],
            [[minx, 0], [maxx, 0]],
        ]

        rr.log(
            "/local/grid",
            rr.LineStrips2D(
                lines,
                radii=0.005,
                # colors=[0.7 * 255, 0.7 * 255, 0.7 * 255],
                colors=[0, 0, 0],
            ),
        )
        rr.log(
            "/local/axes",
            rr.LineStrips2D(
                axes,
                radii=0.02,
                # colors=[0.7 * 255, 0.7 * 255, 0.7 * 255],
                colors=[0, 0, 0],
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


def rerun_dem(
    dem: np.ndarray,
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
    crop_percentage: float = 1.0,
    alpha: int = 70,
):
    """Create a Rerun mesh from a DEM.

    Args:
        dem:
            - z grid with shape (H, W), or
            - xyz map with shape (H, W, C>=3) where channels [0,1,2] are x,y,z.
        x_grid: Optional x coordinates for each DEM cell (H,W) or (W,).
        y_grid: Optional y coordinates for each DEM cell (H,W) or (H,).
        crop_percentage: Percentage of the central area to use (0.0, 1.0].

    Returns:
        rr.Mesh3D object.
    """
    if not (0.0 < crop_percentage <= 1.0):
        raise ValueError("crop_percentage must be in (0.0, 1.0].")

    if dem.ndim == 3 and dem.shape[2] >= 3 and x_grid is None and y_grid is None:
        x_full = dem[:, :, 0].astype(np.float32, copy=False)
        y_full = dem[:, :, 1].astype(np.float32, copy=False)
        z_full = dem[:, :, 2].astype(np.float32, copy=False)
    elif dem.ndim == 2:
        z_full = dem.astype(np.float32, copy=False)
        h, w = z_full.shape
        if x_grid is None:
            x_full = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
        else:
            x_arr = np.asarray(x_grid, dtype=np.float32)
            if x_arr.ndim == 1:
                if x_arr.shape[0] != w:
                    raise ValueError("x_grid 1D length must match DEM width.")
                x_full = np.tile(x_arr[None, :], (h, 1))
            elif x_arr.ndim == 2 and x_arr.shape == z_full.shape:
                x_full = x_arr
            else:
                raise ValueError("x_grid must be shape (W,) or (H,W).")
        if y_grid is None:
            y_full = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
        else:
            y_arr = np.asarray(y_grid, dtype=np.float32)
            if y_arr.ndim == 1:
                if y_arr.shape[0] != h:
                    raise ValueError("y_grid 1D length must match DEM height.")
                y_full = np.tile(y_arr[:, None], (1, w))
            elif y_arr.ndim == 2 and y_arr.shape == z_full.shape:
                y_full = y_arr
            else:
                raise ValueError("y_grid must be shape (H,) or (H,W).")
    else:
        raise ValueError("dem must be shape (H,W) z-grid or (H,W,C>=3) xyz map.")

    h, w = z_full.shape
    crop_h = max(2, int(round(h * crop_percentage)))
    crop_w = max(2, int(round(w * crop_percentage)))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    y1 = min(h, y0 + crop_h)
    x1 = min(w, x0 + crop_w)

    x_crop = x_full[y0:y1, x0:x1]
    y_crop = y_full[y0:y1, x0:x1]
    z_crop = z_full[y0:y1, x0:x1]
    h_crop, w_crop = z_crop.shape

    vertices = np.column_stack([x_crop.reshape(-1), y_crop.reshape(-1), z_crop.reshape(-1)])

    indices = []
    for i in range(h_crop - 1):
        for j in range(w_crop - 1):
            # The flat index of the top-left corner (i, j)
            k_tl = i * w_crop + j
            k_tr = i * w_crop + j + 1
            k_bl = (i + 1) * w_crop + j
            k_br = (i + 1) * w_crop + j + 1

            # Triangle 1: (Top-Left, Bottom-Left, Bottom-Right)
            indices.append(k_tl)
            indices.append(k_bl)
            indices.append(k_br)

            # Triangle 2: (Top-Left, Bottom-Right, Top-Right)
            indices.append(k_tl)
            indices.append(k_br)
            indices.append(k_tr)

    indices_array = np.array(indices, dtype=np.uint32)

    alpha_u8 = int(np.clip(alpha, 0, 255))
    return rr.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=indices_array,
        albedo_factor=rr.AlbedoFactor([120, 120, 120, alpha_u8]),
    )


def rerun_dem_grid_lines(
    dem: np.ndarray,
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
    crop_percentage: float = 1.0,
    stride: int = 8,
) -> list[list[list[float]]]:
    """Create row/column wireframe lines over DEM surface."""
    if dem.ndim == 3 and dem.shape[2] >= 3 and x_grid is None and y_grid is None:
        x_full = dem[:, :, 0].astype(np.float32, copy=False)
        y_full = dem[:, :, 1].astype(np.float32, copy=False)
        z_full = dem[:, :, 2].astype(np.float32, copy=False)
    elif dem.ndim == 2:
        z_full = dem.astype(np.float32, copy=False)
        h, w = z_full.shape
        if x_grid is None:
            x_full = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
        else:
            x_arr = np.asarray(x_grid, dtype=np.float32)
            x_full = np.tile(x_arr[None, :], (h, 1)) if x_arr.ndim == 1 else x_arr
        if y_grid is None:
            y_full = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
        else:
            y_arr = np.asarray(y_grid, dtype=np.float32)
            y_full = np.tile(y_arr[:, None], (1, w)) if y_arr.ndim == 1 else y_arr
    else:
        raise ValueError("dem must be shape (H,W) z-grid or (H,W,C>=3) xyz map.")

    h, w = z_full.shape
    crop_h = max(2, int(round(h * crop_percentage)))
    crop_w = max(2, int(round(w * crop_percentage)))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    y1 = min(h, y0 + crop_h)
    x1 = min(w, x0 + crop_w)
    x_crop = x_full[y0:y1, x0:x1]
    y_crop = y_full[y0:y1, x0:x1]
    z_crop = z_full[y0:y1, x0:x1]

    step = max(1, int(stride))
    lines: list[list[list[float]]] = []
    for i in range(0, z_crop.shape[0], step):
        pts = np.column_stack([x_crop[i, :], y_crop[i, :], z_crop[i, :]])
        lines.append(pts.tolist())
    for j in range(0, z_crop.shape[1], step):
        pts = np.column_stack([x_crop[:, j], y_crop[:, j], z_crop[:, j]])
        lines.append(pts.tolist())
    return lines


def rerun_lander(
    center_xy: tuple[float, float] = (0.0, 0.0),
    base_z: float = 0.0,
    side_length_m: float | None = None,
    alpha: int = 255,
) -> rr.Mesh3D:
    """Create a simple lander mesh as a gold 3D box."""
    side = float(params.LANDER_WIDTH if side_length_m is None else side_length_m)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half = 0.5 * side
    z0 = float(base_z)
    z1 = z0 + side

    vertices = np.array(
        [
            [cx - half, cy - half, z0],  # 0
            [cx + half, cy - half, z0],  # 1
            [cx + half, cy + half, z0],  # 2
            [cx - half, cy + half, z0],  # 3
            [cx - half, cy - half, z1],  # 4
            [cx + half, cy - half, z1],  # 5
            [cx + half, cy + half, z1],  # 6
            [cx - half, cy + half, z1],  # 7
        ],
        dtype=np.float32,
    )

    # 12 triangles (2 per face), counter-clockwise winding when viewed from outside.
    triangle_indices = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # bottom
            [4, 5, 6],
            [4, 6, 7],  # top
            [0, 1, 5],
            [0, 5, 4],  # -y face
            [1, 2, 6],
            [1, 6, 5],  # +x face
            [2, 3, 7],
            [2, 7, 6],  # +y face
            [3, 0, 4],
            [3, 4, 7],  # -x face
        ],
        dtype=np.uint32,
    )

    alpha_u8 = int(np.clip(alpha, 0, 255))
    return rr.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=triangle_indices,
        albedo_factor=rr.AlbedoFactor([255, 215, 0, alpha_u8]),
    )


def rerun_box_mesh(
    center_xyz: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    yaw_rad: float = 0.0,
    rgba: tuple[int, int, int, int] = (0, 255, 0, 160),
) -> rr.Mesh3D:
    """Create a yaw-rotated axis-aligned box mesh in world frame."""
    cx, cy, cz = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
    sx, sy, sz = float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz

    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )

    c, s = np.cos(float(yaw_rad)), np.sin(float(yaw_rad))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    vertices = (local @ Rz.T) + np.array([cx, cy, cz], dtype=np.float32)

    triangle_indices = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.uint32,
    )

    r = int(np.clip(rgba[0], 0, 255))
    g = int(np.clip(rgba[1], 0, 255))
    b = int(np.clip(rgba[2], 0, 255))
    a = int(np.clip(rgba[3], 0, 255))
    return rr.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=triangle_indices,
        albedo_factor=rr.AlbedoFactor([r, g, b, a]),
    )
