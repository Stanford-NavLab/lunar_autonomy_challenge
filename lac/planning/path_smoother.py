"""Terrain-aware path smoothing and trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

import lac.params as params

try:
    from scipy.interpolate import CubicSpline

    _HAS_SCIPY = True
except Exception:
    CubicSpline = None
    _HAS_SCIPY = False


@dataclass
class Path2D:
    xy: np.ndarray  # (N,2) world meters
    meta: dict  # costs, cell_size, etc.


@dataclass
class Trajectory2D:
    t: np.ndarray
    xyt: np.ndarray  # (M,3) x,y,theta
    v: np.ndarray
    w: np.ndarray


@dataclass
class PathSmootherConfig:
    """Configuration for trajectory smoothing and dynamic feasibility."""

    cell_size: float = params.CELL_WIDTH
    ds: float = 0.10

    v_nominal: float = params.TARGET_SPEED
    v_max: float = 0.60
    v_min: float = 0.05
    v_start: float = 0.0
    v_end: float = 0.0

    max_omega: float = params.MAX_STEER
    max_lat_acc: float = 0.5
    max_accel: float = 0.35
    max_decel: float = 0.45

    slope_soft_max: float = 0.55
    slope_hard_max: float = 0.90
    roughness_soft_max: float = 0.08
    roughness_hard_max: float = 0.15
    roughness_window: int = 5

    terrain_speed_floor_frac: float = 0.15
    use_spline: bool = True
    spline_bc_type: str = "natural"
    use_initial_heading: bool = True
    initial_heading_window_m: float = 1.0
    use_goal_heading: bool = True
    goal_heading_window_m: float = 1.0


def _box_mean(z: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return z.copy()
    pad = window // 2
    z_pad = np.pad(z, pad_width=pad, mode="edge")
    integral = (
        np.pad(z_pad, ((1, 0), (1, 0)), mode="constant", constant_values=0.0).cumsum(0).cumsum(1)
    )
    h, w = z.shape
    y0 = np.arange(0, h)
    x0 = np.arange(0, w)
    y1 = y0 + window
    x1 = x0 + window
    total = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    return total / float(window * window)


def _normalize_heightmap(heightmap: np.ndarray) -> np.ndarray:
    if heightmap.ndim == 2:
        return heightmap.astype(np.float32, copy=False)
    if heightmap.ndim == 3 and heightmap.shape[2] >= 3:
        return heightmap[:, :, 2].astype(np.float32, copy=False)
    raise ValueError(f"heightmap must be (H,W) or (H,W,C>=3), got {heightmap.shape}")


def _remove_duplicate_points(xy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if len(xy) <= 1:
        return xy
    keep = [0]
    for i in range(1, len(xy)):
        if np.linalg.norm(xy[i] - xy[keep[-1]]) > eps:
            keep.append(i)
    return xy[np.array(keep, dtype=np.int32)]


def _arc_length(xy: np.ndarray) -> np.ndarray:
    if len(xy) <= 1:
        return np.array([0.0], dtype=np.float64)
    seg = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    return np.hstack(([0.0], np.cumsum(seg)))


def _bilinear_sample(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = x - x0
    wy = y - y0
    v00 = grid[y0, x0]
    v10 = grid[y0, x1]
    v01 = grid[y1, x0]
    v11 = grid[y1, x1]
    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )


def _yaw_from_pose(initial_pose: np.ndarray) -> float:
    pose = np.asarray(initial_pose)
    if pose.shape == (4, 4):
        return float(np.arctan2(pose[1, 0], pose[0, 0]))
    if pose.ndim == 1 and pose.shape[0] >= 3:
        return float(pose[2])
    raise ValueError(f"initial_pose must be 4x4 or [x,y,yaw], got shape {pose.shape}")


class PathSmoother:
    """Generate a smooth terrain-aware and dynamically feasible trajectory from a 2D path."""

    def __init__(self, config: Optional[PathSmootherConfig] = None):
        self.cfg = config or PathSmootherConfig()

    def compute_terrain_maps(self, heightmap: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute slope and roughness maps from DEM z[y,x]."""
        z = _normalize_heightmap(heightmap)
        dz_dy, dz_dx = np.gradient(z, self.cfg.cell_size, edge_order=1)
        slope = np.hypot(dz_dx, dz_dy).astype(np.float32)
        w = self.cfg.roughness_window
        if w < 1 or w % 2 == 0:
            raise ValueError("roughness_window must be odd and >= 1")
        mu = _box_mean(z, w)
        mu2 = _box_mean(z * z, w)
        roughness = np.sqrt(np.maximum(mu2 - mu * mu, 0.0)).astype(np.float32)
        return {"z": z, "slope": slope, "roughness": roughness}

    def _resample_xy(self, xy: np.ndarray) -> np.ndarray:
        if len(xy) < 2:
            return xy
        s = _arc_length(xy)
        if s[-1] <= self.cfg.ds:
            return xy
        s_new = np.arange(0.0, s[-1] + 1e-9, self.cfg.ds)
        if self.cfg.use_spline and _HAS_SCIPY and len(xy) >= 4:
            cx = CubicSpline(s, xy[:, 0], bc_type=self.cfg.spline_bc_type)
            cy = CubicSpline(s, xy[:, 1], bc_type=self.cfg.spline_bc_type)
            x_new = cx(s_new)
            y_new = cy(s_new)
        else:
            x_new = np.interp(s_new, s, xy[:, 0])
            y_new = np.interp(s_new, s, xy[:, 1])
        return np.column_stack([x_new, y_new]).astype(np.float64)

    def _heading_curvature(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = _arc_length(xy)
        if len(xy) < 3 or s[-1] <= 1e-9:
            theta = np.zeros(len(xy), dtype=np.float64)
            kappa = np.zeros(len(xy), dtype=np.float64)
            return theta, kappa

        dx_ds = np.gradient(xy[:, 0], s, edge_order=1)
        dy_ds = np.gradient(xy[:, 1], s, edge_order=1)
        ddx_ds = np.gradient(dx_ds, s, edge_order=1)
        ddy_ds = np.gradient(dy_ds, s, edge_order=1)

        theta = np.unwrap(np.arctan2(dy_ds, dx_ds))
        denom = (dx_ds * dx_ds + dy_ds * dy_ds) ** 1.5
        denom = np.maximum(denom, 1e-8)
        kappa = (dx_ds * ddy_ds - dy_ds * ddx_ds) / denom
        return theta, kappa

    def _blend_start_orientation(self, xy: np.ndarray, initial_yaw_rad: float) -> np.ndarray:
        cfg = self.cfg
        if len(xy) < 2 or cfg.initial_heading_window_m <= 0.0:
            return xy
        s = _arc_length(xy)
        window = float(cfg.initial_heading_window_m)
        start = xy[0].copy()
        heading_dir = np.array([np.cos(initial_yaw_rad), np.sin(initial_yaw_rad)], dtype=np.float64)
        out = xy.copy()
        for i in range(1, len(out)):
            si = float(s[i])
            if si >= window:
                break
            alpha = 1.0 - si / window
            target = start + si * heading_dir
            out[i] = alpha * target + (1.0 - alpha) * out[i]
        return out

    def _blend_end_orientation(self, xy: np.ndarray, goal_yaw_rad: float) -> np.ndarray:
        cfg = self.cfg
        if len(xy) < 2 or cfg.goal_heading_window_m <= 0.0:
            return xy
        s = _arc_length(xy)
        total_len = float(s[-1])
        if total_len <= 1e-9:
            return xy
        window = float(cfg.goal_heading_window_m)
        end = xy[-1].copy()
        heading_dir = np.array([np.cos(goal_yaw_rad), np.sin(goal_yaw_rad)], dtype=np.float64)
        out = xy.copy()
        for i in range(len(out) - 2, -1, -1):
            d_end = total_len - float(s[i])
            if d_end >= window:
                break
            alpha = 1.0 - d_end / window
            target = end - d_end * heading_dir
            out[i] = alpha * target + (1.0 - alpha) * out[i]
        return out

    def _terrain_speed_limit(self, slope: np.ndarray, roughness: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        slope_soft = np.clip(slope / max(cfg.slope_soft_max, 1e-6), 0.0, 1.0)
        rough_soft = np.clip(roughness / max(cfg.roughness_soft_max, 1e-6), 0.0, 1.0)

        slope_scale = 1.0 - (1.0 - cfg.terrain_speed_floor_frac) * slope_soft
        rough_scale = 1.0 - (1.0 - cfg.terrain_speed_floor_frac) * rough_soft
        v = cfg.v_nominal * np.minimum(slope_scale, rough_scale)

        hard_invalid = (slope >= cfg.slope_hard_max) | (roughness >= cfg.roughness_hard_max)
        v[hard_invalid] = 0.0
        return np.clip(v, 0.0, cfg.v_max)

    def _apply_accel_limits(self, v_lim: np.ndarray, ds: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        n = len(v_lim)
        v = np.minimum(v_lim.copy(), cfg.v_max)
        if n == 0:
            return v

        v[0] = min(v[0], cfg.v_start)
        for i in range(n - 1):
            v_next = np.sqrt(max(v[i] * v[i] + 2.0 * cfg.max_accel * ds[i], 0.0))
            v[i + 1] = min(v[i + 1], v_next)

        v[-1] = min(v[-1], cfg.v_end)
        for i in range(n - 2, -1, -1):
            v_prev = np.sqrt(max(v[i + 1] * v[i + 1] + 2.0 * cfg.max_decel * ds[i], 0.0))
            v[i] = min(v[i], v_prev)

        return v

    def smooth(
        self,
        path: Path2D | np.ndarray | Sequence[Sequence[float]],
        heightmap: np.ndarray,
        initial_pose: Optional[np.ndarray] = None,
        goal_yaw_rad: Optional[float] = None,
    ) -> Trajectory2D:
        """Build smooth terrain-aware trajectory from an xy path and heightmap.

        Args:
            path: Path2D or array-like (N,2) in world meters.
            heightmap: DEM map as z[y,x] or map[y,x,c] where channel 2 is z.

        Returns:
            Trajectory2D with sampled (x, y, theta), linear velocity v, angular velocity w, and time t.
        """
        if isinstance(path, Path2D):
            xy_in = np.asarray(path.xy, dtype=np.float64)
        else:
            xy_in = np.asarray(path, dtype=np.float64)

        if xy_in.ndim != 2 or xy_in.shape[1] != 2:
            raise ValueError(f"path must be shape (N,2), got {xy_in.shape}")

        xy = _remove_duplicate_points(xy_in)
        if len(xy) < 2:
            if len(xy) == 0:
                xy = np.zeros((1, 2), dtype=np.float64)
            t = np.array([0.0], dtype=np.float64)
            xyt = np.column_stack([xy, np.array([0.0], dtype=np.float64)])
            return Trajectory2D(t=t, xyt=xyt, v=np.zeros(1), w=np.zeros(1))

        xy = self._resample_xy(xy)
        if initial_pose is not None and self.cfg.use_initial_heading:
            initial_yaw_rad = _yaw_from_pose(initial_pose)
            xy = self._blend_start_orientation(xy, initial_yaw_rad=initial_yaw_rad)
        if goal_yaw_rad is not None and self.cfg.use_goal_heading:
            xy = self._blend_end_orientation(xy, goal_yaw_rad=float(goal_yaw_rad))
        theta, kappa = self._heading_curvature(xy)

        terrain = self.compute_terrain_maps(heightmap)
        h, w = terrain["z"].shape
        # Convert world coordinates to DEM grid coordinates in the same
        # cell-centered frame used by dem_planner.world_to_grid.
        xg = xy[:, 0] / self.cfg.cell_size + (w - 1) / 2.0
        yg = xy[:, 1] / self.cfg.cell_size + (h - 1) / 2.0
        slope_s = _bilinear_sample(terrain["slope"], xg, yg)
        rough_s = _bilinear_sample(terrain["roughness"], xg, yg)

        v_terrain = self._terrain_speed_limit(slope_s, rough_s)
        v_curv = np.sqrt(np.maximum(self.cfg.max_lat_acc / np.maximum(np.abs(kappa), 1e-6), 0.0))
        v_curv = np.clip(v_curv, self.cfg.v_min, self.cfg.v_max)

        v_lim = np.minimum(v_terrain, v_curv)
        ds = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
        ds = np.maximum(ds, 1e-6)
        v = self._apply_accel_limits(v_lim, ds)

        # Enforce angular-rate feasibility: |w| = |kappa*v| <= max_omega
        omega_safe_v = self.cfg.max_omega / np.maximum(np.abs(kappa), 1e-6)
        v = np.minimum(v, omega_safe_v)
        v = np.clip(v, 0.0, self.cfg.v_max)

        # If slope/roughness exceeded hard bounds, vehicle should effectively stop.
        stop_mask = (slope_s >= self.cfg.slope_hard_max) | (rough_s >= self.cfg.roughness_hard_max)
        v[stop_mask] = 0.0

        w = kappa * v

        t = np.zeros(len(xy), dtype=np.float64)
        for i in range(1, len(xy)):
            v_avg = 0.5 * (v[i - 1] + v[i])
            if v_avg <= 1e-4:
                t[i] = t[i - 1] + ds[i - 1] / max(self.cfg.v_min, 1e-3)
            else:
                t[i] = t[i - 1] + ds[i - 1] / v_avg

        xyt = np.column_stack([xy, theta])
        return Trajectory2D(t=t, xyt=xyt, v=v, w=w)

    def __call__(
        self,
        path: Path2D | np.ndarray | Sequence[Sequence[float]],
        heightmap: np.ndarray,
        initial_pose: Optional[np.ndarray] = None,
        goal_yaw_rad: Optional[float] = None,
    ) -> Trajectory2D:
        return self.smooth(path, heightmap, initial_pose=initial_pose, goal_yaw_rad=goal_yaw_rad)
