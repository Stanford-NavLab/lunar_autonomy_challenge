"""Terrain-aware DEM path planning for LAC grids.

This module implements:
- Terrain feature extraction (slope, roughness, step height)
- Costmap construction with hard invalid masks
- A* and optional Theta* (any-angle) planning on a 2D grid
- Path post-processing (collinear simplification, shortcut smoothing, optional spline)
- End-to-end planning from world or grid start/goal inputs

Coordinate conventions:
- DEM indexing is z[y, x]
- World x increases with grid column index
- World y increases with grid row index
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import lac.params as lac_params
from lac.params import CELL_WIDTH

try:
    from scipy.interpolate import CubicSpline

    _HAS_SCIPY = True
except Exception:
    CubicSpline = None
    _HAS_SCIPY = False


GridPoint = Tuple[int, int]  # (x, y)
WorldPoint = Tuple[float, float]  # (x_m, y_m)


@dataclass
class DEMPlannerParams:
    """Configurable parameters for DEM feature/cost/path planning."""

    roughness_window: int = 5
    use_eight_neighbor_step: bool = True

    base_cost: float = 1.0
    w_s: float = 5.0
    w_r: float = 2.0
    w_d: float = 10.0
    s_max: float = 0.6
    r_max: float = 0.1
    d_max: float = 0.15

    hard_slope_max: float = 0.9
    hard_step_max: float = 0.22
    use_lander_keepout: bool = True
    lander_buffer_m: float = 1.0
    lander_center_xy_m: Tuple[float, float] = (0.0, 0.0)

    simplify_collinear: bool = True
    do_shortcut: bool = True
    shortcut_iters: int = 200
    do_spline: bool = True
    spline_spacing_m: float = 0.25
    spline_tension: float = 1.0

    connectivity: int = 8
    seed: int = 0

    @classmethod
    def from_any(cls, params: Optional["DEMPlannerParams | Dict[str, Any]"]) -> "DEMPlannerParams":
        if params is None:
            return cls()
        if isinstance(params, cls):
            return params
        return cls(**params)


def _box_mean(z: np.ndarray, window: int) -> np.ndarray:
    """Fast box mean via integral image with edge padding."""
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


def _neighbor_offsets(use_eight: bool = True) -> List[Tuple[int, int]]:
    if use_eight:
        return [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]
    return [(0, -1), (-1, 0), (1, 0), (0, 1)]


def world_to_grid(
    x_m: float,
    y_m: float,
    cell_size: float = CELL_WIDTH,
    shape: Optional[Tuple[int, int]] = None,
) -> GridPoint:
    """Convert world meters to nearest grid index (x, y).

    If shape=(H,W) is provided, world origin (0,0) is treated as map center.
    Otherwise origin is treated as grid corner (legacy behavior).
    """
    if shape is None:
        ix = int(round(x_m / cell_size))
        iy = int(round(y_m / cell_size))
    else:
        h, w = shape
        ix = int(round(x_m / cell_size + w / 2.0))
        iy = int(round(y_m / cell_size + h / 2.0))
    return ix, iy


def grid_to_world(
    ix: int,
    iy: int,
    cell_size: float = CELL_WIDTH,
    shape: Optional[Tuple[int, int]] = None,
) -> WorldPoint:
    """Convert grid index (x, y) to world meters (x_m, y_m).

    If shape=(H,W) is provided, world origin (0,0) is map center.
    Otherwise origin is treated as grid corner (legacy behavior).
    """
    if shape is None:
        return ix * cell_size, iy * cell_size
    h, w = shape
    return (ix - w / 2.0) * cell_size, (iy - h / 2.0) * cell_size


def clamp_grid(p: GridPoint, width: int, height: int) -> GridPoint:
    """Clamp (x, y) to map bounds."""
    x = int(np.clip(p[0], 0, width - 1))
    y = int(np.clip(p[1], 0, height - 1))
    return x, y


def _normalize_dem(z: np.ndarray) -> np.ndarray:
    """Accept HxW DEM or HxWx4 map, return z[y, x]."""
    if z.ndim == 2:
        return z.astype(np.float32, copy=False)
    if z.ndim == 3 and z.shape[2] >= 3:
        return z[:, :, 2].astype(np.float32, copy=False)
    raise ValueError(f"Expected z as (H,W) or (H,W,C>=3), got shape {z.shape}")


def apply_lander_keepout(
    blocked: np.ndarray,
    cell_size: float = CELL_WIDTH,
    lander_center_xy_m: Tuple[float, float] = (0.0, 0.0),
    buffer_m: float = 1.0,
) -> np.ndarray:
    """Add a lander keepout mask using dimensions from lac.params with extra buffer."""
    keepout = lander_keepout_mask(
        shape=blocked.shape,
        cell_size=cell_size,
        lander_center_xy_m=lander_center_xy_m,
        buffer_m=buffer_m,
    )
    return blocked | keepout


def lander_keepout_mask(
    shape: Tuple[int, int],
    cell_size: float = CELL_WIDTH,
    lander_center_xy_m: Tuple[float, float] = (0.0, 0.0),
    buffer_m: float = 1.0,
) -> np.ndarray:
    """Compute the lander keepout mask (True means blocked by keepout)."""
    lander_xy = lac_params.LANDER_GLOBAL[:, :2]
    x_min = float(np.min(lander_xy[:, 0])) - buffer_m
    x_max = float(np.max(lander_xy[:, 0])) + buffer_m
    y_min = float(np.min(lander_xy[:, 1])) - buffer_m
    y_max = float(np.max(lander_xy[:, 1])) + buffer_m

    cx, cy = lander_center_xy_m
    x_coords = (np.arange(shape[1], dtype=np.float32) - shape[1] / 2.0) * float(cell_size)
    y_coords = (np.arange(shape[0], dtype=np.float32) - shape[0] / 2.0) * float(cell_size)
    x_world = x_coords[None, :] - float(cx)
    y_world = y_coords[:, None] - float(cy)
    return (x_world >= x_min) & (x_world <= x_max) & (y_world >= y_min) & (y_world <= y_max)


def compute_features(
    z: np.ndarray,
    cell_size: float = CELL_WIDTH,
    roughness_window: int = 5,
    use_eight_neighbor_step: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute terrain features from DEM z[y, x].

    Returns dict with:
    - z: float32 HxW
    - slope: |grad z| in m/m
    - roughness: local std(z) in meters
    - step: max abs dz to neighbors in meters
    """
    z2d = _normalize_dem(z)
    if roughness_window < 1 or roughness_window % 2 == 0:
        raise ValueError("roughness_window must be odd and >= 1")

    dz_dy, dz_dx = np.gradient(z2d, cell_size, edge_order=1)
    slope = np.hypot(dz_dx, dz_dy).astype(np.float32)

    mu = _box_mean(z2d, roughness_window)
    mu2 = _box_mean(z2d * z2d, roughness_window)
    roughness = np.sqrt(np.maximum(mu2 - mu * mu, 0.0)).astype(np.float32)

    step = np.zeros_like(z2d, dtype=np.float32)
    for dx, dy in _neighbor_offsets(use_eight_neighbor_step):
        shifted = np.full_like(z2d, z2d[0, 0], dtype=np.float32)
        src_y0 = max(0, -dy)
        src_y1 = z2d.shape[0] - max(0, dy)
        src_x0 = max(0, -dx)
        src_x1 = z2d.shape[1] - max(0, dx)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = z2d[src_y0:src_y1, src_x0:src_x1]
        step = np.maximum(step, np.abs(z2d - shifted))

    return {"z": z2d, "slope": slope, "roughness": roughness, "step": step}


def build_costmap(
    features: Dict[str, np.ndarray],
    params: Optional[DEMPlannerParams | Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build traversal costmap and blocked mask from feature maps."""
    cfg = DEMPlannerParams.from_any(params)
    slope = features["slope"]
    roughness = features["roughness"]
    step = features["step"]

    slope_c = np.clip(slope, 0.0, cfg.s_max)
    roughness_c = np.clip(roughness, 0.0, cfg.r_max)
    step_c = np.clip(step, 0.0, cfg.d_max)

    costmap = (cfg.base_cost + cfg.w_s * slope_c + cfg.w_r * roughness_c + cfg.w_d * step_c).astype(
        np.float32
    )
    blocked = (slope > cfg.hard_slope_max) | (step > cfg.hard_step_max)
    return costmap, blocked


def _euclidean(a: GridPoint, b: GridPoint) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def _bresenham_cells(a: GridPoint, b: GridPoint) -> Iterable[GridPoint]:
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def line_of_sight(blocked: np.ndarray, a: GridPoint, b: GridPoint) -> bool:
    """True if all Bresenham cells from a to b are valid."""
    h, w = blocked.shape
    for x, y in _bresenham_cells(a, b):
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        if blocked[y, x]:
            return False
    return True


def _reconstruct_path(parent: Dict[GridPoint, GridPoint], goal: GridPoint) -> List[GridPoint]:
    p = [goal]
    cur = goal
    while cur in parent:
        cur = parent[cur]
        p.append(cur)
    p.reverse()
    return p


def _edge_cost(costmap: np.ndarray, u: GridPoint, v: GridPoint) -> float:
    d = _euclidean(u, v)
    return d * 0.5 * float(costmap[u[1], u[0]] + costmap[v[1], v[0]])


def astar(
    costmap: np.ndarray,
    blocked: np.ndarray,
    start: GridPoint,
    goal: GridPoint,
    use_theta_star: bool = False,
    connectivity: int = 8,
) -> Tuple[List[GridPoint], float]:
    """Run A* (or Theta* if use_theta_star=True) on grid."""
    h, w = costmap.shape
    if blocked.shape != costmap.shape:
        raise ValueError("blocked mask shape must match costmap")
    start = clamp_grid(start, w, h)
    goal = clamp_grid(goal, w, h)
    if blocked[start[1], start[0]] or blocked[goal[1], goal[0]]:
        return [], float("inf")

    if connectivity == 8:
        moves = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    elif connectivity == 4:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    g: Dict[GridPoint, float] = {start: 0.0}
    parent: Dict[GridPoint, GridPoint] = {}
    open_heap: List[Tuple[float, GridPoint]] = []
    heapq.heappush(open_heap, (_euclidean(start, goal), start))
    closed: set[GridPoint] = set()

    while open_heap:
        _, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        if u == goal:
            return _reconstruct_path(parent, goal), g[goal]
        closed.add(u)

        ux, uy = u
        for dx, dy in moves:
            vx, vy = ux + dx, uy + dy
            if vx < 0 or vy < 0 or vx >= w or vy >= h:
                continue
            v = (vx, vy)
            if blocked[vy, vx] or v in closed:
                continue

            best_parent = u
            cand_cost = g[u] + _edge_cost(costmap, u, v)

            if use_theta_star and u in parent:
                pu = parent[u]
                if line_of_sight(blocked, pu, v):
                    alt_cost = g[pu] + _edge_cost(costmap, pu, v)
                    if alt_cost < cand_cost:
                        best_parent = pu
                        cand_cost = alt_cost

            if cand_cost < g.get(v, float("inf")):
                g[v] = cand_cost
                parent[v] = best_parent
                f = cand_cost + _euclidean(v, goal)
                heapq.heappush(open_heap, (f, v))

    return [], float("inf")


def simplify_collinear(path: Sequence[GridPoint]) -> List[GridPoint]:
    """Remove strictly collinear intermediate points."""
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    for i in range(1, len(path) - 1):
        a = out[-1]
        b = path[i]
        c = path[i + 1]
        abx = b[0] - a[0]
        aby = b[1] - a[1]
        bcx = c[0] - b[0]
        bcy = c[1] - b[1]
        cross = abx * bcy - aby * bcx
        if cross == 0:
            continue
        out.append(b)
    out.append(path[-1])
    return out


def _polyline_cost(path: Sequence[GridPoint], costmap: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(_edge_cost(costmap, path[i], path[i + 1]) for i in range(len(path) - 1)))


def shortcut_smooth_path(
    path: Sequence[GridPoint],
    blocked: np.ndarray,
    costmap: np.ndarray,
    iters: int = 200,
    seed: int = 0,
) -> List[GridPoint]:
    """Randomized shortcutting that preserves validity and lowers path cost."""
    if len(path) <= 2:
        return list(path)
    rng = np.random.default_rng(seed)
    p = list(path)
    for _ in range(max(0, iters)):
        if len(p) <= 2:
            break
        i = int(rng.integers(0, len(p) - 2))
        j = int(rng.integers(i + 2, len(p)))
        a, b = p[i], p[j]
        if not line_of_sight(blocked, a, b):
            continue
        old_seg = p[i : j + 1]
        old_cost = _polyline_cost(old_seg, costmap)
        new_cost = _edge_cost(costmap, a, b)
        if new_cost + 1e-6 < old_cost:
            p = p[: i + 1] + [b] + p[j + 1 :]
    return p


def _resample_polyline_world(path_xy: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(path_xy) <= 1:
        return path_xy
    seg = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1)
    s = np.hstack(([0.0], np.cumsum(seg)))
    if s[-1] <= spacing_m:
        return path_xy
    s_new = np.arange(0.0, s[-1] + 1e-9, spacing_m)
    x = np.interp(s_new, s, path_xy[:, 0])
    y = np.interp(s_new, s, path_xy[:, 1])
    return np.column_stack([x, y])


def _spline_world_path(
    path_world: Sequence[WorldPoint], spacing_m: float, tension: float = 1.0
) -> np.ndarray:
    xy = np.asarray(path_world, dtype=np.float32)
    if len(xy) < 3 or not _HAS_SCIPY:
        return _resample_polyline_world(xy, spacing_m)
    seg = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    s = np.hstack(([0.0], np.cumsum(seg)))
    if s[-1] <= spacing_m:
        return xy
    s_new = np.arange(0.0, s[-1] + 1e-9, spacing_m)
    # Lower bc smoothness indirectly by blending toward polyline with "tension"
    cx = CubicSpline(s, xy[:, 0], bc_type="natural")
    cy = CubicSpline(s, xy[:, 1], bc_type="natural")
    spline_xy = np.column_stack([cx(s_new), cy(s_new)])
    if tension < 1.0:
        poly_xy = _resample_polyline_world(xy, spacing_m)
        n = min(len(poly_xy), len(spline_xy))
        spline_xy = tension * spline_xy[:n] + (1.0 - tension) * poly_xy[:n]
    return spline_xy


def _path_valid_world(path_world: np.ndarray, blocked: np.ndarray, cell_size: float) -> bool:
    h, w = blocked.shape
    if len(path_world) < 2:
        return False
    for i in range(len(path_world) - 1):
        x0, y0 = path_world[i]
        x1, y1 = path_world[i + 1]
        a = clamp_grid(world_to_grid(float(x0), float(y0), cell_size, shape=blocked.shape), w, h)
        b = clamp_grid(world_to_grid(float(x1), float(y1), cell_size, shape=blocked.shape), w, h)
        if not line_of_sight(blocked, a, b):
            return False
    return True


def smooth_path(
    path: Sequence[GridPoint],
    blocked: np.ndarray,
    costmap: np.ndarray,
    cell_size: float = CELL_WIDTH,
    simplify: bool = True,
    do_shortcut: bool = True,
    shortcut_iters: int = 200,
    do_spline: bool = True,
    spline_spacing_m: float = 0.25,
    spline_tension: float = 1.0,
    seed: int = 0,
) -> List[WorldPoint]:
    """Post-process a grid path into smooth world-coordinate waypoints."""
    if not path:
        return []
    p = list(path)
    if simplify:
        p = simplify_collinear(p)
    if do_shortcut:
        p = shortcut_smooth_path(p, blocked, costmap, iters=shortcut_iters, seed=seed)
    world = [grid_to_world(x, y, cell_size, shape=blocked.shape) for x, y in p]

    if do_spline and len(world) >= 3:
        spline_world = _spline_world_path(world, spacing_m=spline_spacing_m, tension=spline_tension)
        if _path_valid_world(spline_world, blocked, cell_size):
            return [(float(x), float(y)) for x, y in spline_world]
    return world


def _parse_xy(
    xy: Sequence[float | int],
    shape: Tuple[int, int],
    cell_size: float,
    input_is_grid: bool,
) -> GridPoint:
    if len(xy) != 2:
        raise ValueError("start/goal must have 2 values")
    h, w = shape
    if input_is_grid:
        ix, iy = int(xy[0]), int(xy[1])
    else:
        ix, iy = world_to_grid(float(xy[0]), float(xy[1]), cell_size, shape=shape)
    return clamp_grid((ix, iy), w, h)


def _parse_xy_unclamped(
    xy: Sequence[float | int],
    shape: Tuple[int, int],
    cell_size: float,
    input_is_grid: bool,
) -> GridPoint:
    if len(xy) != 2:
        raise ValueError("start/goal must have 2 values")
    if input_is_grid:
        return int(xy[0]), int(xy[1])
    return world_to_grid(float(xy[0]), float(xy[1]), cell_size, shape=shape)


def _grid_connected(
    blocked: np.ndarray, start: GridPoint, goal: GridPoint, connectivity: int = 8
) -> bool:
    if start == goal:
        return True
    h, w = blocked.shape
    if connectivity == 8:
        moves = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    elif connectivity == 4:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    queue: List[GridPoint] = [start]
    seen = {start}
    q_idx = 0
    while q_idx < len(queue):
        ux, uy = queue[q_idx]
        q_idx += 1
        for dx, dy in moves:
            vx, vy = ux + dx, uy + dy
            if vx < 0 or vy < 0 or vx >= w or vy >= h:
                continue
            v = (vx, vy)
            if v in seen or blocked[vy, vx]:
                continue
            if v == goal:
                return True
            seen.add(v)
            queue.append(v)
    return False


def plan_path_dem(
    z: np.ndarray,
    cell_size: float = CELL_WIDTH,
    start_xy: Sequence[float | int] = (0, 0),
    goal_xy: Sequence[float | int] = (10, 10),
    params: Optional[DEMPlannerParams | Dict[str, Any]] = None,
    use_theta_star: bool = True,
    do_smooth: bool = True,
    input_is_grid: bool = False,
) -> Tuple[List[WorldPoint], float, Dict[str, Any]]:
    """Plan a path on DEM and return world waypoints, total cost, and debug maps."""
    cfg = DEMPlannerParams.from_any(params)
    z2d = _normalize_dem(z)
    h, w = z2d.shape

    features = compute_features(
        z2d,
        cell_size=cell_size,
        roughness_window=cfg.roughness_window,
        use_eight_neighbor_step=cfg.use_eight_neighbor_step,
    )
    costmap, blocked = build_costmap(features, cfg)
    terrain_blocked = blocked.copy()
    keepout_mask = np.zeros_like(blocked, dtype=bool)
    if cfg.use_lander_keepout:
        keepout_mask = lander_keepout_mask(
            shape=blocked.shape,
            cell_size=cell_size,
            lander_center_xy_m=cfg.lander_center_xy_m,
            buffer_m=cfg.lander_buffer_m,
        )
        blocked = blocked | keepout_mask

    start_unclamped = _parse_xy_unclamped(
        start_xy, shape=z2d.shape, cell_size=cell_size, input_is_grid=input_is_grid
    )
    goal_unclamped = _parse_xy_unclamped(
        goal_xy, shape=z2d.shape, cell_size=cell_size, input_is_grid=input_is_grid
    )
    start = clamp_grid(start_unclamped, w, h)
    goal = clamp_grid(goal_unclamped, w, h)

    start_reasons: List[str] = []
    goal_reasons: List[str] = []
    failure_reasons: List[str] = []
    if start_unclamped != start:
        failure_reasons.append(f"start clamped from {start_unclamped} to {start}")
    if goal_unclamped != goal:
        failure_reasons.append(f"goal clamped from {goal_unclamped} to {goal}")

    def _collect_block_reasons(cell: GridPoint) -> List[str]:
        x, y = cell
        reasons: List[str] = []
        if cfg.use_lander_keepout and keepout_mask[y, x]:
            reasons.append("inside lander keepout")
        if terrain_blocked[y, x]:
            slope_val = float(features["slope"][y, x])
            step_val = float(features["step"][y, x])
            if slope_val > cfg.hard_slope_max:
                reasons.append(f"slope {slope_val:.3f} > hard_slope_max {cfg.hard_slope_max:.3f}")
            if step_val > cfg.hard_step_max:
                reasons.append(f"step {step_val:.3f} > hard_step_max {cfg.hard_step_max:.3f}")
        if blocked[y, x] and not reasons:
            reasons.append("blocked")
        return reasons

    start_reasons = _collect_block_reasons(start)
    goal_reasons = _collect_block_reasons(goal)
    if start_reasons:
        failure_reasons.append("start blocked: " + ", ".join(start_reasons))
    if goal_reasons:
        failure_reasons.append("goal blocked: " + ", ".join(goal_reasons))

    path_grid, total_cost = astar(
        costmap=costmap,
        blocked=blocked,
        start=start,
        goal=goal,
        use_theta_star=use_theta_star,
        connectivity=cfg.connectivity,
    )
    if not path_grid:
        connected = None
        if not start_reasons and not goal_reasons:
            connected = _grid_connected(blocked, start, goal, connectivity=cfg.connectivity)
            if not connected:
                failure_reasons.append("no free-space connection between start and goal")
            else:
                failure_reasons.append("A*/Theta* failed despite connected free space")
        debug = {
            "features": features,
            "costmap": costmap,
            "blocked": blocked,
            "path_grid": [],
            "start_grid": start,
            "goal_grid": goal,
            "failure_reasons": failure_reasons,
            "diagnostics": {
                "start_grid_unclamped": start_unclamped,
                "goal_grid_unclamped": goal_unclamped,
                "start_block_reasons": start_reasons,
                "goal_block_reasons": goal_reasons,
                "free_space_connected": connected,
                "keepout_enabled": cfg.use_lander_keepout,
            },
        }
        return [], float("inf"), debug

    if do_smooth:
        path_world = smooth_path(
            path_grid,
            blocked=blocked,
            costmap=costmap,
            cell_size=cell_size,
            simplify=cfg.simplify_collinear,
            do_shortcut=cfg.do_shortcut,
            shortcut_iters=cfg.shortcut_iters,
            do_spline=cfg.do_spline,
            spline_spacing_m=cfg.spline_spacing_m,
            spline_tension=cfg.spline_tension,
            seed=cfg.seed,
        )
    else:
        path_world = [grid_to_world(x, y, cell_size, shape=z2d.shape) for x, y in path_grid]

    debug = {
        "features": features,
        "costmap": costmap,
        "blocked": blocked,
        "path_grid": path_grid,
        "start_grid": start,
        "goal_grid": goal,
        "failure_reasons": [],
        "diagnostics": {
            "start_grid_unclamped": start_unclamped,
            "goal_grid_unclamped": goal_unclamped,
            "start_block_reasons": start_reasons,
            "goal_block_reasons": goal_reasons,
            "free_space_connected": True,
            "keepout_enabled": cfg.use_lander_keepout,
        },
    }
    return path_world, float(total_cost), debug
