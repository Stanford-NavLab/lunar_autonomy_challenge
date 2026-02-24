"""Heading-aware planner built on DEM planner + path smoother."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lac.planning.dem_planner import DEMPlannerParams, plan_path_dem
from lac.planning.path_smoother import Path2D, PathSmoother, PathSmootherConfig, Trajectory2D
from lac.params import CELL_WIDTH


@dataclass
class HeadingAwarePlannerParams:
    """Configuration for heading-aware planning via existing modules."""

    dem_params: DEMPlannerParams = field(default_factory=DEMPlannerParams)
    smoother_config: PathSmootherConfig = field(default_factory=PathSmootherConfig)
    use_theta_star: bool = True
    do_dem_post_smooth: bool = True


@dataclass
class HeadingAwarePlanResult:
    path_xy: List[Tuple[float, float]]
    path_xyt: np.ndarray
    trajectory: Trajectory2D
    total_cost: float
    debug: Dict[str, Any]


class HeadingAwarePlanner:
    """Compose DEM planner and path smoother for heading-aware trajectories."""

    def __init__(self, config: Optional[HeadingAwarePlannerParams] = None):
        self.cfg = config or HeadingAwarePlannerParams()

    def plan(
        self,
        z: np.ndarray,
        start_xy: Sequence[float],
        start_yaw_rad: float,
        goal_xy: Sequence[float],
        goal_yaw_rad: float,
        cell_size: float = CELL_WIDTH,
    ) -> HeadingAwarePlanResult:
        """Plan heading-aware path by composing DEM planner and smoother."""
        if not np.isclose(cell_size, CELL_WIDTH):
            raise ValueError(
                "HeadingAwarePlanner currently expects cell_size == params.CELL_WIDTH."
            )
        cfg = self.cfg
        z2d = z.astype(np.float32, copy=False) if z.ndim == 2 else z[:, :, 2].astype(np.float32)

        initial_pose = np.array(
            [float(start_xy[0]), float(start_xy[1]), float(start_yaw_rad)],
            dtype=np.float64,
        )
        path_world, total_cost, dem_debug = plan_path_dem(
            z2d,
            cell_size=cell_size,
            start_xy=start_xy,
            goal_xy=goal_xy,
            params=cfg.dem_params,
            use_theta_star=cfg.use_theta_star,
            do_smooth=cfg.do_dem_post_smooth,
            input_is_grid=False,
            initial_pose=initial_pose,
        )
        if len(path_world) == 0:
            raise RuntimeError(
                "DEM planner failed before heading-aware smoothing. "
                f"Reasons: {dem_debug.get('failure_reasons', [])}"
            )

        smoother = PathSmoother(cfg.smoother_config)
        traj = smoother.smooth(
            Path2D(xy=np.asarray(path_world, dtype=np.float64), meta=dem_debug),
            z2d,
            initial_pose=initial_pose,
            goal_yaw_rad=float(goal_yaw_rad),
        )

        path_xyt = np.asarray(traj.xyt, dtype=np.float64)
        path_xy = [(float(p[0]), float(p[1])) for p in path_xyt[:, :2]]
        debug = dict(dem_debug)
        debug["heading_aware"] = {
            "method": "dem_planner_plus_path_smoother",
            "goal_yaw_rad": float(goal_yaw_rad),
            "trajectory_samples": int(len(path_xyt)),
            "path_samples": int(len(path_world)),
        }
        return HeadingAwarePlanResult(
            path_xy=path_xy,
            path_xyt=path_xyt,
            trajectory=traj,
            total_cost=float(total_cost),
            debug=debug,
        )
