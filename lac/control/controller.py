"""Trajectory tracking controller"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import lac.params as params
from lac.planning.path_smoother import Trajectory2D
from lac.util import pose_to_pos_rpy, wrap_angle


@dataclass
class TrajectoryTrackerConfig:
    """Tuning parameters for trajectory tracking."""

    target_speed: float = params.TARGET_SPEED
    max_v: float = 0.6
    max_w: float = params.MAX_STEER
    max_accel: float = 0.35
    max_decel: float = 0.45
    max_w_accel: float = 1.5

    # Lookahead target selection
    lookahead_time: float = 1.0
    min_lookahead: float = 0.35
    max_lookahead: float = 1.8

    # Feedback gains
    k_heading: float = 1.8
    k_cross_track: float = 1.2
    k_goal_v: float = 0.8

    # Stop behavior at trajectory end
    stop_pos_tol: float = 0.20
    stop_heading_tol: float = 0.20

    allow_reverse: bool = False
    default_dt: float = params.DT


class TrajectoryTracker:
    """Track a 2D trajectory and output linear/angular velocity commands.

    The expected reference is `Trajectory2D` from `PathSmoother`, but this class
    also accepts:
      - dict with keys: `xyt` and optionally `v`, `w`, `t`
      - ndarray shaped (N,3) interpreted as [x, y, theta]
    """

    def __init__(self, config: Optional[TrajectoryTrackerConfig] = None):
        self.cfg = config or TrajectoryTrackerConfig()
        self._last_v_cmd = 0.0
        self._last_w_cmd = 0.0
        self._last_target_idx = 0
        self.last_debug: dict[str, Any] = {}

    def reset(self) -> None:
        self._last_v_cmd = 0.0
        self._last_w_cmd = 0.0
        self._last_target_idx = 0
        self.last_debug = {}

    def _extract_pose_xyyaw(self, pose: np.ndarray) -> tuple[float, float, float]:
        pose_arr = np.asarray(pose)
        if pose_arr.shape == (4, 4):
            pos, rpy = pose_to_pos_rpy(pose_arr)
            return float(pos[0]), float(pos[1]), float(rpy[2])
        if pose_arr.shape[0] >= 3:
            return float(pose_arr[0]), float(pose_arr[1]), float(pose_arr[2])
        raise ValueError(f"pose must be 4x4 transform or [x,y,theta], got shape {pose_arr.shape}")

    def _parse_reference(
        self, reference: Trajectory2D | dict[str, Any] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(reference, Trajectory2D):
            xyt = np.asarray(reference.xyt, dtype=np.float64)
            t = np.asarray(reference.t, dtype=np.float64)
            v = np.asarray(reference.v, dtype=np.float64)
            w = np.asarray(reference.w, dtype=np.float64)
        elif isinstance(reference, dict):
            xyt = np.asarray(reference["xyt"], dtype=np.float64)
            n = len(xyt)
            t = np.asarray(reference.get("t", np.arange(n, dtype=np.float64) * self.cfg.default_dt))
            v = np.asarray(reference.get("v", np.full(n, self.cfg.target_speed, dtype=np.float64)))
            w = np.asarray(reference.get("w", np.zeros(n, dtype=np.float64)))
        else:
            xyt = np.asarray(reference, dtype=np.float64)
            n = len(xyt)
            t = np.arange(n, dtype=np.float64) * self.cfg.default_dt
            v = np.full(n, self.cfg.target_speed, dtype=np.float64)
            w = np.zeros(n, dtype=np.float64)

        if xyt.ndim != 2 or xyt.shape[1] < 3:
            raise ValueError(f"reference xyt must be shape (N,>=3), got {xyt.shape}")
        if len(xyt) == 0:
            raise ValueError("reference trajectory is empty")
        return xyt[:, :3], t, v, w

    @staticmethod
    def _clip_rate(
        target: float, prev: float, up_rate: float, down_rate: float, dt: float
    ) -> float:
        if target >= prev:
            return min(target, prev + up_rate * dt)
        return max(target, prev - down_rate * dt)

    def compute_command(
        self,
        reference: Trajectory2D | dict[str, Any] | np.ndarray,
        current_pose: np.ndarray,
        current_time_s: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> tuple[float, float]:
        """Compute (v_cmd, w_cmd) to track a reference trajectory."""
        dt_s = float(self.cfg.default_dt if dt is None else dt)
        xyt_ref, t_ref, v_ref_arr, w_ref_arr = self._parse_reference(reference)
        xy_ref = xyt_ref[:, :2]

        x, y, yaw = self._extract_pose_xyyaw(current_pose)
        pos = np.array([x, y], dtype=np.float64)

        if current_time_s is not None:
            idx = int(np.clip(np.searchsorted(t_ref, current_time_s), 0, len(xy_ref) - 1))
        else:
            start = int(np.clip(self._last_target_idx, 0, len(xy_ref) - 1))
            local = xy_ref[start:]
            nearest_local = int(np.argmin(np.linalg.norm(local - pos[None, :], axis=1)))
            idx = start + nearest_local

        v_ref = float(np.clip(v_ref_arr[idx], 0.0, self.cfg.max_v))
        lookahead = float(
            np.clip(
                self.cfg.lookahead_time * max(v_ref, 0.05),
                self.cfg.min_lookahead,
                self.cfg.max_lookahead,
            )
        )

        target_idx = idx
        accum = 0.0
        while target_idx + 1 < len(xy_ref) and accum < lookahead:
            ds = float(np.linalg.norm(xy_ref[target_idx + 1] - xy_ref[target_idx]))
            accum += ds
            target_idx += 1
        self._last_target_idx = target_idx

        target_xy = xy_ref[target_idx]
        target_theta = float(xyt_ref[target_idx, 2])
        goal_xy = xy_ref[-1]
        goal_theta = float(xyt_ref[-1, 2])

        dx = float(target_xy[0] - x)
        dy = float(target_xy[1] - y)
        heading_to_target = float(np.arctan2(dy, dx))
        heading_err = float(wrap_angle(heading_to_target - yaw))
        heading_path_err = float(wrap_angle(target_theta - yaw))
        cross_track_err = float(-np.sin(yaw) * dx + np.cos(yaw) * dy)

        dist_to_goal = float(np.linalg.norm(goal_xy - pos))
        heading_goal_err = float(wrap_angle(goal_theta - yaw))
        reached_goal = (
            dist_to_goal <= self.cfg.stop_pos_tol
            and abs(heading_goal_err) <= self.cfg.stop_heading_tol
        )

        if reached_goal:
            v_des = 0.0
            w_des = 0.0
        else:
            v_des = min(v_ref * np.cos(heading_err), self.cfg.k_goal_v * max(dist_to_goal, 0.0))
            if not self.cfg.allow_reverse:
                v_des = max(v_des, 0.0)
            w_ff = float(w_ref_arr[target_idx]) if target_idx < len(w_ref_arr) else 0.0
            w_des = (
                w_ff
                + self.cfg.k_heading * heading_path_err
                + self.cfg.k_cross_track * cross_track_err
            )

        v_des = float(
            np.clip(v_des, -self.cfg.max_v if self.cfg.allow_reverse else 0.0, self.cfg.max_v)
        )
        w_des = float(np.clip(w_des, -self.cfg.max_w, self.cfg.max_w))

        v_cmd = self._clip_rate(
            v_des, self._last_v_cmd, self.cfg.max_accel, self.cfg.max_decel, dt_s
        )
        w_cmd = self._clip_rate(
            w_des, self._last_w_cmd, self.cfg.max_w_accel, self.cfg.max_w_accel, dt_s
        )
        w_cmd = float(np.clip(w_cmd, -self.cfg.max_w, self.cfg.max_w))

        self._last_v_cmd = v_cmd
        self._last_w_cmd = w_cmd
        self.last_debug = {
            "nearest_idx": idx,
            "target_idx": target_idx,
            "lookahead_m": lookahead,
            "target_xy": target_xy,
            "dist_to_goal_m": dist_to_goal,
            "heading_err_rad": heading_err,
            "heading_path_err_rad": heading_path_err,
            "cross_track_err_m": cross_track_err,
            "reached_goal": reached_goal,
            "v_des": v_des,
            "w_des": w_des,
            "v_cmd": v_cmd,
            "w_cmd": w_cmd,
        }
        return v_cmd, w_cmd
