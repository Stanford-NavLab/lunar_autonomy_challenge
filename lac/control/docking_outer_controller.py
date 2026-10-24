"""Outer-loop docking controller that emits local trajectory references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


@dataclass
class DockingOuterControllerConfig:
    rotate_enter_rad: float = 0.30
    rotate_exit_rad: float = 0.12
    final_align_radius_m: float = 0.20

    r_tol: float = 0.08
    theta_tol: float = 0.12
    hold_frames: int = 12

    carrot_dist_m: float = 0.35
    slowdown_radius_m: float = 1.0
    v_approach_max: float = 0.28
    v_approach_min: float = 0.08

    search_w_ref: float = 0.20

    stagnation_timeout_s: float = 2.5
    stagnation_progress_eps_m: float = 0.015
    recovery_reverse_dist_m: float = 0.25
    recovery_duration_s: float = 1.2

    v_cmd_max: float = 0.35
    w_cmd_max: float = 0.50
    min_v_effective: float = 0.06
    min_w_effective: float = 0.10
    dv_max: float = 0.35
    dw_max: float = 0.70


class DockingOuterController:
    """State machine that produces local references for trajectory tracking."""

    def __init__(self, cfg: DockingOuterControllerConfig | None = None):
        self.cfg = cfg or DockingOuterControllerConfig()
        self.reset()

    def reset(self) -> None:
        self.phase = "SEARCH"
        self.hold_count = 0
        self.time_s = 0.0
        self.last_progress_time_s = 0.0
        self.best_rho = np.inf
        self.recovery_until_s = -1.0
        self.v_prev = 0.0
        self.w_prev = 0.0

    def _make_ref(
        self,
        p0: tuple[float, float, float],
        p1: tuple[float, float, float],
        v1: float,
        dt: float,
    ) -> dict:
        dt_s = max(float(dt), 1e-3)
        xyt = np.array([p0, p1], dtype=np.float64)
        t = np.array([0.0, dt_s], dtype=np.float64)
        v = np.array([0.0, float(max(v1, 0.0))], dtype=np.float64)
        w = np.zeros(2, dtype=np.float64)
        return {"xyt": xyt, "t": t, "v": v, "w": w}

    def _apply_command_shaping(self, v_cmd: float, w_cmd: float, dt: float) -> tuple[float, float]:
        cfg = self.cfg
        dt_s = max(float(dt), 1e-4)

        v_des = clamp(float(v_cmd), -cfg.v_cmd_max, cfg.v_cmd_max)
        w_des = clamp(float(w_cmd), -cfg.w_cmd_max, cfg.w_cmd_max)

        dv_lim = cfg.dv_max * dt_s
        dw_lim = cfg.dw_max * dt_s
        v_rate = self.v_prev + clamp(v_des - self.v_prev, -dv_lim, dv_lim)
        w_rate = self.w_prev + clamp(w_des - self.w_prev, -dw_lim, dw_lim)

        if abs(v_rate) > 1e-6 and abs(v_rate) < cfg.min_v_effective:
            v_rate = np.sign(v_rate) * cfg.min_v_effective
        if abs(w_rate) > 1e-6 and abs(w_rate) < cfg.min_w_effective:
            w_rate = np.sign(w_rate) * cfg.min_w_effective

        v_rate = clamp(v_rate, -cfg.v_cmd_max, cfg.v_cmd_max)
        w_rate = clamp(w_rate, -cfg.w_cmd_max, cfg.w_cmd_max)
        self.v_prev, self.w_prev = float(v_rate), float(w_rate)
        return float(v_rate), float(w_rate)

    def shape_command(self, v_cmd: float, w_cmd: float, dt: float) -> tuple[float, float]:
        return self._apply_command_shaping(v_cmd, w_cmd, dt)

    def step(
        self,
        current_xyt: Sequence[float],
        target_xyt: Sequence[float],
        dt: float,
        target_visible: bool = True,
    ) -> dict:
        cfg = self.cfg
        dt_s = max(float(dt), 1e-4)
        self.time_s += dt_s

        x, y, yaw = float(current_xyt[0]), float(current_xyt[1]), float(current_xyt[2])
        xg, yg, yaw_g = float(target_xyt[0]), float(target_xyt[1]), float(target_xyt[2])

        dx = xg - x
        dy = yg - y
        rho = float(np.hypot(dx, dy))
        bearing = float(np.arctan2(dy, dx))
        bearing_err = wrap_angle(bearing - yaw)
        e_theta = wrap_angle(yaw_g - yaw)

        good = (rho < cfg.r_tol) and (abs(e_theta) < cfg.theta_tol)
        self.hold_count = self.hold_count + 1 if good else 0
        success = self.hold_count >= cfg.hold_frames
        if success:
            return {
                "phase": "SUCCESS",
                "reference": None,
                "success": True,
                "rho": rho,
                "bearing_err": bearing_err,
                "e_theta": e_theta,
                "hold_count": int(self.hold_count),
                "stagnating": False,
                "recovering": False,
            }

        if rho < self.best_rho - cfg.stagnation_progress_eps_m:
            self.best_rho = rho
            self.last_progress_time_s = self.time_s
        stagnating = (self.time_s - self.last_progress_time_s) > cfg.stagnation_timeout_s
        if stagnating and self.recovery_until_s < self.time_s:
            self.recovery_until_s = self.time_s + cfg.recovery_duration_s

        recovering = self.time_s < self.recovery_until_s

        if recovering:
            self.phase = "APPROACH"
            x_back = x - cfg.recovery_reverse_dist_m * np.cos(yaw)
            y_back = y - cfg.recovery_reverse_dist_m * np.sin(yaw)
            ref = self._make_ref((x, y, yaw), (x_back, y_back, yaw), v1=cfg.v_approach_min, dt=dt_s)
        elif not target_visible:
            self.phase = "SEARCH"
            theta_search = yaw + np.sign(cfg.search_w_ref) * 0.6
            ref = self._make_ref((x, y, yaw), (x, y, theta_search), v1=0.0, dt=dt_s)
        elif rho <= cfg.final_align_radius_m:
            self.phase = "FINAL_ALIGN"
            ref = self._make_ref((x, y, yaw), (x, y, yaw_g), v1=0.0, dt=dt_s)
        else:
            # Rotate gate with hysteresis to avoid turn-drive oscillations.
            if self.phase in ("SEARCH", "ROTATE_TO_TARGET"):
                rotate_gate = abs(bearing_err) > cfg.rotate_exit_rad
            else:
                rotate_gate = abs(bearing_err) > cfg.rotate_enter_rad

            if rotate_gate:
                self.phase = "ROTATE_TO_TARGET"
                ref = self._make_ref((x, y, yaw), (x, y, bearing), v1=0.0, dt=dt_s)
            else:
                self.phase = "APPROACH"
                step_len = min(cfg.carrot_dist_m, rho)
                x_ref = x + step_len * np.cos(bearing)
                y_ref = y + step_len * np.sin(bearing)
                speed_scale = np.clip(rho / max(cfg.slowdown_radius_m, 1e-3), 0.0, 1.0)
                v_ref = cfg.v_approach_min + (cfg.v_approach_max - cfg.v_approach_min) * speed_scale
                ref = self._make_ref((x, y, yaw), (x_ref, y_ref, bearing), v1=float(v_ref), dt=dt_s)

        return {
            "phase": self.phase,
            "reference": ref,
            "success": False,
            "rho": rho,
            "bearing_err": bearing_err,
            "e_theta": e_theta,
            "hold_count": int(self.hold_count),
            "stagnating": bool(stagnating),
            "recovering": bool(recovering),
        }
