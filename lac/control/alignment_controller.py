"""Ground-truth pose alignment controller for unicycle rover."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


@dataclass
class AlignmentControllerConfig:
    k_rho: float = 0.8
    k_alpha: float = 1.2
    k_theta: float = 0.6
    k_theta_final: float = 1.2
    k_rho_final: float = 0.0

    v_max: float = 0.3
    w_max: float = 0.6
    forward_only: bool = False

    rho_final: float = 0.2
    dv_max: float = 0.8  # m/s^2
    dw_max: float = 1.5  # rad/s^2

    r_tol: float = 0.08
    theta_tol: float = 0.12
    hold_frames: int = 12


class AlignmentController:
    def __init__(self, cfg: AlignmentControllerConfig | None = None):
        self.cfg = cfg or AlignmentControllerConfig()
        self.reset()

    def reset(self) -> None:
        self.v_prev = 0.0
        self.w_prev = 0.0
        self._good_count = 0

    def step(
        self,
        current_xyt: Sequence[float],
        target_xyt: Sequence[float],
        dt: float,
    ) -> tuple[float, float, bool, dict]:
        x, y, theta = float(current_xyt[0]), float(current_xyt[1]), float(current_xyt[2])
        xg, yg, thetag = float(target_xyt[0]), float(target_xyt[1]), float(target_xyt[2])

        dx = xg - x
        dy = yg - y
        rho = float(np.hypot(dx, dy))

        alpha = wrap_angle(float(np.arctan2(dy, dx)) - theta)
        e_theta = wrap_angle(thetag - theta)

        if rho < self.cfg.rho_final:
            v_des = self.cfg.k_rho_final * rho
            w_des = self.cfg.k_theta_final * e_theta
            phase = "yaw_lock"
        else:
            v_des = self.cfg.k_rho * rho * np.cos(alpha)
            w_des = self.cfg.k_alpha * alpha + self.cfg.k_theta * e_theta
            phase = "pose_drive"

        v_des = clamp(v_des, -self.cfg.v_max, self.cfg.v_max)
        w_des = clamp(w_des, -self.cfg.w_max, self.cfg.w_max)

        if self.cfg.forward_only:
            v_des = max(v_des, 0.0)

        # Rate limits for terrain robustness.
        dt_safe = max(float(dt), 1e-4)
        dv_lim = self.cfg.dv_max * dt_safe
        dw_lim = self.cfg.dw_max * dt_safe
        v_cmd = self.v_prev + clamp(v_des - self.v_prev, -dv_lim, dv_lim)
        w_cmd = self.w_prev + clamp(w_des - self.w_prev, -dw_lim, dw_lim)
        v_cmd = clamp(v_cmd, -self.cfg.v_max, self.cfg.v_max)
        w_cmd = clamp(w_cmd, -self.cfg.w_max, self.cfg.w_max)
        self.v_prev, self.w_prev = v_cmd, w_cmd

        good = (rho < self.cfg.r_tol) and (abs(e_theta) < self.cfg.theta_tol)
        self._good_count = self._good_count + 1 if good else 0
        aligned = self._good_count >= self.cfg.hold_frames
        if aligned:
            v_cmd, w_cmd = 0.0, 0.0

        debug = {
            "phase": phase,
            "dx": dx,
            "dy": dy,
            "rho": rho,
            "alpha": alpha,
            "e_theta": e_theta,
            "v_des": float(v_des),
            "w_des": float(w_des),
            "v_cmd": float(v_cmd),
            "w_cmd": float(w_cmd),
            "good_count": int(self._good_count),
            "aligned": bool(aligned),
        }
        return float(v_cmd), float(w_cmd), bool(aligned), debug
