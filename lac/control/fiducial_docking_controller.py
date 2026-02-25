"""Multi-stage fiducial visual-servo docking controller.

Implements:
SEARCH -> ROTATE -> APPROACH -> FINAL_YAW (optional) -> DONE
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from lac.control.lateral_drift_compensator import (
    LateralDriftCompensator,
    LateralDriftCompensatorConfig,
)


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def smoothstep01(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def yaw_from_R(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


@dataclass
class FiducialDockingConfig:
    # Desired virtual box in base frame relative to tag.
    x_des: float = 1.0
    y_des: float = 0.0
    psi_des: float = 0.0
    use_final_yaw: bool = True
    use_yaw_term_in_approach: bool = True

    # Stage transitions.
    n_detect: int = 2
    n_gate: int = 3
    alpha_gate: float = np.deg2rad(8.0)
    # Optional override to enter approach when close even if alpha gate is not met.
    # Set <= 0 to disable override and enforce strict rotate gating.
    early_approach_rho_override_m: float = 0.0
    rho_final: float = 0.8
    y_final: float = 0.25

    # Core gains/limits.
    k_alpha: float = 1.2
    k_v: float = 0.45
    k_psi: float = 0.8
    k_psi_final: float = 1.0
    k_v_final: float = 0.20
    v_max_far: float = 0.20
    v_final_max: float = 0.08
    w_max: float = 0.50
    max_w_approach: float = 0.30
    w_final_max: float = 0.20
    w_search: float = 0.15
    # Sign bridge for angular command convention mismatches.
    # +1: canonical CCW-positive yaw command.
    # -1: flip controller-produced yaw commands.
    yaw_cmd_sign: float = 1.0
    k_slow: float = 0.35
    rho_close: float = 1.0
    # Keep some translation while trimming lateral error near x_des.
    lateral_engage_m: float = 0.20
    approach_min_speed: float = 0.04
    # Near x_des, prefer forward crawl for lateral correction stability.
    reverse_deadband_m: float = 0.12
    lateral_crawl_forward_bias: bool = True
    # Optional slip/lateral-drift compensation (modular helper).
    use_lateral_drift_comp: bool = False
    lateral_drift_beta: float = 0.08
    lateral_drift_max_m: float = 0.35
    lateral_drift_x_window_m: float = 0.70
    lateral_drift_min_speed_mps: float = 0.03
    lateral_drift_gain: float = 0.8
    lateral_drift_decay: float = 0.98

    # Robustness/timing.
    ema_beta: float = 0.30
    dropout_time: float = 0.25
    hold_last_meas_time: float = 0.15
    eps: float = 1e-3
    dv_max: float = 0.8
    dw_max: float = 1.2

    # Success criteria.
    x_tol: float = 0.50
    y_tol: float = 0.50
    psi_tol: float = np.deg2rad(30.0)
    hold_time_s: float = 0.4
    # If False, controller never self-latches to "done"; external checker decides completion.
    latch_success: bool = False

    # Backward-compatible aliases used by current project config.
    max_v: float = 0.20
    max_w: float = 0.50
    max_w_yaw_align: float = 0.20
    max_w_depth_correction: float = 0.15
    k_w_rot: float = 1.2
    k_v_x: float = 0.35
    k_w_y: float = 0.60
    rot_enter_tol_rad: float = np.deg2rad(20.0)
    rot_exit_tol_rad: float = np.deg2rad(14.0)
    depth_tol_m: float = 0.08
    lateral_tol_m: float = 0.10
    trim_nudge_v: float = 0.06
    trim_nudge_period_steps: int = 10
    antenna_xy_tol_m: float = 0.50
    antenna_rot_tol_rad: float = np.deg2rad(30.0)
    aligned_hold_frames: int = 8
    search_w: float = 0.15
    allow_reverse: bool = True


class FiducialDockingController:
    """State-machine visual-servo controller for front-virtual docking."""

    def __init__(self, config: FiducialDockingConfig | None = None):
        self.cfg = config or FiducialDockingConfig()
        self._drift_comp = LateralDriftCompensator()
        self.reset()

    def reset(self, initial_phase: str = "search") -> None:
        self.stage = initial_phase
        self.detect_count = 0
        self.gate_count = 0
        self.success_hold_s = 0.0
        self.success_latched = False
        self.time_since_seen = 1e9
        self.time_since_meas = 1e9
        self.last_meas: dict[str, float] | None = None
        self.ex_f = 0.0
        self.ey_f = 0.0
        self.epsi_f = 0.0
        self.alpha_f = 0.0
        self.rho_f = 0.0
        self.has_filter = False
        self.v_prev = 0.0
        self.w_prev = 0.0
        self._drift_comp.reset()

    def update(
        self,
        tag_detection: dict[str, Any] | None,
        T_base_cam: np.ndarray,
        dt: float,
        params: FiducialDockingConfig | None = None,
    ) -> tuple[float, float, dict[str, Any]]:
        """Update controller from tag pose in camera frame.

        tag_detection:
          - None, or dict with:
            * "T_cam_tag": (4,4), or
            * "R_cam_tag": (3,3) and "t_cam_tag": (3,)
        """
        cfg = self.cfg if params is None else params
        cfg = self._normalize_cfg(cfg)
        self._drift_comp.cfg = LateralDriftCompensatorConfig(
            enabled=bool(cfg.use_lateral_drift_comp),
            beta=float(cfg.lateral_drift_beta),
            max_bias_m=float(cfg.lateral_drift_max_m),
            x_window_m=float(cfg.lateral_drift_x_window_m),
            min_speed_mps=float(cfg.lateral_drift_min_speed_mps),
            gain=float(cfg.lateral_drift_gain),
            decay=float(cfg.lateral_drift_decay),
        )
        meas_valid, meas = self._extract_measurement(tag_detection, T_base_cam, cfg)
        return self._update_from_measurement(meas_valid, meas, dt, cfg)

    # Backward-compatible legacy entry point used by current agent wiring.
    def step(
        self, pose_estimate: dict[str, Any] | None, dt: float = 0.05
    ) -> tuple[float, float, bool, dict[str, Any]]:
        cfg = self._normalize_cfg(self.cfg)
        if pose_estimate is None:
            v, w, dbg = self._update_from_measurement(False, None, dt, cfg)
            return v, w, bool(dbg["success"]), dbg
        rel = np.asarray(
            pose_estimate.get("rel_target_pos_rover_m", [np.nan, np.nan, 0.0]),
            dtype=np.float64,
        )
        e_x = float(rel[0] - cfg.x_des)
        e_y = float(rel[1] - cfg.y_des)
        psi = float(
            pose_estimate.get(
                "axis_yaw_err_rad", pose_estimate.get("rel_target_rpy_rad", [0.0, 0.0, 0.0])[2]
            )
        )
        e_psi = wrap_angle(psi - cfg.psi_des)
        alpha = float(np.arctan2(e_y, max(e_x, cfg.eps)))
        rho = float(np.hypot(e_x, e_y))
        meas = {"e_x": e_x, "e_y": e_y, "e_psi": e_psi, "alpha": alpha, "rho": rho}
        v, w, dbg = self._update_from_measurement(True, meas, dt, cfg)
        return v, w, bool(dbg["success"]), dbg

    def _normalize_cfg(self, cfg: FiducialDockingConfig) -> FiducialDockingConfig:
        x_tol = cfg.x_tol if cfg.x_tol != FiducialDockingConfig.x_tol else cfg.antenna_xy_tol_m
        y_tol = cfg.y_tol if cfg.y_tol != FiducialDockingConfig.y_tol else cfg.antenna_xy_tol_m
        psi_tol = (
            cfg.psi_tol if cfg.psi_tol != FiducialDockingConfig.psi_tol else cfg.antenna_rot_tol_rad
        )
        return replace(
            cfg,
            k_alpha=cfg.k_alpha if cfg.k_alpha != FiducialDockingConfig.k_alpha else cfg.k_w_rot,
            k_v=cfg.k_v if cfg.k_v != FiducialDockingConfig.k_v else cfg.k_v_x,
            k_psi=cfg.k_psi if cfg.k_psi != FiducialDockingConfig.k_psi else 0.5 * cfg.k_w_rot,
            v_max_far=min(cfg.v_max_far, cfg.max_v),
            w_max=min(cfg.w_max, cfg.max_w),
            w_search=cfg.search_w,
            x_tol=x_tol,
            y_tol=y_tol,
            psi_tol=psi_tol,
        )

    def _extract_measurement(
        self,
        tag_detection: dict[str, Any] | None,
        T_base_cam: np.ndarray,
        cfg: FiducialDockingConfig,
    ) -> tuple[bool, dict[str, float] | None]:
        if tag_detection is None:
            return False, None
        if "T_cam_tag" in tag_detection:
            T_cam_tag = np.asarray(tag_detection["T_cam_tag"], dtype=np.float64)
        else:
            R_cam_tag = np.asarray(tag_detection.get("R_cam_tag"), dtype=np.float64)
            t_cam_tag = np.asarray(tag_detection.get("t_cam_tag"), dtype=np.float64).reshape(3)
            if R_cam_tag.shape != (3, 3):
                return False, None
            T_cam_tag = make_T(R_cam_tag, t_cam_tag)

        T_base_tag = np.asarray(T_base_cam, dtype=np.float64) @ T_cam_tag
        x = float(T_base_tag[0, 3])
        y = float(T_base_tag[1, 3])
        psi = yaw_from_R(T_base_tag[:3, :3])
        e_x = x - cfg.x_des
        e_y = y - cfg.y_des
        e_psi = wrap_angle(psi - cfg.psi_des)
        # Use full atan2 so behind-target cases (e_x < 0) are represented correctly.
        alpha = float(np.arctan2(e_y, e_x))
        rho = float(np.hypot(e_x, e_y))
        return True, {"e_x": e_x, "e_y": e_y, "e_psi": e_psi, "alpha": alpha, "rho": rho}

    def _ema(self, prev: float, meas: float, beta: float) -> float:
        return (1.0 - beta) * prev + beta * meas

    def _rate_limit(self, cmd: float, prev: float, max_rate: float, dt: float) -> float:
        max_step = max_rate * max(dt, 1e-6)
        return clamp(cmd, prev - max_step, prev + max_step)

    def _update_from_measurement(
        self,
        meas_valid: bool,
        meas: dict[str, float] | None,
        dt: float,
        cfg: FiducialDockingConfig,
    ) -> tuple[float, float, dict[str, Any]]:
        dt = max(float(dt), 1e-4)

        if meas_valid and meas is not None:
            self.time_since_seen = 0.0
            self.time_since_meas = 0.0
            self.last_meas = meas
            self.detect_count += 1
        else:
            self.time_since_seen += dt
            self.time_since_meas += dt
            self.detect_count = 0

        use_meas = None
        if meas_valid and meas is not None:
            use_meas = meas
        elif self.last_meas is not None and self.time_since_meas <= cfg.hold_last_meas_time:
            use_meas = self.last_meas

        if self.time_since_seen > cfg.dropout_time:
            self.stage = "search"

        if self.stage == "search" and self.detect_count >= cfg.n_detect and use_meas is not None:
            self.stage = "rotate"
            self.gate_count = 0

        if use_meas is not None:
            e_x = float(use_meas["e_x"])
            e_y = float(use_meas["e_y"])
            e_psi = float(use_meas["e_psi"])
            alpha = float(use_meas["alpha"])
            rho = float(use_meas["rho"])
            if not self.has_filter:
                self.ex_f, self.ey_f, self.epsi_f, self.alpha_f, self.rho_f = (
                    e_x,
                    e_y,
                    e_psi,
                    alpha,
                    rho,
                )
                self.has_filter = True
            else:
                b = cfg.ema_beta
                self.ex_f = self._ema(self.ex_f, e_x, b)
                self.ey_f = self._ema(self.ey_f, e_y, b)
                self.epsi_f = wrap_angle(self._ema(self.epsi_f, e_psi, b))
                self.alpha_f = wrap_angle(self._ema(self.alpha_f, alpha, b))
                self.rho_f = self._ema(self.rho_f, rho, b)
        else:
            e_x = self.ex_f
            e_y = self.ey_f
            e_psi = self.epsi_f
            alpha = self.alpha_f
            rho = self.rho_f

        # Reverse-aware heading error:
        # if target is behind and reverse is allowed, align using the backward axis.
        ey_ctrl = self._drift_comp.compensate(self.ey_f)
        alpha_ctrl = float(np.arctan2(ey_ctrl, self.ex_f))
        if cfg.allow_reverse and self.ex_f < 0.0:
            # When target is behind, steer using backward-axis bearing.
            alpha_ctrl = float(np.arctan2(ey_ctrl, -self.ex_f))

        if self.stage == "search":
            v_des = 0.0
            w_des = cfg.yaw_cmd_sign * cfg.w_search
        elif self.stage == "rotate":
            v_des = 0.0
            w_des = clamp(cfg.yaw_cmd_sign * cfg.k_alpha * alpha_ctrl, -cfg.w_max, cfg.w_max)
            if abs(alpha_ctrl) < cfg.alpha_gate:
                self.gate_count += 1
            else:
                self.gate_count = 0
            # Optional close-range override; disabled by default for strict centering first.
            if (
                cfg.early_approach_rho_override_m > 0.0
                and self.rho_f < cfg.early_approach_rho_override_m
            ):
                self.gate_count = max(self.gate_count, cfg.n_gate)
            if self.gate_count >= cfg.n_gate:
                self.stage = "approach"
                self.gate_count = 0
        elif self.stage == "approach":
            heading_gate = clamp(np.cos(alpha_ctrl), 0.0, 1.0)
            v_cap = min(cfg.v_max_far, cfg.k_slow * max(self.rho_f, 0.0))
            v_des = clamp(cfg.k_v * self.ex_f * heading_gate, -v_cap, v_cap)
            if abs(self.ey_f) > cfg.lateral_engage_m and abs(v_des) < cfg.approach_min_speed:
                # When close in range but laterally offset, forward crawl tends to reduce
                # persistent side drift better than reversing on sloped terrain.
                if cfg.lateral_crawl_forward_bias and abs(self.ex_f) < cfg.reverse_deadband_m:
                    crawl_sign = 1.0
                else:
                    crawl_sign = -1.0 if self.ex_f <= 0.0 else 1.0
                v_des = clamp(crawl_sign * cfg.approach_min_speed, -v_cap, v_cap)
            yaw_ramp = smoothstep01(1.0 - self.rho_f / max(cfg.rho_close, cfg.eps))
            epsi_term = (
                (cfg.k_psi * self.epsi_f * yaw_ramp) if cfg.use_yaw_term_in_approach else 0.0
            )
            w_des = clamp(
                cfg.yaw_cmd_sign * (cfg.k_alpha * alpha_ctrl + epsi_term),
                -min(cfg.w_max, cfg.max_w_approach),
                min(cfg.w_max, cfg.max_w_approach),
            )
            if self.rho_f < cfg.rho_final and abs(self.ey_f) < cfg.y_final:
                self.stage = "final_yaw" if cfg.use_final_yaw else "final_hold"
        elif self.stage == "final_yaw":
            v_des = clamp(cfg.k_v_final * self.ex_f, -cfg.v_final_max, cfg.v_final_max)
            w_des = clamp(
                cfg.yaw_cmd_sign * cfg.k_psi_final * self.epsi_f,
                -cfg.w_final_max,
                cfg.w_final_max,
            )
        elif self.stage == "final_hold":
            # Final translation/lateral tightening when final yaw stage is disabled.
            heading_gate = clamp(np.cos(alpha_ctrl), 0.0, 1.0)
            v_des = clamp(
                cfg.k_v_final * self.ex_f * heading_gate, -cfg.v_final_max, cfg.v_final_max
            )
            w_des = clamp(
                cfg.yaw_cmd_sign * cfg.k_alpha * alpha_ctrl,
                -cfg.w_final_max,
                cfg.w_final_max,
            )
        else:
            v_des, w_des = 0.0, 0.0

        if not cfg.allow_reverse:
            v_des = max(v_des, 0.0)

        self._drift_comp.update(
            stage=self.stage,
            ex_m=self.ex_f,
            ey_m=self.ey_f,
            v_ref_mps=v_des,
            meas_valid=(use_meas is not None),
        )

        v_cmd = self._rate_limit(v_des, self.v_prev, cfg.dv_max, dt)
        w_cmd = self._rate_limit(w_des, self.w_prev, cfg.dw_max, dt)
        self.v_prev, self.w_prev = v_cmd, w_cmd

        if cfg.latch_success:
            success_now = (
                abs(self.ex_f) < cfg.x_tol
                and abs(self.ey_f) < cfg.y_tol
                and (abs(self.epsi_f) < cfg.psi_tol if cfg.use_final_yaw else True)
            )
            if success_now:
                self.success_hold_s += dt
            else:
                self.success_hold_s = 0.0
            if self.success_hold_s >= cfg.hold_time_s:
                self.success_latched = True
                self.stage = "done"
                v_cmd, w_cmd = 0.0, 0.0
        else:
            self.success_hold_s = 0.0
            self.success_latched = False

        dbg = {
            "stage": self.stage,
            "target_found": bool(meas_valid),
            "e_x": float(self.ex_f),
            "e_y": float(self.ey_f),
            "e_psi": float(self.epsi_f),
            "alpha": float(self.alpha_f),
            "alpha_ctrl": float(alpha_ctrl),
            "ey_ctrl": float(ey_ctrl),
            "lateral_bias_y_m": float(self._drift_comp.bias_y_m),
            "rho": float(self.rho_f),
            "e_x_raw": float(e_x),
            "e_y_raw": float(e_y),
            "e_psi_raw": float(e_psi),
            "alpha_raw": float(alpha),
            "rho_raw": float(rho),
            "v_des": float(v_des),
            "w_des": float(w_des),
            "v_cmd": float(v_cmd),
            "w_cmd": float(w_cmd),
            "tag_age_s": float(self.time_since_seen),
            "hold_success_s": float(self.success_hold_s),
            "success": bool(self.success_latched),
        }
        return v_cmd, w_cmd, dbg


def _simulate_demo() -> None:
    np.random.seed(0)
    cfg = FiducialDockingConfig(
        use_final_yaw=False,
        x_des=1.0,
        y_des=0.0,
        x_tol=0.08,
        y_tol=0.08,
        hold_time_s=0.30,
    )
    ctrl = FiducialDockingController(cfg)

    dt = 0.05
    T_base_cam = np.eye(4, dtype=np.float64)
    T_base_cam[:3, 3] = np.array([0.0, 0.0, 0.2], dtype=np.float64)

    x_r, y_r, yaw_r = 0.0, -1.0, np.deg2rad(20.0)
    x_tag, y_tag, yaw_tag = 3.0, 0.0, np.pi

    def T_world_pose(x: float, y: float, yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.array([x, y, 0.0], dtype=np.float64)
        return T

    T_w_tag = T_world_pose(x_tag, y_tag, yaw_tag)

    for k in range(600):
        T_w_b = T_world_pose(x_r, y_r, yaw_r)
        T_b_tag = inv_T(T_w_b) @ T_w_tag
        T_c_tag = inv_T(T_base_cam) @ T_b_tag

        dropout = np.random.rand() < 0.08
        if dropout:
            det = None
        else:
            t = T_c_tag[:3, 3].copy()
            t[:2] += 0.02 * np.random.randn(2)
            yaw_n = yaw_from_R(T_c_tag[:3, :3]) + np.deg2rad(2.0) * np.random.randn()
            c, s = np.cos(yaw_n), np.sin(yaw_n)
            Rn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            det = {"R_cam_tag": Rn, "t_cam_tag": t, "confidence": 1.0}

        v, w, dbg = ctrl.update(det, T_base_cam, dt)
        x_r += v * np.cos(yaw_r) * dt
        y_r += v * np.sin(yaw_r) * dt
        yaw_r = wrap_angle(yaw_r + w * dt)

        if k % 20 == 0 or dbg["success"]:
            print(
                f"k={k:03d} stage={dbg['stage']:<10} found={dbg['target_found']} "
                f"ex={dbg['e_x']:+.3f} ey={dbg['e_y']:+.3f} alpha={dbg['alpha']:+.3f} "
                f"epsi={dbg['e_psi']:+.3f} rho={dbg['rho']:.3f} "
                f"v={v:+.3f} w={w:+.3f} tag_age={dbg['tag_age_s']:.2f} "
                f"success={dbg['success']}"
            )
        if dbg["success"]:
            print("Demo converged.")
            break


if __name__ == "__main__":
    _simulate_demo()
