"""Mode-configurable docking success checks (outside controller)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lac.control.docking_profiles import DockingModeProfile


@dataclass
class DockingSuccessChecker:
    profile: DockingModeProfile
    hold_frames: int = 8
    start_power_wh: float | None = None

    def __post_init__(self):
        self._good_count = 0

    def reset(self, start_power_wh: float | None = None) -> None:
        self._good_count = 0
        self.start_power_wh = start_power_wh

    def check(
        self, pose_estimate: dict[str, Any] | None, current_power_wh: float | None = None
    ) -> tuple[bool, dict[str, Any]]:
        if pose_estimate is None:
            self._good_count = 0
            return False, {"reason": "no_pose_estimate", "good_count": self._good_count}

        rel = np.asarray(pose_estimate["rel_target_pos_rover_m"], dtype=np.float64)
        rot_err = float(pose_estimate.get("rot_abs_max_rad", np.inf))
        pos_ok = (
            abs(float(rel[0])) <= self.profile.pos_tol_xy_m
            and abs(float(rel[1])) <= self.profile.pos_tol_xy_m
        )
        rot_ok = rot_err <= self.profile.rot_tol_rad
        if pos_ok and rot_ok:
            self._good_count += 1
        else:
            self._good_count = 0

        geometric_ok = self._good_count >= self.hold_frames
        if not geometric_ok:
            return False, {
                "reason": "geometric_not_stable",
                "good_count": self._good_count,
                "pos_ok": pos_ok,
                "rot_ok": rot_ok,
                "rot_err": rot_err,
            }

        if not self.profile.require_power_recovery:
            return True, {
                "reason": "virtual_docking_geometric_success",
                "good_count": self._good_count,
                "rot_err": rot_err,
            }

        if self.start_power_wh is None or current_power_wh is None:
            return False, {
                "reason": "missing_power_for_charging_check",
                "good_count": self._good_count,
            }
        power_delta = float(current_power_wh - self.start_power_wh)
        if power_delta >= self.profile.power_delta_wh:
            return True, {
                "reason": "charging_detected",
                "good_count": self._good_count,
                "power_delta_wh": power_delta,
            }
        return False, {
            "reason": "waiting_for_charging",
            "good_count": self._good_count,
            "power_delta_wh": power_delta,
        }
