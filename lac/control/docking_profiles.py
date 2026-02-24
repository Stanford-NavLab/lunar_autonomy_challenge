"""Docking mode profiles and geometry-derived configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from lac.params import GEOMETRY_DICT


DockingMode = Literal["side", "front_virtual"]


@dataclass
class DockingModeProfile:
    mode: DockingMode
    camera_name: str
    target_tag_id: int
    lander_target_point_lander: np.ndarray
    lander_target_axis_lander: np.ndarray
    rover_target_point_rover: np.ndarray
    rover_target_axis_rover: np.ndarray
    pos_tol_xy_m: float
    rot_tol_rad: float
    require_power_recovery: bool
    power_delta_wh: float


def _np3(vals) -> np.ndarray:
    return np.asarray(vals, dtype=np.float64).reshape(3)


def make_docking_mode_profile(mode: DockingMode) -> DockingModeProfile:
    lander_locator = GEOMETRY_DICT["lander"]["locator"]
    lander_antenna = GEOMETRY_DICT["lander"]["antenna"]
    rover_antenna = GEOMETRY_DICT["rover"]["antenna"]

    if mode == "side":
        return DockingModeProfile(
            mode="side",
            camera_name="Right",
            target_tag_id=int(lander_locator["id"]),
            lander_target_point_lander=_np3(
                [lander_antenna["x"], lander_antenna["y"], lander_antenna["z"]]
            ),
            lander_target_axis_lander=_np3(lander_antenna["orientation"]),
            rover_target_point_rover=_np3(
                [rover_antenna["x"], rover_antenna["y"], rover_antenna["z"]]
            ),
            rover_target_axis_rover=_np3(rover_antenna["orientation"]),
            pos_tol_xy_m=0.10,
            rot_tol_rad=0.52,
            require_power_recovery=True,
            power_delta_wh=2.0,
        )

    # Front virtual docking: use same fiducial anchor but front camera and virtual success (no power check).
    return DockingModeProfile(
        mode="front_virtual",
        camera_name="FrontLeft",
        target_tag_id=int(lander_locator["id"]),
        lander_target_point_lander=_np3(
            [lander_locator["x"], lander_locator["y"], lander_locator["z"]]
        ),
        # Virtual target axis points roughly away from fiducial/antenna direction for head-on alignment.
        lander_target_axis_lander=_np3(lander_antenna["orientation"]),
        rover_target_point_rover=_np3([0.0, 0.0, 0.0]),
        rover_target_axis_rover=_np3([1.0, 0.0, 0.0]),
        pos_tol_xy_m=0.20,
        rot_tol_rad=0.35,
        require_power_recovery=False,
        power_delta_wh=0.0,
    )
