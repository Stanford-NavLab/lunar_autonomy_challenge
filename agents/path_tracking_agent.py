#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Path tracking agent using DEM planning + trajectory smoothing."""

from __future__ import annotations

import json
import os
import signal
from typing import Any
from datetime import datetime

import carla
import numpy as np
import rerun as rr
from rich import print

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.control.controller import TrajectoryTracker, TrajectoryTrackerConfig
from lac.planning.dem_grid import DEMGrid
from lac.planning.dem_planner import DEMPlannerParams, plan_path_dem, yaw_from_pose
from lac.planning.heading_aware_planner import HeadingAwarePlanner, HeadingAwarePlannerParams
from lac.planning.path_smoother import Path2D, PathSmoother, PathSmootherConfig, Trajectory2D
from lac.utils.data_logger import DataLogger
from lac.utils.rerun_interface import Rerun, rerun_dem, rerun_dem_grid_lines, rerun_lander
from lac.util import transform_to_numpy
import lac.params as params


# ----------------------------- User constants ----------------------------- #
# Start is taken from the agent's actual initial pose at runtime.
GOAL_XY_M = (0.0, 2.0)
GOAL_YAW_RAD = 0.0

# Ground-truth DEM map path used by the planner.
DEM_MAP_PATH = "/home/shared/data_raw/Lunar/LAC/maps/competition/Moon_Map_01_preset_0.dat"

# Wait for arms to move before driving.
ARM_RAISE_WAIT_FRAMES = 80
RERUN_ENABLED = True
LOG_DATA = True
USE_HEADING_AWARE_PLANNER = True
DEM_SWAP_XY = True


def get_entry_point():
    return "PathTrackingAgent"


class PathTrackingAgent(AutonomousAgent):
    """Minimal navigation agent: plan once, then track the generated trajectory."""

    def setup(self, path_to_conf_file: str):
        # Optional config (used only for lightweight overrides if present).
        self.config: dict[str, Any] = {}
        if path_to_conf_file:
            try:
                self.config = json.load(open(path_to_conf_file))
            except Exception:
                self.config = {}

        self.step = 0
        self.current_v = 0.0
        self.current_w = 0.0
        self.rerun_enabled = RERUN_ENABLED
        self.rerun_rover_positions: list[np.ndarray] = []
        self.rover_trajectory_world: list[np.ndarray] = []
        self.path_vis_z_offset_m = 0.5 * params.WHEEL_DIAMETER + params.WHEEL_CENTER_Z_OFFSET

        self.initial_pose = transform_to_numpy(self.get_initial_position())
        self.current_pose = self.initial_pose.copy()
        self.start_xy_m = (float(self.initial_pose[0, 3]), float(self.initial_pose[1, 3]))

        self.cameras = params.CAMERA_CONFIG_INIT
        for cam in self.config["cameras"]:
            cam_config = self.config["cameras"][cam].copy()
            # Convert string "True"/"False" to boolean
            if isinstance(cam_config.get("active"), str):
                cam_config["active"] = cam_config["active"].lower() == "true"
            if isinstance(cam_config.get("semantic"), str):
                cam_config["semantic"] = cam_config["semantic"].lower() == "true"
            self.cameras[cam] = cam_config
        self.active_cameras = [cam for cam, config in self.cameras.items() if config["active"]]

        map_arr = np.load(DEM_MAP_PATH, allow_pickle=True)
        self.dem = DEMGrid.from_map_array(map_arr, cell_size=params.CELL_WIDTH, swap_xy=DEM_SWAP_XY)
        self.dem_z = self.dem.z
        self.dem_x_grid, self.dem_y_grid = self.dem.mesh_grids()

        planner_cfg = DEMPlannerParams(
            roughness_window=5,
            w_s=5.0,
            w_r=2.0,
            w_d=10.0,
            s_max=0.6,
            r_max=0.1,
            d_max=0.15,
            hard_slope_max=0.9,
            hard_step_max=0.22,
            use_lander_keepout=True,
            lander_buffer_m=0.1,
            lander_center_xy_m=(0.0, 0.0),
            do_spline=True,
        )

        smoother_cfg = PathSmootherConfig(
            cell_size=params.CELL_WIDTH,
            ds=0.10,
            v_nominal=params.TARGET_SPEED,
            v_max=0.6,
            max_omega=params.MAX_STEER,
        )

        if USE_HEADING_AWARE_PLANNER:
            heading_planner = HeadingAwarePlanner(
                HeadingAwarePlannerParams(
                    dem_params=planner_cfg,
                    smoother_config=smoother_cfg,
                )
            )
            heading_result = heading_planner.plan(
                self.dem_z,
                start_xy=self.start_xy_m,
                start_yaw_rad=yaw_from_pose(self.initial_pose),
                goal_xy=GOAL_XY_M,
                goal_yaw_rad=GOAL_YAW_RAD,
                cell_size=params.CELL_WIDTH,
            )
            self.path_world = heading_result.path_xy
            self.path_cost = heading_result.total_cost
            self.path_debug = heading_result.debug
            self.trajectory = heading_result.trajectory
        else:
            self.path_world, self.path_cost, self.path_debug = plan_path_dem(
                self.dem_z,
                cell_size=params.CELL_WIDTH,
                start_xy=self.start_xy_m,
                goal_xy=GOAL_XY_M,
                params=planner_cfg,
                use_theta_star=True,
                do_smooth=True,
                input_is_grid=False,
                initial_pose=self.initial_pose,
            )
            if len(self.path_world) == 0:
                raise RuntimeError("DEM planner failed to generate a path.")
            smoother = PathSmoother(smoother_cfg)
            self.trajectory: Trajectory2D = smoother.smooth(
                Path2D(xy=np.asarray(self.path_world), meta=self.path_debug),
                self.dem_z,
                initial_pose=self.initial_pose,
                goal_yaw_rad=GOAL_YAW_RAD,
            )

        tracker_cfg = TrajectoryTrackerConfig(
            target_speed=params.TARGET_SPEED,
            max_v=0.6,
            max_w=0.5,
            max_w_accel=0.4,
            lookahead_time=0.6,
            min_lookahead=0.20,
            max_lookahead=0.80,
            k_heading=0.8,
            k_cross_track=0.3,
            k_goal_v=0.8,
            default_dt=params.DT,
        )
        self.tracker = TrajectoryTracker(tracker_cfg)

        print(
            f"[green]Planned path from start={self.start_xy_m} to goal={GOAL_XY_M} "
            f"with {len(self.path_world)} waypoints, "
            f"{len(self.trajectory.t)} trajectory samples."
        )
        print(
            f"[green]Start/goal grid: {self.path_debug['start_grid']} -> {self.path_debug['goal_grid']}; "
            f"cost={self.path_cost:.3f}"
        )
        print(f"[cyan]Planned world path points:\n{np.asarray(self.path_world)}")
        print(f"[cyan]Tracker config: {tracker_cfg}")
        print(f"[cyan]Path visualization z-offset: {self.path_vis_z_offset_m:.3f} m")

        if LOG_DATA:
            self.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            preset = os.environ.get("MISSIONS_SUBSET")
            seed = os.environ.get("SEED")
            self.data_logger = DataLogger(
                self,
                get_entry_point(),
                self.run_name,
                preset,
                seed,
                self.cameras,
            )
            self.data_logger.data["planned_path_world_xy"] = np.asarray(self.path_world).tolist()
            self.data_logger.data["smoothed_trajectory_xyt"] = np.asarray(
                self.trajectory.xyt
            ).tolist()
            self.data_logger.data["smoothed_trajectory_t"] = np.asarray(self.trajectory.t).tolist()
            self.data_logger.data["smoothed_trajectory_v"] = np.asarray(self.trajectory.v).tolist()
            self.data_logger.data["smoothed_trajectory_w"] = np.asarray(self.trajectory.w).tolist()
            self.data_logger.data["path_planning"] = {
                "start_world_xy": list(self.start_xy_m),
                "goal_world_xy": list(GOAL_XY_M),
                "goal_yaw_rad": float(GOAL_YAW_RAD),
                "heading_aware_planner": bool(USE_HEADING_AWARE_PLANNER),
                "start_grid": list(self.path_debug.get("start_grid", (-1, -1))),
                "goal_grid": list(self.path_debug.get("goal_grid", (-1, -1))),
                "total_cost": float(self.path_cost),
            }

        if self.rerun_enabled:
            try:
                Rerun.init3d(img_compress=True)
                dem_mesh = rerun_dem(
                    self.dem_z,
                    x_grid=self.dem_x_grid,
                    y_grid=self.dem_y_grid,
                    alpha=60,
                )
                Rerun.log_3d_mesh(dem_mesh, topic="/world/dem_mesh", static=True)
                lander_mesh = rerun_lander()
                Rerun.log_3d_mesh(lander_mesh, topic="/world/lander_mesh", static=True)
                dem_lines = rerun_dem_grid_lines(
                    self.dem_z,
                    x_grid=self.dem_x_grid,
                    y_grid=self.dem_y_grid,
                    stride=10,
                )
                Rerun.log_3d_line_strips(
                    dem_lines,
                    topic="/world/dem_mesh_grid",
                    color=[170, 170, 170],
                    radius=0.003,
                    static=True,
                )

                planned_xyz = self._xy_to_xyz(
                    np.asarray(self.path_world), z_offset_m=self.path_vis_z_offset_m
                )
                traj_xyz = self._xy_to_xyz(
                    self.trajectory.xyt[:, :2], z_offset_m=self.path_vis_z_offset_m
                )
                Rerun.log_3d_trajectory(
                    0, planned_xyz, trajectory_string="planned_path", color=[255, 0, 0]
                )
                Rerun.log_3d_trajectory(
                    0, traj_xyz, trajectory_string="smoothed_trajectory", color=[0, 255, 255]
                )

                self.rerun_rover_positions = [self.current_pose[:3, 3].copy()]
                self.rover_trajectory_world = [self.current_pose[:3, 3].copy()]
                rr.set_time_sequence("frame_id", 0)
                Rerun.log_3d_trajectory(
                    0,
                    np.asarray(self.rerun_rover_positions),
                    trajectory_string="rover_trajectory_live",
                    color=[0, 255, 0],
                )
            except Exception as exc:
                self.rerun_enabled = False
                print(f"[yellow]Rerun initialization skipped: {exc}")

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def _xy_to_xyz(self, xy: np.ndarray, z_offset_m: float = 0.0) -> np.ndarray:
        """Lift world xy path to xyz using DEM height lookups."""
        return self.dem.lift_xy_to_xyz(xy, z_offset_m=z_offset_m)

    def handle_interrupt(self, signal_received, frame):
        print("\nCtrl+C detected! Exiting mission")
        self.mission_complete()

    def initialize(self):
        self.set_front_arm_angle(params.FRONT_ARM_ANGLE_STATIC_RAD)
        self.set_back_arm_angle(params.ARM_ANGLE_STATIC_RAD)

    def use_fiducials(self):
        return False

    def sensors(self):
        sensors = {}
        for cam, config in self.cameras.items():
            sensors[getattr(carla.SensorPosition, cam)] = {
                "camera_active": config["active"],
                "light_intensity": config["light"],
                "width": config["width"],
                "height": config["height"],
                "use_semantic": config["semantic"],
            }
        return sensors

    def run_step(self, input_data):
        if self.step == 0:
            self.initialize()
        self.step += 1

        self.current_pose = transform_to_numpy(self.get_transform())
        self.rover_trajectory_world.append(self.current_pose[:3, 3].copy())
        if self.rerun_enabled:
            self.rerun_rover_positions.append(self.current_pose[:3, 3].copy())
            rr.set_time_sequence("frame_id", self.step)
            Rerun.log_3d_trajectory(
                self.step,
                np.asarray(self.rerun_rover_positions),
                trajectory_string="rover_trajectory_live",
                color=[0, 255, 0],
            )

        if self.step < ARM_RAISE_WAIT_FRAMES:
            control = carla.VehicleVelocityControl(0.0, 0.0)
            if LOG_DATA:
                self.data_logger.log_data(self.step, control, est_pose=self.current_pose)
                self.data_logger.data["rover_trajectory_world_xyz"] = np.asarray(
                    self.rover_trajectory_world
                ).tolist()
            return control

        v_cmd, w_cmd = self.tracker.compute_command(
            reference=self.trajectory,
            current_pose=self.current_pose,
            dt=params.DT,
        )
        self.current_v, self.current_w = v_cmd, w_cmd

        dbg = self.tracker.last_debug
        nearest_idx = int(dbg.get("nearest_idx", 0))
        target_idx = int(dbg.get("target_idx", 0))
        progress_pct = 100.0 * nearest_idx / max(len(self.trajectory.t) - 1, 1)
        dist_to_goal = float(dbg.get("dist_to_goal_m", 0.0))
        print(
            f"[blue]Tracking step={self.step} | progress={progress_pct:5.1f}% "
            f"(nearest {nearest_idx}, target {target_idx}, N={len(self.trajectory.t) - 1}) | "
            f"dist_to_goal={dist_to_goal:.2f} m | v={self.current_v:.3f} m/s | w={self.current_w:.3f} rad/s"
        )
        print(
            f"[magenta]  pose=({dbg.get('current_xy', np.array([np.nan, np.nan]))[0]:.2f}, "
            f"{dbg.get('current_xy', np.array([np.nan, np.nan]))[1]:.2f}, "
            f"{dbg.get('current_yaw_rad', np.nan):.2f}rad) | "
            f"target=({dbg.get('target_xy', np.array([np.nan, np.nan]))[0]:.2f}, "
            f"{dbg.get('target_xy', np.array([np.nan, np.nan]))[1]:.2f}, "
            f"{dbg.get('target_theta_rad', np.nan):.2f}rad)"
        )
        print(
            f"[magenta]  errors: heading={dbg.get('heading_err_rad', np.nan):.3f}, "
            f"path_heading={dbg.get('heading_path_err_rad', np.nan):.3f}, "
            f"cross_track={dbg.get('cross_track_err_m', np.nan):.3f} m, "
            f"goal_heading={dbg.get('heading_goal_err_rad', np.nan):.3f}"
        )
        print(
            f"[magenta]  cmds: v_ref={dbg.get('v_ref', np.nan):.3f}, "
            f"v_ref_eff={dbg.get('v_ref_eff', np.nan):.3f}, "
            f"v_prev={dbg.get('v_prev_cmd', np.nan):.3f}, v_des={dbg.get('v_des', np.nan):.3f}, "
            f"v_cmd={dbg.get('v_cmd', np.nan):.3f} | "
            f"w_ref={dbg.get('w_ref', np.nan):.3f}, w_prev={dbg.get('w_prev_cmd', np.nan):.3f}, "
            f"w_des={dbg.get('w_des', np.nan):.3f}, w_cmd={dbg.get('w_cmd', np.nan):.3f}"
        )

        if self.tracker.last_debug.get("reached_goal", False):
            print("[bold green]Reached trajectory goal. Mission complete.")
            control = carla.VehicleVelocityControl(0.0, 0.0)
            if LOG_DATA:
                self.data_logger.log_data(self.step, control, est_pose=self.current_pose)
                self.data_logger.data["rover_trajectory_world_xyz"] = np.asarray(
                    self.rover_trajectory_world
                ).tolist()
            self.mission_complete()
            return control

        control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        if LOG_DATA:
            waypoint = np.asarray(dbg.get("target_xy", [np.nan, np.nan]), dtype=np.float64)
            self.data_logger.log_data(
                self.step, control, est_pose=self.current_pose, waypoint=waypoint
            )
            self.data_logger.data["rover_trajectory_world_xyz"] = np.asarray(
                self.rover_trajectory_world
            ).tolist()
        return control

    def finalize(self):
        if LOG_DATA:
            self.data_logger.data["rover_trajectory_world_xyz"] = np.asarray(
                self.rover_trajectory_world
            ).tolist()
            self.data_logger.save_log()
        print("[blue]PathTrackingAgent finalize")
