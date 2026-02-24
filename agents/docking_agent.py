#!/usr/bin/env python

"""Two-phase docking agent: path tracking + fiducial visual servoing."""

from __future__ import annotations

import json
import signal
from typing import Any

import carla
import numpy as np
import rerun as rr
from rich import print

import lac.params as params
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from lac.control.controller import TrajectoryTracker, TrajectoryTrackerConfig
from lac.control.fiducial_docking_controller import FiducialDockingConfig, FiducialDockingController
from lac.planning.dem_grid import DEMGrid
from lac.planning.dem_planner import DEMPlannerParams, plan_path_dem, yaw_from_pose
from lac.planning.heading_aware_planner import HeadingAwarePlanner, HeadingAwarePlannerParams
from lac.planning.path_smoother import Path2D, PathSmoother, PathSmootherConfig, Trajectory2D
from lac.utils.rerun_interface import Rerun, rerun_dem, rerun_dem_grid_lines, rerun_lander
from lac.utils.visualization import overlay_tag_detections
from lac.util import transform_to_numpy


GOAL_XY_M = (0.0, 2.0)
GOAL_YAW_RAD = 0.0
DEM_MAP_PATH = "/home/shared/data_raw/Lunar/LAC/maps/competition/Moon_Map_01_preset_0.dat"

ARM_RAISE_WAIT_FRAMES = 80
USE_HEADING_AWARE_PLANNER = True
DEM_SWAP_XY = True

DOCKING_SWITCH_DIST_M = 0.2
CHARGE_SUCCESS_DELTA_WH = 2.0
RERUN_ENABLED = True


def get_entry_point():
    return "DockingAgent"


class DockingAgent(AutonomousAgent):
    def setup(self, path_to_conf_file: str):
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
        self.path_vis_z_offset_m = 0.5 * params.WHEEL_DIAMETER + params.WHEEL_CENTER_Z_OFFSET

        self.phase = "nav"
        self.dock_start_power = None
        self.last_dock_cmd = (0.0, 0.0)

        self.initial_pose = transform_to_numpy(self.get_initial_position())
        self.current_pose = self.initial_pose.copy()
        self.start_xy_m = (float(self.initial_pose[0, 3]), float(self.initial_pose[1, 3]))

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

        self.visual_servo = FiducialDockingController(FiducialDockingConfig())

        self.cameras = params.CAMERA_CONFIG_INIT
        for cam in self.cameras:
            self.cameras[cam] = self.cameras[cam].copy()
            self.cameras[cam]["active"] = False
            self.cameras[cam]["semantic"] = False
        self.cameras["FrontLeft"]["active"] = True
        self.cameras["FrontLeft"]["light"] = 1.0
        self.cameras["FrontLeft"]["width"] = 1280
        self.cameras["FrontLeft"]["height"] = 720
        self.cameras["Right"]["active"] = True
        self.cameras["Right"]["light"] = 1.0
        self.cameras["Right"]["width"] = 1280
        self.cameras["Right"]["height"] = 720

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
                planned_xyz = self.dem.lift_xy_to_xyz(
                    np.asarray(self.path_world), z_offset_m=self.path_vis_z_offset_m
                )
                traj_xyz = self.dem.lift_xy_to_xyz(
                    self.trajectory.xyt[:, :2], z_offset_m=self.path_vis_z_offset_m
                )
                Rerun.log_3d_trajectory(
                    0, planned_xyz, trajectory_string="planned_path", color=[255, 0, 0]
                )
                Rerun.log_3d_trajectory(
                    0, traj_xyz, trajectory_string="smoothed_trajectory", color=[0, 255, 255]
                )
                self.rerun_rover_positions = [self.current_pose[:3, 3].copy()]
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

        print(
            f"[green]Docking phase-1 path planned: {len(self.path_world)} waypoints, "
            f"{len(self.trajectory.t)} trajectory samples."
        )
        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signal_received, frame):
        print("\nCtrl+C detected! Exiting mission")
        self.mission_complete()

    def initialize(self):
        self.set_front_arm_angle(params.FRONT_ARM_ANGLE_STATIC_RAD)
        self.set_back_arm_angle(params.ARM_ANGLE_STATIC_RAD)

    def use_fiducials(self):
        return True

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

    def _switch_to_docking_phase(self):
        if self.phase == "dock":
            return
        self.phase = "dock"
        self.visual_servo.reset()
        self.dock_start_power = float(self.get_current_power())
        self.set_radiator_cover_state(carla.RadiatorCoverState.Open)
        print("[yellow]Switching to docking visual servo phase.")

    def run_step(self, input_data):
        if self.step == 0:
            self.initialize()
        self.step += 1
        self.current_pose = transform_to_numpy(self.get_transform())
        if self.rerun_enabled:
            self.rerun_rover_positions.append(self.current_pose[:3, 3].copy())
            rr.set_time_sequence("frame_id", self.step)
            Rerun.log_3d_trajectory(
                self.step,
                np.asarray(self.rerun_rover_positions),
                trajectory_string="rover_trajectory_live",
                color=[0, 255, 0],
            )
            grayscale_data = input_data.get("Grayscale", {})
            front_left_img = grayscale_data.get(carla.SensorPosition.FrontLeft)
            if front_left_img is not None:
                fl_detections = self.visual_servo.detector.detect(front_left_img)
                front_left_overlay = overlay_tag_detections(front_left_img, fl_detections)
                Rerun.log_img_seq(
                    "/world/cameras/front_left", self.step, front_left_overlay, adjust_rgb=True
                )
            right_img_for_vis = grayscale_data.get(carla.SensorPosition.Right)
            if right_img_for_vis is not None:
                right_detections = self.visual_servo.detector.detect(right_img_for_vis)
                right_overlay = overlay_tag_detections(right_img_for_vis, right_detections)
                Rerun.log_img_seq("/world/cameras/right", self.step, right_overlay, adjust_rgb=True)

        if self.phase == "nav":
            if self.step < ARM_RAISE_WAIT_FRAMES:
                return carla.VehicleVelocityControl(0.0, 0.0)

            v_cmd, w_cmd = self.tracker.compute_command(
                reference=self.trajectory,
                current_pose=self.current_pose,
                dt=params.DT,
            )
            self.current_v, self.current_w = v_cmd, w_cmd

            dist_to_goal = float(self.tracker.last_debug.get("dist_to_goal_m", np.inf))
            print(f"[blue]approach phase: dist to goal: {dist_to_goal:.2f} m")
            if (
                self.tracker.last_debug.get("reached_goal", False)
                or dist_to_goal <= DOCKING_SWITCH_DIST_M
            ):
                self._switch_to_docking_phase()
                return carla.VehicleVelocityControl(0.0, 0.0)

            return carla.VehicleVelocityControl(self.current_v, self.current_w)

        right_img = None
        if "Grayscale" in input_data:
            right_img = input_data["Grayscale"].get(carla.SensorPosition.Right)
        v_cmd, w_cmd, aligned, dbg = self.visual_servo.step(right_img)
        self.last_dock_cmd = (v_cmd, w_cmd)

        if aligned:
            power_now = float(self.get_current_power())
            if (
                self.dock_start_power is not None
                and power_now >= self.dock_start_power + CHARGE_SUCCESS_DELTA_WH
            ):
                print("[bold green]Charging detected. Mission complete.")
                self.mission_complete()
                return carla.VehicleVelocityControl(0.0, 0.0)
            print(
                "[cyan]Docking alignment achieved; waiting for charging "
                f"(power delta={power_now - (self.dock_start_power or power_now):.2f} Wh)"
            )

        print(
            f"[blue]Dock phase step={self.step} | found={dbg.get('target_found', False)} | "
            f"x_err={dbg.get('x_err_norm', np.nan):.3f} | area={dbg.get('area_ratio', np.nan):.4f} | "
            f"aligned={dbg.get('aligned', False)} | v={v_cmd:.3f} w={w_cmd:.3f}"
        )
        return carla.VehicleVelocityControl(v_cmd, w_cmd)

    def finalize(self):
        print("[blue]DockingAgent finalize")
