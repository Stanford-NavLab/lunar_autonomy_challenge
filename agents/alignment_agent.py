#!/usr/bin/env python

"""Navigation + ground-truth final pose alignment agent."""

from __future__ import annotations

import json
import signal
from dataclasses import asdict
from typing import Any

import carla
import numpy as np
import rerun as rr
from rich import print

import lac.params as params
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from lac.control.alignment_controller import AlignmentController, AlignmentControllerConfig
from lac.control.controller import TrajectoryTracker, TrajectoryTrackerConfig
from lac.planning.dem_grid import DEMGrid
from lac.planning.dem_planner import DEMPlannerParams, plan_path_dem, yaw_from_pose
from lac.planning.heading_aware_planner import HeadingAwarePlanner, HeadingAwarePlannerParams
from lac.planning.path_smoother import Path2D, PathSmoother, PathSmootherConfig, Trajectory2D
from lac.utils.rerun_interface import (
    Rerun,
    rerun_box_mesh,
    rerun_dem,
    rerun_dem_grid_lines,
    rerun_lander,
)
from lac.util import transform_to_numpy, wrap_angle


DEFAULT_GOAL_XY_M = (0.0, 2.5)
DEFAULT_GOAL_YAW_RAD = -np.pi / 2
DEFAULT_DEM_MAP_PATH = "/home/shared/data_raw/Lunar/LAC/maps/competition/Moon_Map_01_preset_0.dat"
DEFAULT_ARM_RAISE_WAIT_FRAMES = 80
DEFAULT_USE_HEADING_AWARE_PLANNER = True
DEFAULT_DEM_SWAP_XY = True
DEFAULT_RERUN_ENABLED = True


def get_entry_point():
    return "AlignmentAgent"


class AlignmentAgent(AutonomousAgent):
    def setup(self, path_to_conf_file: str):
        self.config: dict[str, Any] = {}
        if path_to_conf_file:
            try:
                self.config = json.load(open(path_to_conf_file))
            except Exception:
                self.config = {}

        nav_cfg = self.config.get("navigation", {})
        dem_cfg = self.config.get("dem", {})
        planner_cfg_json = self.config.get("planner", {})
        align_cfg = self.config.get("alignment", {})
        vis_cfg = self.config.get("visualization", {})

        self.goal_xy_m = tuple(nav_cfg.get("goal_xy_m", DEFAULT_GOAL_XY_M))
        self.goal_yaw_rad = float(nav_cfg.get("goal_yaw_rad", DEFAULT_GOAL_YAW_RAD))
        self.use_heading_aware_planner = bool(
            planner_cfg_json.get("use_heading_aware_planner", DEFAULT_USE_HEADING_AWARE_PLANNER)
        )
        self.arm_raise_wait_frames = int(
            nav_cfg.get("arm_raise_wait_frames", DEFAULT_ARM_RAISE_WAIT_FRAMES)
        )
        self.dem_swap_xy = bool(dem_cfg.get("swap_xy", DEFAULT_DEM_SWAP_XY))
        self.dem_map_path = str(dem_cfg.get("map_path", DEFAULT_DEM_MAP_PATH))
        self.approach_handoff_idx_buffer = int(nav_cfg.get("approach_handoff_idx_buffer", 2))
        self.approach_handoff_timeout_steps = int(nav_cfg.get("approach_handoff_timeout_steps", 60))
        self.rerun_enabled = bool(vis_cfg.get("rerun_enabled", DEFAULT_RERUN_ENABLED))

        self.step = 0
        self.phase = "nav"
        self.current_pose = transform_to_numpy(self.get_initial_position())
        self.current_v = 0.0
        self.current_w = 0.0
        self.approach_terminal_wait_steps = 0
        self.rerun_rover_positions: list[np.ndarray] = []
        self.path_vis_z_offset_m = 0.5 * params.WHEEL_DIAMETER + params.WHEEL_CENTER_Z_OFFSET
        self.cameras = params.CAMERA_CONFIG_INIT
        for cam in self.cameras:
            self.cameras[cam] = self.cameras[cam].copy()
            self.cameras[cam]["active"] = False
            self.cameras[cam]["semantic"] = False

        self.initial_pose = self.current_pose.copy()
        self.start_xy_m = (float(self.initial_pose[0, 3]), float(self.initial_pose[1, 3]))
        self.initial_lander_pose_rover = transform_to_numpy(self.get_initial_lander_position())
        self.initial_lander_pose_world = self.initial_pose @ self.initial_lander_pose_rover

        map_arr = np.load(self.dem_map_path, allow_pickle=True)
        self.dem = DEMGrid.from_map_array(
            map_arr, cell_size=params.CELL_WIDTH, swap_xy=self.dem_swap_xy
        )
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
            lander_buffer_m=float(planner_cfg_json.get("lander_buffer_m", 0.5)),
            lander_center_xy_m=(0.0, 0.0),
            do_spline=True,
        )
        smoother_cfg = PathSmootherConfig(
            cell_size=params.CELL_WIDTH,
            ds=float(planner_cfg_json.get("smoother_ds", 0.10)),
            v_nominal=float(planner_cfg_json.get("v_nominal", params.TARGET_SPEED)),
            v_max=float(planner_cfg_json.get("v_max", 0.6)),
            max_omega=float(planner_cfg_json.get("smoother_max_omega", params.MAX_STEER)),
            max_lat_acc=float(planner_cfg_json.get("smoother_max_lat_acc", 0.5)),
        )

        if self.use_heading_aware_planner:
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
                goal_xy=self.goal_xy_m,
                goal_yaw_rad=self.goal_yaw_rad,
                cell_size=params.CELL_WIDTH,
            )
            self.path_world = heading_result.path_xy
            self.trajectory = heading_result.trajectory
        else:
            self.path_world, _, path_debug = plan_path_dem(
                self.dem_z,
                cell_size=params.CELL_WIDTH,
                start_xy=self.start_xy_m,
                goal_xy=self.goal_xy_m,
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
                Path2D(xy=np.asarray(self.path_world), meta=path_debug),
                self.dem_z,
                initial_pose=self.initial_pose,
                goal_yaw_rad=self.goal_yaw_rad,
            )

        tracker_cfg = TrajectoryTrackerConfig(
            target_speed=float(planner_cfg_json.get("v_nominal", params.TARGET_SPEED)),
            max_v=float(planner_cfg_json.get("v_max", 0.6)),
            max_w=float(planner_cfg_json.get("tracker_max_w", 0.5)),
            max_w_accel=0.4,
            lookahead_time=0.6,
            min_lookahead=0.20,
            max_lookahead=0.80,
            k_heading=0.8,
            k_cross_track=0.3,
            k_goal_v=0.8,
            stop_pos_tol=float(planner_cfg_json.get("tracker_stop_pos_tol", 0.2)),
            stop_heading_tol=float(planner_cfg_json.get("tracker_stop_heading_tol", 0.2)),
            default_dt=params.DT,
        )
        self.tracker = TrajectoryTracker(tracker_cfg)

        # Build global target pose from lander frame target.
        target_point_lander = np.asarray(
            align_cfg.get("target_point_lander_m", [0.0, 2.0, 0.0]), dtype=np.float64
        )
        target_yaw_lander = float(align_cfg.get("target_yaw_lander_rad", 0.0))
        lander_R_world = self.initial_lander_pose_world[:3, :3]
        lander_t_world = self.initial_lander_pose_world[:3, 3]
        p_target_world = lander_R_world @ target_point_lander + lander_t_world
        yaw_lander_world = float(yaw_from_pose(self.initial_lander_pose_world))
        yaw_target_world = float(wrap_angle(yaw_lander_world + target_yaw_lander))
        self.target_pose_xyt = np.array(
            [float(p_target_world[0]), float(p_target_world[1]), yaw_target_world], dtype=np.float64
        )
        wheel_xy = np.asarray(params.WHEEL_RIG_POINTS, dtype=np.float64)[:, :2]
        wheel_base_len = float(np.max(wheel_xy[:, 0]) - np.min(wheel_xy[:, 0]))
        wheel_base_wid = float(np.max(wheel_xy[:, 1]) - np.min(wheel_xy[:, 1]))
        box_height = float(max(params.WHEEL_DIAMETER, 0.30))
        self.rover_box_size_xyz = np.array(
            [wheel_base_len, wheel_base_wid, box_height], dtype=np.float64
        )
        z_target = self.dem.sample_height(
            float(self.target_pose_xyt[0]), float(self.target_pose_xyt[1])
        ) + 0.5 * float(self.rover_box_size_xyz[2])
        self.target_box_center_xyz = np.array(
            [float(self.target_pose_xyt[0]), float(self.target_pose_xyt[1]), float(z_target)],
            dtype=np.float64,
        )

        ctrl_cfg_dict = asdict(AlignmentControllerConfig())
        ctrl_cfg_dict.update(align_cfg.get("controller", {}))
        self.alignment_controller = AlignmentController(AlignmentControllerConfig(**ctrl_cfg_dict))

        print(
            f"[green]AlignmentAgent planned nav path with {len(self.path_world)} waypoints, "
            f"{len(self.trajectory.t)} samples."
        )
        print(
            f"[green]Alignment target world pose: x={self.target_pose_xyt[0]:.3f}, "
            f"y={self.target_pose_xyt[1]:.3f}, yaw={self.target_pose_xyt[2]:.3f} rad"
        )
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
                target_box = rerun_box_mesh(
                    center_xyz=tuple(self.target_box_center_xyz.tolist()),
                    size_xyz=tuple(self.rover_box_size_xyz.tolist()),
                    yaw_rad=float(self.target_pose_xyt[2]),
                    rgba=(255, 165, 0, 120),
                )
                Rerun.log_3d_mesh(target_box, topic="/world/alignment_target_box", static=True)

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
        signal.signal(signal.SIGINT, self.handle_interrupt)

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

    def _switch_to_alignment_phase(self):
        if self.phase == "align":
            return
        self.phase = "align"
        self.alignment_controller.reset()
        print("[yellow]Switching to alignment phase (GT pose controller).")

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
            rover_yaw = yaw_from_pose(self.current_pose)
            rover_xy = self.current_pose[:2, 3]
            rover_z = self.dem.sample_height(float(rover_xy[0]), float(rover_xy[1])) + 0.5 * float(
                self.rover_box_size_xyz[2]
            )
            rover_box = rerun_box_mesh(
                center_xyz=(float(rover_xy[0]), float(rover_xy[1]), float(rover_z)),
                size_xyz=tuple(self.rover_box_size_xyz.tolist()),
                yaw_rad=float(rover_yaw),
                rgba=(0, 180, 255, 140),
            )
            Rerun.log_3d_mesh(rover_box, topic="/world/rover_box", static=False)

        if self.phase == "nav":
            if self.step < self.arm_raise_wait_frames:
                return carla.VehicleVelocityControl(0.0, 0.0)

            v_cmd, w_cmd = self.tracker.compute_command(
                reference=self.trajectory,
                current_pose=self.current_pose,
                dt=params.DT,
            )
            self.current_v, self.current_w = v_cmd, w_cmd

            nearest_idx = int(self.tracker.last_debug.get("nearest_idx", 0))
            near_end_idx = max(0, len(self.trajectory.t) - 1 - self.approach_handoff_idx_buffer)
            reached_goal = bool(self.tracker.last_debug.get("reached_goal", False))
            approach_complete = nearest_idx >= near_end_idx
            if reached_goal or approach_complete:
                self.approach_terminal_wait_steps += 1
            else:
                self.approach_terminal_wait_steps = 0

            if self.approach_terminal_wait_steps >= self.approach_handoff_timeout_steps:
                self._switch_to_alignment_phase()
                return carla.VehicleVelocityControl(0.0, 0.0)

            return carla.VehicleVelocityControl(v_cmd, w_cmd)

        # Alignment phase (uses GT pose).
        x, y = float(self.current_pose[0, 3]), float(self.current_pose[1, 3])
        yaw = float(yaw_from_pose(self.current_pose))
        v_cmd, w_cmd, aligned, dbg = self.alignment_controller.step(
            current_xyt=(x, y, yaw),
            target_xyt=self.target_pose_xyt,
            dt=params.DT,
        )
        print(
            f"[bold cyan]align phase[/] step={self.step} "
            f"| rho={dbg['rho']:.3f} | alpha={dbg['alpha']:.3f} | e_theta={dbg['e_theta']:.3f} "
            f"| good={dbg['good_count']} | aligned={aligned} | v={v_cmd:.3f} w={w_cmd:.3f}"
        )
        if aligned:
            print("[bold green]Alignment success. Mission complete.")
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)
        return carla.VehicleVelocityControl(v_cmd, w_cmd)

    def finalize(self):
        print("[blue]AlignmentAgent finalize")
