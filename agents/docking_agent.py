#!/usr/bin/env python

"""Two-phase docking agent: path tracking + fiducial visual servoing."""

from __future__ import annotations

import json
import signal
from typing import Any
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import carla
import numpy as np
import rerun as rr
from rich import print

import lac.params as params
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from lac.control.controller import TrajectoryTracker, TrajectoryTrackerConfig
from lac.control.fiducial_docking_controller import FiducialDockingConfig, FiducialDockingController
from lac.control.docking_profiles import DockingMode, make_docking_mode_profile
from lac.control.fiducial_docking_pose_estimator import (
    DockingPoseEstimatorConfig,
    FiducialDockingPoseEstimator,
)
from lac.control.docking_success_checker import DockingSuccessChecker
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
from lac.utils.visualization import overlay_tag_detections
from lac.utils.frames import get_cam_pose_rover
from lac.util import transform_to_numpy, wrap_angle


DEFAULT_GOAL_XY_M = (0.0, 2.5)
DEFAULT_GOAL_YAW_RAD = -np.pi / 2
DEFAULT_DEM_MAP_PATH = "/home/shared/data_raw/Lunar/LAC/maps/competition/Moon_Map_01_preset_0.dat"

DEFAULT_ARM_RAISE_WAIT_FRAMES = 80
DEFAULT_USE_HEADING_AWARE_PLANNER = True
DEFAULT_DEM_SWAP_XY = True

DEFAULT_CHARGE_SUCCESS_DELTA_WH = 2.0
DEFAULT_RERUN_ENABLED = True


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

        nav_cfg = self.config.get("navigation", {})
        dem_cfg = self.config.get("dem", {})
        docking_cfg = self.config.get("docking", {})
        vis_cfg = self.config.get("visualization", {})
        planner_cfg_json = self.config.get("planner", {})
        perception_cfg = self.config.get("perception", {})

        self.goal_xy_m = tuple(nav_cfg.get("goal_xy_m", DEFAULT_GOAL_XY_M))
        self.goal_yaw_rad = float(nav_cfg.get("goal_yaw_rad", DEFAULT_GOAL_YAW_RAD))
        self.rerun_enabled = bool(vis_cfg.get("rerun_enabled", DEFAULT_RERUN_ENABLED))
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
        self.approach_handoff_fiducial_dist_m = float(
            nav_cfg.get("approach_handoff_fiducial_dist_m", 1.0)
        )
        self.docking_estimate_hold_steps = int(perception_cfg.get("docking_estimate_hold_steps", 3))
        self.dock_settle_steps = int(docking_cfg.get("settle_steps", 20))
        self.dock_settle_on_reacquire = bool(docking_cfg.get("settle_on_reacquire", False))
        self.dock_debug_print_every_steps = int(docking_cfg.get("debug_print_every_steps", 1))
        self.dock_debug_rerun_scalars = bool(docking_cfg.get("debug_rerun_scalars", True))
        debug_log_cfg = docking_cfg.get("debug_log", {})
        self.dock_debug_log_enabled = bool(debug_log_cfg.get("enabled", False))
        self.dock_debug_log_flush_every = max(1, int(debug_log_cfg.get("flush_every", 1)))
        self.dock_debug_log_path_cfg = str(
            debug_log_cfg.get("path", "logs/docking_debug_{timestamp}.jsonl")
        )
        self._dock_debug_log_fp = None
        self._dock_debug_log_write_count = 0

        self.step = 0
        self.current_v = 0.0
        self.current_w = 0.0
        self.rerun_rover_positions: list[np.ndarray] = []
        self.path_vis_z_offset_m = 0.5 * params.WHEEL_DIAMETER + params.WHEEL_CENTER_Z_OFFSET

        self.phase = "nav"
        self.dock_start_power = None
        self.last_dock_cmd = (0.0, 0.0)
        self.approach_terminal_wait_steps = 0
        self.last_docking_estimate: dict[str, Any] | None = None
        self.last_docking_estimate_step: int = -(10**9)
        self.dock_settle_until_step = -1
        self.prev_fid_visible = False
        self.prev_ctrl_phase = "n/a"
        self.docking_mode: DockingMode = docking_cfg.get(
            "mode", self.config.get("docking_mode", "side")
        )
        self.mode_profile = make_docking_mode_profile(self.docking_mode)
        success_cfg = docking_cfg.get("success", {})
        if "position_tol_xy_m" in success_cfg:
            self.mode_profile.pos_tol_xy_m = float(success_cfg["position_tol_xy_m"])
        if "angle_tol_deg" in success_cfg:
            self.mode_profile.rot_tol_rad = float(np.deg2rad(success_cfg["angle_tol_deg"]))
        elif "angle_tol_rad" in success_cfg:
            self.mode_profile.rot_tol_rad = float(success_cfg["angle_tol_rad"])
        if "require_power_recovery" in success_cfg:
            self.mode_profile.require_power_recovery = bool(success_cfg["require_power_recovery"])
        if "power_delta_wh" in success_cfg:
            self.mode_profile.power_delta_wh = float(success_cfg["power_delta_wh"])
        self.charge_success_delta_wh = float(
            success_cfg.get("power_delta_wh", DEFAULT_CHARGE_SUCCESS_DELTA_WH)
        )
        hold_frames = int(success_cfg.get("hold_frames", 8))

        self.initial_pose = transform_to_numpy(self.get_initial_position())
        self.current_pose = self.initial_pose.copy()
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
            self.path_cost = heading_result.total_cost
            self.path_debug = heading_result.debug
            self.trajectory = heading_result.trajectory
        else:
            self.path_world, self.path_cost, self.path_debug = plan_path_dem(
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
                Path2D(xy=np.asarray(self.path_world), meta=self.path_debug),
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

        ctrl_cfg_dict = asdict(FiducialDockingConfig())
        ctrl_cfg_dict.update(docking_cfg.get("controller", {}))
        target_cfg = docking_cfg.get("target", {})
        tag_point_lander = np.asarray(
            target_cfg.get("tag_point_lander_m", [0.0, 0.662, 0.325]), dtype=np.float64
        )
        target_point_lander = np.asarray(
            target_cfg.get("target_point_lander_m", [0.0, 3.0, 0.0]), dtype=np.float64
        )
        psi_des_target = float(target_cfg.get("psi_des_rad", ctrl_cfg_dict.get("psi_des", 0.0)))
        tag_yaw_lander = float(target_cfg.get("tag_yaw_lander_rad", np.pi))
        target_yaw_lander = float(
            target_cfg.get("target_yaw_lander_rad", tag_yaw_lander - psi_des_target)
        )

        # Make controller and target-box viz use the same configured target.
        # Controller now tracks the virtual box pose directly:
        # T_base_box = T_base_tag @ T_tag_box, then target is identity in base frame.
        self.T_tag_box = np.eye(4, dtype=np.float64)
        self._target_debug_payload: dict[str, Any] = {}
        try:
            lander_R_world = self.initial_lander_pose_world[:3, :3]
            lander_t_world = self.initial_lander_pose_world[:3, 3]
            p_tag_world = lander_R_world @ tag_point_lander + lander_t_world
            p_target_world = lander_R_world @ target_point_lander + lander_t_world
            yaw_lander_world = float(yaw_from_pose(self.initial_lander_pose_world))
            yaw_tag_world = float(wrap_angle(yaw_lander_world + tag_yaw_lander))
            yaw_base_des = float(wrap_angle(yaw_tag_world - psi_des_target))
            c, s = np.cos(yaw_base_des), np.sin(yaw_base_des)
            R_world_base_des = np.array([[c, -s], [s, c]], dtype=np.float64)
            # Desired tag position in base frame at the configured target pose.
            d_world_xy = np.asarray(p_tag_world[:2] - p_target_world[:2], dtype=np.float64)
            d_base_xy = R_world_base_des.T @ d_world_xy
            # Convert configured lander-frame poses into T_tag_box (3D pose of virtual box in tag frame).
            c_tag, s_tag = np.cos(tag_yaw_lander), np.sin(tag_yaw_lander)
            R_lander_tag = np.array(
                [[c_tag, -s_tag, 0.0], [s_tag, c_tag, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            )
            c_box, s_box = np.cos(target_yaw_lander), np.sin(target_yaw_lander)
            R_lander_box = np.array(
                [[c_box, -s_box, 0.0], [s_box, c_box, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            )
            T_lander_tag = np.eye(4, dtype=np.float64)
            T_lander_tag[:3, :3] = R_lander_tag
            T_lander_tag[:3, 3] = tag_point_lander
            T_lander_box = np.eye(4, dtype=np.float64)
            T_lander_box[:3, :3] = R_lander_box
            T_lander_box[:3, 3] = target_point_lander
            self.T_tag_box = np.linalg.inv(T_lander_tag) @ T_lander_box

            # Control target is now the virtual box frame origin.
            ctrl_cfg_dict["x_des"] = 0.0
            ctrl_cfg_dict["y_des"] = 0.0
            ctrl_cfg_dict["psi_des"] = 0.0
            self.docking_box_center_xyz = np.array(
                [float(p_target_world[0]), float(p_target_world[1]), 0.0], dtype=np.float64
            )
            self.docking_box_yaw_rad = yaw_base_des
            self._target_debug_payload = {
                "psi_des_target": float(psi_des_target),
                "tag_yaw_lander": float(tag_yaw_lander),
                "target_yaw_lander": float(target_yaw_lander),
                "legacy_tag_offset_x_des": float(d_base_xy[0]),
                "legacy_tag_offset_y_des": float(d_base_xy[1]),
                "tag_point_lander_m": tag_point_lander.tolist(),
                "target_point_lander_m": target_point_lander.tolist(),
                "T_tag_box": self.T_tag_box.tolist(),
            }
        except Exception:
            self.docking_box_center_xyz = None
            self.docking_box_yaw_rad = 0.0
            self._target_debug_payload = {"target_build_failed": True}

        ctrl_cfg_dict["antenna_xy_tol_m"] = self.mode_profile.pos_tol_xy_m
        ctrl_cfg_dict["antenna_rot_tol_rad"] = self.mode_profile.rot_tol_rad
        self.visual_servo = FiducialDockingController(FiducialDockingConfig(**ctrl_cfg_dict))
        wheel_xy = np.asarray(params.WHEEL_RIG_POINTS, dtype=np.float64)[:, :2]
        wheel_base_len = float(np.max(wheel_xy[:, 0]) - np.min(wheel_xy[:, 0]))
        wheel_base_wid = float(np.max(wheel_xy[:, 1]) - np.min(wheel_xy[:, 1]))
        box_height = float(max(params.WHEEL_DIAMETER, 0.30))
        self.rover_box_size_xyz = np.array(
            [wheel_base_len, wheel_base_wid, box_height],
            dtype=np.float64,
        )
        if self.docking_box_center_xyz is not None:
            z_des = self.dem.sample_height(
                float(self.docking_box_center_xyz[0]), float(self.docking_box_center_xyz[1])
            ) + 0.5 * float(self.rover_box_size_xyz[2])
            self.docking_box_center_xyz[2] = float(z_des)

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
        # Refresh estimator intrinsics with final camera configuration.
        self.docking_pose_estimator = FiducialDockingPoseEstimator(
            camera_config=self.cameras,
            config=DockingPoseEstimatorConfig(
                cam_name=self.mode_profile.camera_name,
                target_tag_id=self.mode_profile.target_tag_id,
                lander_target_point_lander=self.mode_profile.lander_target_point_lander,
                lander_target_axis_lander=self.mode_profile.lander_target_axis_lander,
                rover_target_point_rover=self.mode_profile.rover_target_point_rover,
                rover_target_axis_rover=self.mode_profile.rover_target_axis_rover,
                locator_point_lander=np.array([0.0, 0.662, 0.325], dtype=np.float64),
                lander_target_rpy_lander=np.array([0.0, 0.0, np.pi], dtype=np.float64),
                tag_size_m=0.253,
            ),
        )
        self.T_base_cam_servo = get_cam_pose_rover(self.mode_profile.camera_name)
        self.locator_point_lander = np.array([0.0, 0.662, 0.325], dtype=np.float64)
        self.locator_point_world = (
            self.initial_lander_pose_world[:3, :3] @ self.locator_point_lander
            + self.initial_lander_pose_world[:3, 3]
        )
        self.search_point_k_w = float(
            docking_cfg.get(
                "search_point_k_w",
                docking_cfg.get("controller", {}).get("k_w_rot", 1.2),
            )
        )
        self.search_point_max_w = float(
            docking_cfg.get(
                "search_point_max_w",
                docking_cfg.get("controller", {}).get("max_w", 0.5),
            )
        )
        self.success_checker = DockingSuccessChecker(self.mode_profile, hold_frames=hold_frames)
        if self.dock_debug_log_enabled:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path_str = self.dock_debug_log_path_cfg.replace("{timestamp}", ts)
            log_path = Path(log_path_str).expanduser()
            if not log_path.is_absolute():
                log_path = Path.cwd() / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._dock_debug_log_fp = log_path.open("a", encoding="utf-8")
            self._dock_log_event(
                event="log_start",
                payload={
                    "path": str(log_path),
                    "mode": self.docking_mode,
                    "target_cfg": docking_cfg.get("target", {}),
                    "controller_cfg": ctrl_cfg_dict,
                },
            )
            self._dock_log_event(event="target_frames_built", payload=self._target_debug_payload)
            print(f"[cyan]Dock debug log enabled:[/] {log_path}")

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
                if self.docking_box_center_xyz is not None:
                    docking_target_box = rerun_box_mesh(
                        center_xyz=tuple(self.docking_box_center_xyz.tolist()),
                        size_xyz=tuple(self.rover_box_size_xyz.tolist()),
                        yaw_rad=self.docking_box_yaw_rad,
                        rgba=(255, 165, 0, 120),
                    )
                    Rerun.log_3d_mesh(
                        docking_target_box, topic="/world/docking_target_box", static=True
                    )
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
            f"{len(self.trajectory.t)} trajectory samples. "
            f"mode={self.docking_mode}, goal={self.goal_xy_m}, goal_yaw={self.goal_yaw_rad:.3f}"
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

    def _switch_to_docking_phase(
        self, initial_ctrl_phase: str = "search", preserve_latest_estimate: bool = False
    ):
        if self.phase == "dock":
            return
        self.phase = "dock"
        self.visual_servo.reset(initial_phase=initial_ctrl_phase)
        self.dock_start_power = float(self.get_current_power())
        self.success_checker.reset(start_power_wh=self.dock_start_power)
        if not preserve_latest_estimate:
            self.last_docking_estimate = None
            self.last_docking_estimate_step = -(10**9)
        self.prev_fid_visible = False
        self.dock_settle_until_step = self.step + self.dock_settle_steps
        self.set_radiator_cover_state(carla.RadiatorCoverState.Open)
        print(
            f"[yellow]Switching to docking visual servo phase "
            f"(mode={self.docking_mode}, ctrl_phase={initial_ctrl_phase})."
        )

    def _search_pointing_command(self) -> tuple[float, float, float]:
        """Search by pointing toward known fiducial location in world frame."""
        yaw = yaw_from_pose(self.current_pose)
        dx = float(self.locator_point_world[0] - self.current_pose[0, 3])
        dy = float(self.locator_point_world[1] - self.current_pose[1, 3])
        desired_yaw = float(np.arctan2(dy, dx))
        yaw_err = float(wrap_angle(desired_yaw - yaw))
        w_cmd = float(
            np.clip(
                self.search_point_k_w * yaw_err, -self.search_point_max_w, self.search_point_max_w
            )
        )
        return 0.0, w_cmd, yaw_err

    def run_step(self, input_data):
        print("[bold bright_black]-----------------------------------------------[/]\n")
        if self.step == 0:
            self.initialize()
        self.step += 1
        self.current_pose = transform_to_numpy(self.get_transform())
        grayscale_data = input_data.get("Grayscale", {})
        front_left_img = grayscale_data.get(carla.SensorPosition.FrontLeft)
        right_img = grayscale_data.get(carla.SensorPosition.Right)
        active_img = right_img if self.mode_profile.camera_name == "Right" else front_left_img
        docking_estimate, active_detections, _ = self.docking_pose_estimator.estimate(active_img)
        fid_handoff_dist = np.inf
        if docking_estimate is not None:
            rel_target = np.asarray(
                docking_estimate.get("rel_target_pos_rover_m", [np.nan, np.nan, np.nan])
            )
            fid_handoff_dist = float(np.linalg.norm(rel_target[:2]))
        if docking_estimate is not None:
            self.last_docking_estimate = docking_estimate
            self.last_docking_estimate_step = self.step
        estimate_age = self.step - self.last_docking_estimate_step
        using_stale_estimate = (
            docking_estimate is None
            and self.last_docking_estimate is not None
            and estimate_age <= self.docking_estimate_hold_steps
        )
        control_estimate = self.last_docking_estimate if using_stale_estimate else docking_estimate
        fid_visible = control_estimate is not None
        if self.phase == "dock" and self.prev_fid_visible and not fid_visible:
            print(
                f"[bold red]dock perception lost[/] at step={self.step} "
                f"(stale_age={estimate_age}, hold_steps={self.docking_estimate_hold_steps})"
            )
            self._dock_log_event(
                event="perception_lost",
                payload={
                    "estimate_age": estimate_age,
                    "hold_steps": self.docking_estimate_hold_steps,
                },
            )
        if self.phase == "dock" and (not self.prev_fid_visible) and fid_visible:
            print(
                f"[bold green]dock perception reacquired[/] at step={self.step} "
                f"(stale_est={using_stale_estimate}, age={estimate_age})"
            )
            self._dock_log_event(
                event="perception_reacquired",
                payload={"stale_est": using_stale_estimate, "estimate_age": estimate_age},
            )
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
            if front_left_img is not None:
                fl_detections = self.docking_pose_estimator.detect(front_left_img)
                front_left_overlay = overlay_tag_detections(front_left_img, fl_detections)
                Rerun.log_img_seq(
                    "/world/cameras/front_left", self.step, front_left_overlay, adjust_rgb=True
                )
            if right_img is not None:
                right_overlay = overlay_tag_detections(
                    right_img,
                    active_detections if self.mode_profile.camera_name == "Right" else [],
                )
                Rerun.log_img_seq("/world/cameras/right", self.step, right_overlay, adjust_rgb=True)

        if self.phase == "nav":
            if self.step < self.arm_raise_wait_frames:
                return carla.VehicleVelocityControl(0.0, 0.0)

            v_cmd, w_cmd = self.tracker.compute_command(
                reference=self.trajectory,
                current_pose=self.current_pose,
                dt=params.DT,
            )
            self.current_v, self.current_w = v_cmd, w_cmd

            dist_to_goal = float(self.tracker.last_debug.get("dist_to_goal_m", np.inf))
            nearest_idx = int(self.tracker.last_debug.get("nearest_idx", 0))
            traj_last_idx = max(0, len(self.trajectory.t) - 1)
            near_end_idx = max(0, len(self.trajectory.t) - 1 - self.approach_handoff_idx_buffer)
            approach_complete = nearest_idx >= near_end_idx
            print(
                f"[bold bright_yellow]approach phase:[/] dist_to_goal={dist_to_goal:.2f} m | "
                f"nearest_idx={nearest_idx}/{traj_last_idx} | handoff_idx>={near_end_idx} | "
                f"fid_dist={fid_handoff_dist:.2f} m"
            )

            # Primary handoff condition: fiducial visible and sufficiently close.
            if (
                docking_estimate is not None
                and fid_handoff_dist <= self.approach_handoff_fiducial_dist_m
            ):
                print(
                    f"[bold bright_yellow]Switching to docking handoff[/] "
                    f"(reason=fiducial_visible_close, fid_dist={fid_handoff_dist:.2f} m, "
                    f"threshold={self.approach_handoff_fiducial_dist_m:.2f} m, "
                    f"nearest_idx={nearest_idx}/{traj_last_idx})."
                )
                self._switch_to_docking_phase(
                    initial_ctrl_phase="yaw_align", preserve_latest_estimate=True
                )
                return carla.VehicleVelocityControl(0.0, 0.0)

            reached_goal = self.tracker.last_debug.get("reached_goal", False)
            if reached_goal or approach_complete:
                self.approach_terminal_wait_steps += 1
            else:
                self.approach_terminal_wait_steps = 0
            if self.approach_terminal_wait_steps >= self.approach_handoff_timeout_steps:
                reason = "reached_goal_timeout" if reached_goal else "near_end_timeout"
                print(
                    f"[bold bright_yellow]Switching to docking handoff[/] "
                    f"(reason={reason}, wait_steps={self.approach_terminal_wait_steps})."
                )
                self._switch_to_docking_phase(initial_ctrl_phase="search")
                return carla.VehicleVelocityControl(0.0, 0.0)

            return carla.VehicleVelocityControl(self.current_v, self.current_w)

        # If fiducial was just acquired, briefly stop to settle before servoing.
        if self.phase == "dock":
            if self.dock_settle_on_reacquire and fid_visible and not self.prev_fid_visible:
                self.dock_settle_until_step = max(
                    self.dock_settle_until_step, self.step + self.dock_settle_steps
                )
            self.prev_fid_visible = fid_visible
            if self.step < self.dock_settle_until_step:
                settle_left = self.dock_settle_until_step - self.step
                print(
                    f"[bold bright_yellow]dock settle:[/] holding still for {settle_left} more steps"
                )
                return carla.VehicleVelocityControl(0.0, 0.0)

        if control_estimate is None:
            v_cmd, w_cmd, yaw_err = self._search_pointing_command()
            controller_ready = False
            _, _, ctrl_dbg = self.visual_servo.update(
                tag_detection=None,
                T_base_cam=self.T_base_cam_servo,
                dt=params.DT,
            )
            dbg = {
                "phase": "search_pointing",
                "target_found": False,
                "antenna_x_err_m": np.nan,
                "antenna_y_err_m": np.nan,
                "antenna_rot_err_rad": np.nan,
                "aligned": False,
                "search_yaw_err_rad": yaw_err,
                "ctrl_stage": ctrl_dbg.get("stage", "n/a"),
                "tag_age_s": ctrl_dbg.get("tag_age_s", np.inf),
                "alpha": np.nan,
                "e_psi": np.nan,
                "rho": np.nan,
            }
        else:
            T_cam_tag = np.asarray(control_estimate["T_cam_tag"], dtype=np.float64)
            T_base_tag = self.T_base_cam_servo @ T_cam_tag
            T_base_box = T_base_tag @ self.T_tag_box
            tag_detection = {
                "T_cam_tag": T_base_box,
                "confidence": 1.0,
            }
            v_cmd, w_cmd, dbg = self.visual_servo.update(
                tag_detection=tag_detection,
                T_base_cam=np.eye(4, dtype=np.float64),
                dt=params.DT,
            )
            controller_ready = bool(
                abs(float(dbg.get("e_x", np.inf))) < float(self.visual_servo.cfg.x_tol)
                and abs(float(dbg.get("e_y", np.inf))) < float(self.visual_servo.cfg.y_tol)
            )
            if bool(self.visual_servo.cfg.use_final_yaw):
                controller_ready = controller_ready and (
                    abs(float(dbg.get("e_psi", np.inf))) < float(self.visual_servo.cfg.psi_tol)
                )
            # Cross-check: reconstruct tag planar pose in base frame for debugging.
            try:
                box_x = float(T_base_box[0, 3])
                box_y = float(T_base_box[1, 3])
                box_yaw = float(np.arctan2(T_base_box[1, 0], T_base_box[0, 0]))
                ex = box_x - float(self.visual_servo.cfg.x_des)
                ey = box_y - float(self.visual_servo.cfg.y_des)
                alpha = float(np.arctan2(ey, max(ex, float(self.visual_servo.cfg.eps))))
                epsi = float(wrap_angle(box_yaw - float(self.visual_servo.cfg.psi_des)))
                rho = float(np.hypot(ex, ey))
                dbg["box_x_base_m"] = box_x
                dbg["box_y_base_m"] = box_y
                dbg["box_yaw_base_rad"] = box_yaw
                dbg["alpha_meas"] = alpha
                dbg["e_psi_meas"] = epsi
                dbg["rho_meas"] = rho
            except Exception:
                pass
        self.last_dock_cmd = (v_cmd, w_cmd)
        success, success_dbg = self.success_checker.check(
            control_estimate, current_power_wh=float(self.get_current_power())
        )
        if success:
            print(f"[bold green]Docking success ({success_dbg.get('reason')}). Mission complete.")
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)
        if controller_ready and self.mode_profile.require_power_recovery:
            print(
                "[cyan]Geometric alignment achieved; waiting for charging "
                f"(reason={success_dbg.get('reason')}, "
                f"power_delta={success_dbg.get('power_delta_wh', np.nan):.2f} Wh)"
            )

        ctrl_phase = dbg.get("phase", dbg.get("stage", "n/a"))
        should_print = (
            self.step % max(1, self.dock_debug_print_every_steps) == 0
            or ctrl_phase != self.prev_ctrl_phase
        )
        if should_print:
            print(
                f"[bold magenta]Dock phase[/] step={self.step} | "
                f"ctrl_phase={ctrl_phase} | mode={self.docking_mode} | "
                f"found={dbg.get('target_found', False)} | "
                f"stale_est={using_stale_estimate} (age={estimate_age}) | "
                f"tag_age={dbg.get('tag_age_s', np.nan):.2f}s | "
                f"ant_x={dbg.get('antenna_x_err_m', dbg.get('e_x', np.nan)):.3f} m | "
                f"ant_y={dbg.get('antenna_y_err_m', dbg.get('e_y', np.nan)):.3f} m | "
                f"alpha={dbg.get('alpha', dbg.get('alpha_meas', np.nan)):.3f} rad | "
                f"epsi={dbg.get('e_psi', dbg.get('e_psi_meas', np.nan)):.3f} rad | "
                f"rho={dbg.get('rho', dbg.get('rho_meas', np.nan)):.3f} m | "
                f"rot={dbg.get('antenna_rot_err_rad', dbg.get('e_psi', np.nan)):.3f} rad | "
                f"ctrl_ready={controller_ready} | success_reason={success_dbg.get('reason')} | "
                f"v={v_cmd:.3f} w={w_cmd:.3f}"
            )
        self.prev_ctrl_phase = ctrl_phase

        if self.rerun_enabled and self.dock_debug_rerun_scalars and self.phase == "dock":
            try:
                Rerun.log_scalar("/world/docking/debug/v_cmd", float(v_cmd))
                Rerun.log_scalar("/world/docking/debug/w_cmd", float(w_cmd))
                if np.isfinite(float(dbg.get("e_x", np.nan))):
                    Rerun.log_scalar("/world/docking/debug/e_x", float(dbg["e_x"]))
                if np.isfinite(float(dbg.get("e_y", np.nan))):
                    Rerun.log_scalar("/world/docking/debug/e_y", float(dbg["e_y"]))
                if np.isfinite(float(dbg.get("alpha", np.nan))):
                    Rerun.log_scalar("/world/docking/debug/alpha", float(dbg["alpha"]))
                if np.isfinite(float(dbg.get("e_psi", np.nan))):
                    Rerun.log_scalar("/world/docking/debug/e_psi", float(dbg["e_psi"]))
                if np.isfinite(float(dbg.get("rho", np.nan))):
                    Rerun.log_scalar("/world/docking/debug/rho", float(dbg["rho"]))
                if np.isfinite(float(dbg.get("tag_age_s", np.nan))):
                    Rerun.log_scalar("/world/docking/debug/tag_age_s", float(dbg["tag_age_s"]))
            except Exception:
                pass
        if self.phase == "dock":
            self._dock_log_tick(
                ctrl_phase=ctrl_phase,
                using_stale_estimate=using_stale_estimate,
                estimate_age=estimate_age,
                dbg=dbg,
                v_cmd=float(v_cmd),
                w_cmd=float(w_cmd),
                controller_ready=bool(controller_ready),
                success_reason=success_dbg.get("reason"),
            )
        return carla.VehicleVelocityControl(v_cmd, w_cmd)

    def finalize(self):
        if self._dock_debug_log_fp is not None:
            self._dock_log_event(event="log_end", payload={"step": int(self.step)})
            self._dock_debug_log_fp.close()
            self._dock_debug_log_fp = None
        print("[blue]DockingAgent finalize")

    def _dock_log_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        if self._dock_debug_log_fp is None:
            return
        row: dict[str, Any] = {
            "type": "event",
            "event": event,
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "step": int(self.step),
            "phase": str(self.phase),
        }
        if payload:
            row.update(payload)
        self._dock_debug_log_fp.write(json.dumps(self._to_jsonable(row)) + "\n")
        self._dock_debug_log_write_count += 1
        if self._dock_debug_log_write_count % self.dock_debug_log_flush_every == 0:
            self._dock_debug_log_fp.flush()

    def _dock_log_tick(
        self,
        ctrl_phase: str,
        using_stale_estimate: bool,
        estimate_age: int,
        dbg: dict[str, Any],
        v_cmd: float,
        w_cmd: float,
        controller_ready: bool,
        success_reason: str | None,
    ) -> None:
        if self._dock_debug_log_fp is None:
            return
        row = {
            "type": "tick",
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "step": int(self.step),
            "phase": str(self.phase),
            "ctrl_phase": str(ctrl_phase),
            "mode": str(self.docking_mode),
            "found": bool(dbg.get("target_found", False)),
            "stale_estimate": bool(using_stale_estimate),
            "estimate_age": int(estimate_age),
            "tag_age_s": float(dbg.get("tag_age_s", np.nan)),
            "e_x": float(dbg.get("antenna_x_err_m", dbg.get("e_x", np.nan))),
            "e_y": float(dbg.get("antenna_y_err_m", dbg.get("e_y", np.nan))),
            "alpha": float(dbg.get("alpha", dbg.get("alpha_meas", np.nan))),
            "alpha_ctrl": float(dbg.get("alpha_ctrl", np.nan)),
            "e_psi": float(dbg.get("antenna_rot_err_rad", dbg.get("e_psi", np.nan))),
            "rho": float(dbg.get("rho", dbg.get("rho_meas", np.nan))),
            "v_cmd": float(v_cmd),
            "w_cmd": float(w_cmd),
            "controller_ready": bool(controller_ready),
            "success_reason": success_reason,
            "ctrl_x_des": float(self.visual_servo.cfg.x_des),
            "ctrl_y_des": float(self.visual_servo.cfg.y_des),
            "ctrl_psi_des": float(self.visual_servo.cfg.psi_des),
            "box_x_base_m": float(dbg.get("box_x_base_m", np.nan)),
            "box_y_base_m": float(dbg.get("box_y_base_m", np.nan)),
            "box_yaw_base_rad": float(dbg.get("box_yaw_base_rad", np.nan)),
        }
        self._dock_debug_log_fp.write(json.dumps(self._to_jsonable(row)) + "\n")
        self._dock_debug_log_write_count += 1
        if self._dock_debug_log_write_count % self.dock_debug_log_flush_every == 0:
            self._dock_debug_log_fp.flush()

    def _to_jsonable(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        return obj
