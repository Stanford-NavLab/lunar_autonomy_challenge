#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Full agent

"""

import carla
import cv2 as cv
import numpy as np
import signal
from collections import deque

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.planning.temporal_arc_planner import TemporalArcPlanner
from lac.util import transform_to_numpy
from lac.slam.semantic_feature_tracker import SemanticFeatureTracker
from lac.slam.frontend import Frontend
from lac.slam.backend import Backend
from lac.planning.arc_planner import ArcPlanner
from lac.planning.waypoint_planner import WaypointPlanner
from lac.mapping.mapper import process_map
from lac.mapping.map_utils import get_geometric_score, get_rocks_score
from lac.utils.data_logger import DataLogger
from lac.utils.rerun_interface import Rerun
from lac.util import get_positions_from_poses
import lac.params as params


""" Agent parameters and settings """
EVAL = False  # Whether running in evaluation mode (disable ground truth)
BACK_CAMERAS = True

USE_GROUND_TRUTH_NAV = False  # Whether to use ground truth pose for navigation
ARM_RAISE_WAIT_FRAMES = 80  # Number of frames to wait for the arms to raise
MISSION_TIMEOUT = 100000  # Number of frames to end mission after

LOG_DATA = True  # Whether to log data
RERUN = True  # Whether to use rerun for visualization
USE_TEMPORAL = True

if EVAL:
    USE_GROUND_TRUTH_NAV = False
    DISPLAY_IMAGES = False
    LOG_DATA = False
    RERUN = False


def get_entry_point():
    return "NavAgent"


class NavAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Control variables"""
        self.current_v = 0
        self.current_w = 0

        """Initialize a counter to keep track of the number of simulation steps."""
        self.step = 0

        """ Initialize a counter for backup maneuvers. """
        self.backup_counter = 0

        """ Initialize a counter for how long the rover below a velocity threshold. """
        self.stuck_counter = 0

        """ Initialize a counter for total time that the rover is stuck."""
        self.stuck_timer = 0

        """ Camera config """
        self.cameras = params.CAMERA_CONFIG_INIT
        self.cameras["FrontLeft"] = {
            "active": True,
            "light": 1.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        self.cameras["FrontRight"] = {
            "active": True,
            "light": 1.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        # Turn on front camera light
        self.cameras["Front"]["light"] = 1.0
        if BACK_CAMERAS:
            self.cameras["BackLeft"] = {
                "active": True,
                "light": 1.0,
                "width": 1280,
                "height": 720,
                "semantic": False,
            }
            self.cameras["BackRight"] = {
                "active": True,
                "light": 1.0,
                "width": 1280,
                "height": 720,
                "semantic": False,
            }
        self.active_cameras = [cam for cam, config in self.cameras.items() if config["active"]]

        """ Planning """
        self.initial_pose = transform_to_numpy(self.get_initial_position())
        self.lander_pose = self.initial_pose @ transform_to_numpy(
            self.get_initial_lander_position()
        )
        self.planner = WaypointPlanner(self.initial_pose, triangle_loops=True)
        self.arc_planner = ArcPlanner()
        arc_config_val = 31
        arc_duration_val = 8
        max_omega = 0.8
        if USE_TEMPORAL == True:
            self.arc_planner = TemporalArcPlanner(
                arc_config=arc_config_val, arc_duration=arc_duration_val, max_omega=max_omega
            )
        else:
            self.arc_planner = ArcPlanner(
                arc_config=arc_config_val, arc_duration=arc_duration_val, max_omega=max_omega
            )
        self.arcs = self.arc_planner.np_candidate_arcs

        """ State variables """
        self.current_pose = self.initial_pose
        self.current_velocity = np.zeros(3)
        self.imu_measurements = deque(maxlen=2)  # IMU measurements since last image

        """ SLAM """
        feature_tracker = SemanticFeatureTracker(self.cameras)
        back_feature_tracker = SemanticFeatureTracker(self.cameras, cam="BackLeft")
        self.frontend = Frontend(feature_tracker, back_feature_tracker, self.initial_pose)
        self.backend = Backend(self.initial_pose, feature_tracker)

        """ Data logging """
        if LOG_DATA:
            agent_name = get_entry_point()
            self.data_logger = DataLogger(self, agent_name, self.cameras)
        if RERUN:
            Rerun.init_vo()
            self.gt_poses = [self.initial_pose]

        """ Load the ground truth map for real-time score updates """
        if not EVAL:
            self.ground_truth_map = np.load(
                "/home/shared/data_raw/LAC/heightmaps/competition/Moon_Map_01_preset_4.dat",
                allow_pickle=True,
            )

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signal_received, frame):
        print("\nCtrl+C detected! Exiting mission")
        self.mission_complete()

    def initialize(self):
        # Move the arms out of the way
        self.set_front_arm_angle(params.FRONT_ARM_ANGLE_STATIC_RAD)
        self.set_back_arm_angle(params.ARM_ANGLE_STATIC_RAD)

    def image_available(self):
        return self.step % 2 == 0  # Image data is available every other step

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

    def run_backup_maneuver(self):
        print("Running backup maneuver")
        self.backup_counter += 1
        BACKUP_TIME = 5.0  # [s]
        ROTATE_TIME = 2.0  # [s]
        if self.backup_counter <= params.FRAME_RATE * BACKUP_TIME:
            # Go backwards for 3 seconds
            print("   Backing up")
            control = carla.VehicleVelocityControl(-0.2, 0.0)

        elif self.backup_counter <= params.FRAME_RATE * (BACKUP_TIME + ROTATE_TIME):
            # Rotate 90 deg/s for 1.5 seconds (overcorrecting because it isn't rotating in 1 second)
            print("   Rotating")
            control = carla.VehicleVelocityControl(0.0, np.pi / 4)
        # elif self.backup_counter <= params.FRAME_RATE * 9:
        #     # Go forward for 6 seconds
        #     print("   Moving forward")
        #     control = carla.VehicleVelocityControl(0.2, 0.0)
        else:
            self.backup_counter = 0
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        return control

    def check_stuck(self):
        # Agent is stuck if the velocity is less than 0.1 m/s
        if self.step < ARM_RAISE_WAIT_FRAMES + 10:
            return False
        is_stuck = np.linalg.norm(self.current_velocity) < 0.25 * params.TARGET_SPEED
        if is_stuck and self.stuck_timer == 0:
            self.stuck_counter += 1
            self.stuck_timer += 1
        elif is_stuck and self.stuck_timer > 0:
            self.stuck_counter += 1
        if (
            self.stuck_timer > params.FRAME_RATE * 2 and self.stuck_timer < params.FRAME_RATE * 3
        ):  # between 2 and 3 seconds
            if (self.stuck_counter / self.stuck_timer) > 0.5:
                self.stuck_counter = 0
                self.stuck_timer = 0
                return True
        elif self.stuck_timer >= params.FRAME_RATE * 3:  # more than 3 seconds
            if (self.stuck_counter / self.stuck_timer) < 0.5:
                self.stuck_counter = 0
                self.stuck_timer = 0
        return False

    def run_step(self, input_data):
        if self.step == 0:
            self.initialize()
        self.step += 1  # Starts at 0 at init, equal to 1 on the first run_step call
        print("\nStep: ", self.step)

        if self.step > MISSION_TIMEOUT:
            print("Mission timed out!")
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        if self.stuck_timer > 0:
            self.stuck_timer += 1

        if not EVAL:
            ground_truth_pose = transform_to_numpy(self.get_transform())

        if USE_GROUND_TRUTH_NAV:
            nav_pose = ground_truth_pose
        else:
            nav_pose = self.current_pose

        """ Waypoint navigation """
        waypoint, advanced = self.planner.get_waypoint(self.step, nav_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)
        if advanced:
            agent_map = self.update_map()
            if RERUN:
                geometric_score = get_geometric_score(self.ground_truth_map, agent_map)
                rocks_score = get_rocks_score(self.ground_truth_map, agent_map)
                Rerun.log_2d_seq_scalar("/scores/geometric", self.step, geometric_score)
                Rerun.log_2d_seq_scalar("/scores/rocks", self.step, rocks_score)
                Rerun.log_scalar("/metrics/rocks_score", rocks_score)

        self.imu_measurements.append(self.get_imu_data())
        control = (0.0, 0.0)

        """ Image processing """
        if self.image_available():
            images_gray = {}
            for cam in self.active_cameras:
                images_gray[cam] = input_data["Grayscale"][getattr(carla.SensorPosition, cam)]

            # Stereo VO
            if self.step >= ARM_RAISE_WAIT_FRAMES:
                if self.step == ARM_RAISE_WAIT_FRAMES:
                    self.frontend.initialize(images_gray)
                else:
                    images_gray["step"] = self.step
                    images_gray["imu_measurements"] = self.imu_measurements
                    images_gray["prev_pose"] = self.current_pose

                    data = self.frontend.process_frame(images_gray)
                    self.backend.update(data)
                    depth = data["depth"]

                    # Path planning
                    self.arcs = self.arc_planner.np_candidate_arcs
                    if USE_TEMPORAL:
                        control, path, waypoint_local = self.arc_planner.plan_arc(
                            self.step,
                            waypoint,
                            nav_pose,
                            data["rock_data"]["centers"],
                            data["rock_data"]["radii"],
                            self.current_velocity,
                            depth,
                        )
                    else:
                        control, path, waypoint_local = self.arc_planner.plan_arc(
                            waypoint, nav_pose, data["rock_data"]
                        )
                    if control is not None:
                        self.current_v, self.current_w = control
                        # Proportional feedback on v
                        v_norm = np.linalg.norm(self.current_velocity)
                        v_delta = params.KP_LINEAR * (params.TARGET_SPEED - v_norm)
                        self.current_v += np.clip(
                            v_delta, -params.MAX_LINEAR_DELTA, params.MAX_LINEAR_DELTA
                        )

                        if RERUN:
                            Rerun.log_2d_trajectory(frame_id=self.step, trajectory=path)
                            if len(data["rock_data"]["centers"]) > 0:
                                # TODO: crop rocks within certain bounds
                                Rerun.log_2d_obstacle_map(
                                    frame_id=self.step,
                                    centers=data["rock_data"]["centers"][:, :2],
                                    radii=data["rock_data"]["radii"],
                                )
                        if self.step % 100 == 0:
                            if USE_TEMPORAL:
                                combined_map = self.arc_planner.get_combined_rock_map(nav_pose)
                                self.arc_planner.plot_rocks(
                                    combined_map, self.arcs, path, self.step
                                )
                            else:
                                rock_data = (
                                    data["rock_data"]["centers"],
                                    data["rock_data"]["radii"],
                                )
                                self.arc_planner.plot_rocks(rock_data, self.arcs, path, self.step)

            if LOG_DATA:
                self.data_logger.log_images(self.step, input_data)
            if RERUN:
                Rerun.log_img(images_gray["FrontLeft"])

        """ Control """
        if self.step < ARM_RAISE_WAIT_FRAMES:  # Wait for arms to raise before moving
            carla_control = carla.VehicleVelocityControl(0.0, 0.0)
        # If agent is stuck, perform backup maneuver
        elif self.backup_counter > 0 or self.check_stuck():
            print("Agent is stuck.")
            carla_control = self.run_backup_maneuver()
        elif control is None:
            print("No safe paths found by planner!")
            carla_control = self.run_backup_maneuver()
        else:
            carla_control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        """ Data logging """
        if LOG_DATA:
            self.data_logger.log_data(self.step, carla_control, self.current_pose)

        """ Update state """
        slam_poses = self.backend.get_trajectory()
        self.current_pose = slam_poses[-1]
        self.current_velocity = self.frontend.current_velocity

        """ Rerun logging """
        if RERUN:
            self.gt_poses.append(ground_truth_pose)
            gt_trajectory = np.array([pose[:3, 3] for pose in self.gt_poses])
            slam_trajectory = get_positions_from_poses(slam_poses)
            position_error = slam_trajectory[-1] - ground_truth_pose[:3, 3]
            Rerun.log_3d_trajectory(
                self.step, gt_trajectory, trajectory_string="ground_truth", color=[20, 20, 20]
            )
            Rerun.log_3d_trajectory(
                self.step, slam_trajectory, trajectory_string="slam", color=[0, 50, 200]
            )
            Rerun.log_2d_seq_scalar("/trajectory_error/err_x", self.step, position_error[0])
            Rerun.log_2d_seq_scalar("/trajectory_error/err_y", self.step, position_error[1])
            Rerun.log_2d_seq_scalar("/trajectory_error/err_z", self.step, position_error[2])
            Rerun.log_2d_seq_scalar(
                "/trajectory_error/velocity", self.step, np.linalg.norm(self.current_velocity)
            )

        print("\n-----------------------------------------------")

        return carla_control

    def update_map(self):
        """Update the map with current backend state"""
        print("Updating map")
        g_map = self.get_geometric_map()
        map_array = g_map.get_map_array()
        semantic_points = self.backend.project_point_map()
        print("Number of semantic points: ", len(semantic_points.points))
        if LOG_DATA:
            semantic_points.save(f"output/{get_entry_point()}/default_run/semantic_points.npz")
        map_array = process_map(semantic_points, map_array)
        return map_array.copy()

    def finalize(self):
        print("Running finalize")
        self.update_map()

        if LOG_DATA:
            self.data_logger.save_log()
            slam_poses = np.array(self.backend.get_trajectory())
            np.save(f"output/{get_entry_point()}/default_run/slam_poses.npy", slam_poses)

            backend_state = self.backend.get_state()
            # with open(f"output/{get_entry_point()}/default_run/backend_state.", 'w') as file:
            #     json.dump(data, file, indent=4) # indent for readability
            np.savez_compressed(
                f"output/{get_entry_point()}/default_run/backend_state.npz",
                odometry=backend_state["odometry"],
                loop_closures=backend_state["loop_closures"],
                loop_closure_poses=backend_state["loop_closures_poses"],
            )
