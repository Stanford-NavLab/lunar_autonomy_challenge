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

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.util import transform_to_numpy
from lac.perception.segmentation import UnetSegmentation
from lac.slam.semantic_feature_tracker import SemanticFeatureTracker
from lac.slam.frontend import Frontend
from lac.slam.backend import Backend
from lac.planning.arc_planner import ArcPlanner
from lac.planning.waypoint_planner import WaypointPlanner
from lac.mapping.mapper import process_map
from lac.utils.data_logger import DataLogger
from lac.utils.rerun_interface import Rerun
import lac.params as params


""" Agent parameters and settings """
EVAL = False  # Whether running in evaluation mode (disable ground truth)
USE_FIDUCIALS = False
BACK_CAMERAS = True

EARLY_STOP_STEP = 0  # Number of steps before stopping the mission (0 for no early stop)
USE_GROUND_TRUTH_NAV = False  # Whether to use ground truth pose for navigation
ARM_RAISE_WAIT_FRAMES = 80  # Number of frames to wait for the arms to raise

DISPLAY_IMAGES = True  # Whether to display the camera views
LOG_DATA = True  # Whether to log data

if EVAL:
    USE_GROUND_TRUTH_NAV = False
    DISPLAY_IMAGES = False
    LOG_DATA = False


def get_entry_point():
    return "NavAgent"


class NavAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Controller variables"""
        self.steer_delta = 0.0

        """ Perception modules """
        self.segmentation = UnetSegmentation()

        """ Initialize a counter to keep track of the number of simulation steps. """
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
        self.planner = WaypointPlanner(
            self.initial_pose, spiral_min=2.5, spiral_max=2.5, spiral_step=1.0
        )
        self.arc_planner = ArcPlanner()

        """ State variables """
        self.current_pose = self.initial_pose
        self.current_velocity = np.zeros(3)

        """ SLAM """
        feature_tracker = SemanticFeatureTracker(self.cameras)
        self.frontend = Frontend(feature_tracker)
        self.backend = Backend(self.initial_pose, feature_tracker)

        """ Data logging """
        if LOG_DATA:
            agent_name = get_entry_point()
            self.data_logger = DataLogger(self, agent_name, self.cameras)

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signal_received, frame):
        print("\nCtrl+C detected! Exiting mission")
        self.mission_complete()

    def initialize(self):
        # Move the arms out of the way
        self.set_front_arm_angle(params.ARM_ANGLE_STATIC_RAD)
        self.set_back_arm_angle(params.ARM_ANGLE_STATIC_RAD)

    def image_available(self):
        return self.step % 2 == 0  # Image data is available every other step

    def use_fiducials(self):
        return USE_FIDUCIALS

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
        frame_rate = params.FRAME_RATE
        self.backup_counter += 1
        if self.backup_counter <= frame_rate * 1.5:  # Go backwards for 3 seconds
            control = carla.VehicleVelocityControl(-0.2, 0.0)
        elif (
            self.backup_counter <= frame_rate * 3
        ):  # Rotate 90 deg/s for 1.5 seconds (overcorrecting because it isn't rotating in 1 second)
            control = carla.VehicleVelocityControl(0.0, np.pi / 4)
        elif self.backup_counter <= frame_rate * 9:
            # Go forward for 6 seconds
            control = carla.VehicleVelocityControl(0.2, 0.0)
        else:
            self.backup_counter = 0
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        return control

    def check_stuck(self):
        # Agent is stuck if the velocity is less than 0.1 m/s
        if self.step < ARM_RAISE_WAIT_FRAMES + 10:
            return False
        is_stuck = np.linalg.norm(self.current_velocity) < 0.5 * params.TARGET_SPEED
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

        if self.stuck_timer > 0:
            self.stuck_timer += 1

        if not EVAL:
            ground_truth_pose = transform_to_numpy(self.get_transform())

        if USE_GROUND_TRUTH_NAV:
            nav_pose = ground_truth_pose
        else:
            nav_pose = self.current_pose

        """ Waypoint navigation """
        waypoint, advanced = self.planner.get_waypoint(nav_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        """ Image processing """
        if self.image_available():
            images_gray = {}
            for cam in self.active_cameras:
                images_gray[cam] = input_data["Grayscale"][getattr(carla.SensorPosition, cam)]

            # Stereo VO
            if self.step >= ARM_RAISE_WAIT_FRAMES:
                if self.step == ARM_RAISE_WAIT_FRAMES:
                    self.frontend.initialize(images_gray["FrontLeft"], images_gray["FrontRight"])
                else:
                    images_gray["step"] = self.step
                    images_gray["imu"] = self.get_imu_data()
                    data = self.frontend.process_frame(images_gray)
                    self.backend.update(data)

                # Path planning
                control, path, waypoint_local = self.arc_planner.plan_arc(
                    waypoint, nav_pose, data["rock_data"]
                )
                if control is not None:
                    self.current_v, self.current_w = control
                else:
                    print("No safe paths found!")

            if LOG_DATA:
                self.data_logger.log_images(self.step, input_data)

        """ Control """
        if self.step < ARM_RAISE_WAIT_FRAMES:  # Wait for arms to raise before moving
            control = carla.VehicleVelocityControl(0.0, 0.0)
        # If agent is stuck, perform backup maneuver
        elif self.backup_counter > 0 or self.check_stuck():
            print("Agent is stuck.")
            control = self.run_backup_maneuver()
        else:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        """ Data logging """
        if LOG_DATA:
            self.data_logger.log_data(self.step, control, self.current_pose)

        print("\n-----------------------------------------------")

        return control

    def update_map(self):
        """Update the map with current backend state"""
        print("Updating map")
        g_map = self.get_geometric_map()
        map_array = g_map.get_map_array()
        semantic_points = self.backend.project_point_map()
        map_array = process_map(semantic_points, map_array)

    def finalize(self):
        print("Running finalize")
        self.update_map()

        if LOG_DATA:
            self.data_logger.save_log()

        if DISPLAY_IMAGES:
            cv.destroyAllWindows()
