#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Mapping agent

"""

import carla
import cv2 as cv
import numpy as np
from pynput import keyboard
import signal

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.util import transform_to_numpy
from lac.perception.segmentation import SemanticClasses
from lac.planning.waypoint_planner import WaypointPlanner
from lac.planning.arc_planner import ArcPlanner
from lac.control.steering import waypoint_steering
from lac.slam.semantic_feature_tracker import SemanticFeatureTracker
from lac.slam.frontend import Frontend
from lac.slam.backend import Backend
from lac.mapping.mapper import process_map
from lac.utils.data_logger import DataLogger
from lac.utils.rerun_interface import Rerun
from lac.util import get_positions_from_poses
import lac.params as params

""" Agent parameters and settings """
USE_GROUND_TRUTH_NAV = True  # Whether to use ground truth pose for navigation
ARM_RAISE_WAIT_FRAMES = 100  # Number of frames to wait for the arms to raise

DISPLAY_IMAGES = True  # Whether to display the camera views
RERUN_PLOT_POINTS = False  # Whether to plot points in rerun
TELEOP = True  # Whether to use teleop control or autonomous control


def get_entry_point():
    return "SlamAgent"


class SlamAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys."""
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ Control variables """
        self.current_v = 0
        self.current_w = 0

        """ State variable for velocity reset stop """
        self.stop_reset_counter = 0

        """ Controller variables """
        self.steer_delta = 0.0

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.step = 0

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
        self.active_cameras = [cam for cam, config in self.cameras.items() if config["active"]]

        """ Planner """
        self.initial_pose = transform_to_numpy(self.get_initial_position())
        self.lander_pose = self.initial_pose @ transform_to_numpy(
            self.get_initial_lander_position()
        )
        self.planner = WaypointPlanner(
            self.initial_pose, spiral_min=3.0, spiral_max=7.0, spiral_step=0.25, repeat=0
        )
        self.arc_planner = ArcPlanner()

        """ SLAM """
        feature_tracker = SemanticFeatureTracker(self.cameras)
        self.frontend = Frontend(feature_tracker)
        self.backend = Backend(self.initial_pose, feature_tracker)

        """ Data logging """
        agent_name = get_entry_point()
        self.data_logger = DataLogger(self, agent_name, self.cameras)

        Rerun.init_vo()
        self.gt_poses = [self.initial_pose]
        self.svo_poses = []
        self.current_pose = self.initial_pose

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signal_received, frame):
        print("\nCtrl+C detected! Exiting mission")
        self.mission_complete()

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
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

    def initialize(self):
        # Move the arms out of the way
        self.set_front_arm_angle(params.ARM_ANGLE_STATIC_RAD)
        self.set_back_arm_angle(params.ARM_ANGLE_STATIC_RAD)

    def image_available(self):
        return self.step % 2 == 0  # Image data is available every other step

    def run_step(self, input_data):
        if self.step == 0:
            self.initialize()
        self.step += 1  # Starts at 0 at init, equal to 1 on the first run_step call
        print("\nStep: ", self.step)

        ground_truth_pose = transform_to_numpy(self.get_transform())
        self.gt_poses.append(ground_truth_pose)

        if USE_GROUND_TRUTH_NAV:
            nav_pose = ground_truth_pose
        else:
            nav_pose = self.current_pose

        """ Waypoint navigation """
        waypoint, _ = self.planner.get_waypoint(nav_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)
        nominal_steering = waypoint_steering(waypoint, nav_pose)

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

            self.data_logger.log_images(self.step, input_data)
            Rerun.log_img(images_gray["FrontLeft"])

            if len(self.backend.point_map) > 0 and RERUN_PLOT_POINTS:
                semantic_points = self.backend.project_point_map()
                Rerun.log_3d_semantic_points(semantic_points)

        """ Control """
        if self.step < ARM_RAISE_WAIT_FRAMES:
            control = carla.VehicleVelocityControl(0.0, 0.0)
        elif TELEOP:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        else:
            control = carla.VehicleVelocityControl(params.TARGET_SPEED, nominal_steering)

        """ Data logging """
        self.data_logger.log_data(self.step, control)

        """ Rerun logging """
        gt_trajectory = np.array([pose[:3, 3] for pose in self.gt_poses])
        slam_trajectory = get_positions_from_poses(self.backend.get_trajectory())
        position_error = slam_trajectory[-1] - ground_truth_pose[:3, 3]
        Rerun.log_3d_trajectory(
            self.step, gt_trajectory, trajectory_string="ground_truth", color=[20, 20, 20]
        )
        Rerun.log_3d_trajectory(
            self.step, slam_trajectory, trajectory_string="slam", color=[0, 50, 200]
        )
        Rerun.log_2d_seq_scalar("trajectory_error/err_x", self.step, position_error[0])
        Rerun.log_2d_seq_scalar("trajectory_error/err_y", self.step, position_error[1])
        Rerun.log_2d_seq_scalar("trajectory_error/err_z", self.step, position_error[2])

        print("\n-----------------------------------------------")

        return control

    def finalize(self):
        print("Running finalize")

        self.data_logger.save_log()

        # Set map
        g_map = self.get_geometric_map()
        map_array = g_map.get_map_array()
        semantic_points = self.backend.project_point_map()
        map_array = process_map(semantic_points, map_array)

        """In the finalize method, we should clear up anything we've previously initialized that might be taking up memory or resources.
        In this case, we should close the OpenCV window."""
        cv.destroyAllWindows()

    def on_press(self, key):
        """This is the callback executed when a key is pressed. If the key pressed is either the up or down arrow, this method will add
        or subtract target linear velocity. If the key pressed is either the left or right arrow, this method will set a target angular
        velocity of 0.6 radians per second."""

        if key == keyboard.Key.up:
            self.current_v += 0.1
            self.current_v = np.clip(self.current_v, 0, 0.3)
        if key == keyboard.Key.down:
            self.current_v -= 0.1
            self.current_v = np.clip(self.current_v, -0.3, 0)
        if key == keyboard.Key.left:
            self.current_w = 0.6
        if key == keyboard.Key.right:
            self.current_w = -0.6

    def on_release(self, key):
        """This method sets the angular or linear velocity to zero when the arrow key is released. Stopping the robot."""

        if key == keyboard.Key.up:
            self.current_v = 0
        if key == keyboard.Key.down:
            self.current_v = 0
        if key == keyboard.Key.left:
            self.current_w = 0
        if key == keyboard.Key.right:
            self.current_w = 0

        """ Press escape to end the mission. """
        if key == keyboard.Key.esc:
            self.mission_complete()
            cv.destroyAllWindows()
