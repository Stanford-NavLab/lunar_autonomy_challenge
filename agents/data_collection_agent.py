#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Data collection agent

"""

import numpy as np
import carla
import cv2 as cv
from pynput import keyboard
import signal

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.planning.waypoint_planner import Planner
from lac.control.controller import waypoint_steering
from lac.utils.data_logger import DataLogger
from lac.util import transform_to_numpy
import lac.params as params

# Attributes for teleop sensitivity and max speed
MAX_SPEED = 0.2
SPEED_INCREMENT = 0.05
TURN_RATE = 0.3

MODE = "dynamics"  # {"teleop", "waypoint", "dynamics"}
DISPLAY_IMAGES = False  # Set to False to disable image display


def get_entry_point():
    return "DataCollectionAgent"


class DataCollectionAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys."""
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ Add some attributes to store values for the target linear and angular velocity. """
        self.current_v = 0
        self.current_w = 0

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.step = 0

        """ Planner """
        initial_pose = transform_to_numpy(self.get_initial_position())
        self.lander_pose = initial_pose @ transform_to_numpy(self.get_initial_lander_position())
        self.planner = Planner(initial_pose, spiral_min=3.5, spiral_max=13.5, spiral_step=2.0)

        # Camera config
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
        self.cameras["Front"] = {
            "active": True,
            "light": 1.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        self.cameras["BackLeft"] = {
            "active": False,
            "light": 0.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        self.cameras["BackRight"] = {
            "active": False,
            "light": 0.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        self.cameras["Back"] = {
            "active": False,
            "light": 0.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        self.cameras["Left"] = {
            "active": True,
            "light": 1.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        self.cameras["Right"] = {
            "active": True,
            "light": 1.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        agent_name = get_entry_point()

        # For dynamics data collection
        self.v = 0.2
        self.w = 0.2
        log_file = f"results/dynamics/v{self.v}_w{self.w}_scaled2.json"
        self.data_logger = DataLogger(self, agent_name, self.cameras, log_file=log_file)

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signal_received, frame):
        print("\nCtrl+C detected! Exiting mission")
        self.mission_complete()

    def use_fiducials(self):
        return False

    def sensors(self):
        """In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048)
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light.
        """
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

    def image_available(self):
        return self.step % 2 == 0  # Image data is available every other step

    def initialize(self):
        # Move the arms out of the way
        self.set_front_arm_angle(params.ARM_ANGLE_STATIC_RAD)
        self.set_back_arm_angle(params.ARM_ANGLE_STATIC_RAD)

    def run_step(self, input_data):
        if self.step == 0:
            self.initialize()
        self.step += 1  # Starts at 0 at init, equal to 1 on the first run_step call
        print("\nStep: ", self.step)

        if self.image_available():
            self.data_logger.log_images(self.step, input_data)
            if DISPLAY_IMAGES:
                FL_gray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
                cv.imshow("Front left", FL_gray)
                cv.waitKey(1)

        if MODE == "teleop":
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        elif MODE == "waypoint":
            ground_truth_pose = transform_to_numpy(self.get_transform())
            waypoint, _ = self.planner.get_waypoint(ground_truth_pose, print_progress=True)
            if waypoint is None:
                self.mission_complete()
                return carla.VehicleVelocityControl(0.0, 0.0)
            nominal_steering = waypoint_steering(waypoint, ground_truth_pose)

            if self.step < 100:  # Wait for arms to raise before moving
                control = carla.VehicleVelocityControl(0.0, 0.0)
            else:
                control = carla.VehicleVelocityControl(0.2, nominal_steering)
        elif MODE == "dynamics":
            if self.step >= 100:
                control = carla.VehicleVelocityControl(self.v, 2 * self.w)
            else:
                control = carla.VehicleVelocityControl(0.0, 0.0)

            if self.step >= 400:
                self.mission_complete()

        self.data_logger.log_data(self.step, control)

        return control

    def finalize(self):
        print("Running finalize")
        self.data_logger.save_log()

        """In the finalize method, we should clear up anything we've previously initialized that might be taking up memory or resources.
        In this case, we should close the OpenCV window."""
        if DISPLAY_IMAGES:
            cv.destroyAllWindows()

    def on_press(self, key):
        """This is the callback executed when a key is pressed. If the key pressed is either the up or down arrow, this method will add
        or subtract target linear velocity. If the key pressed is either the left or right arrow, this method will set a target angular
        velocity of 0.6 radians per second."""

        if key == keyboard.Key.up:
            self.current_v += SPEED_INCREMENT
            self.current_v = np.clip(self.current_v, 0, MAX_SPEED)
        if key == keyboard.Key.down:
            self.current_v -= SPEED_INCREMENT
            self.current_v = np.clip(self.current_v, -MAX_SPEED, 0)
        if key == keyboard.Key.left:
            self.current_w = TURN_RATE
        if key == keyboard.Key.right:
            self.current_w = -TURN_RATE

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
