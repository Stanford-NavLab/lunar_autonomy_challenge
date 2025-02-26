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

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.utils.data_logger import DataLogger
import lac.params as params

UPDATE_LOG_RATE = 100  # Update log file every 100 steps

# Attributes for teleop sensitivity and max speed
MAX_SPEED = 0.3
SPEED_INCREMENT = 0.1
TURN_RATE = 0.6


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

        # Camera config
        self.cameras = params.CAMERA_CONFIG_INIT
        self.cameras["FrontLeft"] = {
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

        self.data_logger = DataLogger(self, "data_collection", self.cameras)

    def use_fiducials(self):
        return True

    def sensors(self):
        """In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048)
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light."""
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

        if self.image_available():
            self.data_logger.log_images(self.step, input_data)
            FL_gray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
            cv.imshow("Front left", FL_gray)
            # R_img = input_data["Grayscale"][carla.SensorPosition.Right]
            # cv.imshow("Right", R_img)
            cv.waitKey(1)

        control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        self.data_logger.log_data(self.step, control)
        if self.step % UPDATE_LOG_RATE == 0:
            self.data_logger.save_log()

        return control

    def finalize(self):
        print("Running finalize")
        self.data_logger.save_log()

        """In the finalize method, we should clear up anything we've previously initialized that might be taking up memory or resources.
        In this case, we should close the OpenCV window."""
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
