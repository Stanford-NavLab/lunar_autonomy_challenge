#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Data collection agent

"""

import json
import os
import random
import time
from math import radians

import carla
import cv2 as cv
import numpy as np
from pynput import keyboard
from PIL import Image

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.util import pose_to_rpy_pos, transform_to_numpy, wrap_angle, color_mask
from lac.perception.segmentation import Segmentation
from lac.perception.depth import DepthAnything


def get_entry_point():
    return "NavAgent"


class NavAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys."""
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        # For teleop
        self.current_v = 0
        self.current_w = 0
        self.TELEOP = True

        """ Controller params """
        self.KP_STEER = 0.3
        self.KP_LINEAR = 0.1
        self.TARGET_SPEED = 0.2  # m/s

        """ Perception modules """
        self.segmentation = Segmentation()
        self.depth = DepthAnything()

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.frame = 0
        self.rate = 1  # Sub-sample rate. Max rate is 10Hz

        self.waypoints = np.array(
            [[-9.0, -9.0], [-9.0, 9.0], [9.0, 9.0], [9.0, -9.0], [-9.0, -9.0]]
        )
        self.waypoint_idx = 0

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return True

    def sensors(self):
        """In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048)
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light."""

        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "1280",
                "height": "720",
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1280",
                "height": "720",
                "use_semantic": True,
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1280",
                "height": "720",
                "use_semantic": True,
            },
            carla.SensorPosition.Left: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "1280",
                "height": "720",
            },
            carla.SensorPosition.Right: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "1280",
                "height": "720",
            },
            carla.SensorPosition.BackLeft: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "1280",
                "height": "720",
            },
            carla.SensorPosition.BackRight: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "1280",
                "height": "720",
            },
            carla.SensorPosition.Back: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "1280",
                "height": "720",
            },
        }
        return sensors

    def run_step(self, input_data):
        # Move the arms out of the way
        if self.frame == 0:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        # Show camera POV
        FL_gray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        FR_gray = input_data["Grayscale"][carla.SensorPosition.FrontRight]
        if FL_gray is not None:
            # Run segmentation
            results, mask = self.segmentation.segment_rocks(Image.fromarray(FL_gray).convert("RGB"))

            FL_rgb = cv.cvtColor(FL_gray, cv.COLOR_GRAY2BGR)

            mask_colored = color_mask(mask, (0, 0, 1)).astype(FL_rgb.dtype)
            print("Mask colored shape: ", mask_colored.shape)
            print("FL_rgb shape: ", FL_rgb.shape)
            overlay = cv.addWeighted(FL_rgb, 1.0, mask_colored, beta=0.5, gamma=0)
            print("ran segmentation")

            # cv.imshow("Left camera view", FL_rgb)
            cv.imshow("Rock segmentation", overlay)
            # cv.imshow("Right camera view", FR_rgb)
            # cv.imshow("Segmentation mask", mask * np.array([1, 0, 0]))
            cv.waitKey(1)

        current_pose = transform_to_numpy(self.get_transform())
        rpy, pos = pose_to_rpy_pos(current_pose)
        heading = rpy[2]

        # Navigate to the next waypoint
        waypoint = self.waypoints[self.waypoint_idx]

        angle = np.arctan2(waypoint[1] - pos[1], waypoint[0] - pos[0])
        angle_diff = wrap_angle(angle - heading)
        steering = np.clip(self.KP_STEER * angle_diff, -1.0, 1.0)

        if self.TELEOP:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        else:
            control = carla.VehicleVelocityControl(self.TARGET_SPEED, steering)

        if self.frame >= 5000:
            self.mission_complete()

        return control

    def finalize(self):
        print("Running finalize")

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
