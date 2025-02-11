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

from lac.util import (
    pose_to_rpy_pos,
    transform_to_numpy,
    wrap_angle,
    color_mask,
    draw_steering_arc,
    mask_centroid,
)


def get_entry_point():
    return "MappingAgent"


class MappingAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys."""
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        # For teleop
        self.current_v = 0
        self.current_w = 0
        self.max_speed = 0.2
        self.speed_increment = 0.05
        self.turn_rate = 0.3

        self.wheel_rig = np.array(
            [
                [0.222, 0.203, -0.134],
                [0.222, -0.203, -0.134],
                [-0.222, 0.203, -0.134],
                [-0.222, 0.203, -0.134],
            ]
        )
        self.wheel_rig_coords = np.concatenate((self.wheel_rig.T, np.ones((1, 4))), axis=0)

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.frame = 0
        self.rate = 1  # Sub-sample rate. Max rate is 10Hz

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return False

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
                "use_semantic": False,
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": False,
                "light_intensity": 1.0,
                "width": "1280",
                "height": "720",
                "use_semantic": False,
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

        # Cheat with perfect localization
        current_pose = transform_to_numpy(self.get_transform())

        # Wheel contact mapping
        wheel_contact_points = current_pose @ self.wheel_rig_coords
        wheel_contact_points = wheel_contact_points[:3, :].T

        g_map = self.get_geometric_map()
        for point in wheel_contact_points:
            current_height = g_map.get_height(point[0], point[1])
            if (current_height == -np.inf) or (current_height > point[2]):
                g_map.set_height(point[0], point[1], point[2])

        # Show camera POV
        FL_gray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        if FL_gray is not None:
            cv.imshow("Left camera view", FL_gray)
            cv.waitKey(1)

        control = carla.VehicleVelocityControl(self.current_v, self.current_w)

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
            self.current_v += self.speed_increment
            self.current_v = np.clip(self.current_v, 0, self.max_speed)
        if key == keyboard.Key.down:
            self.current_v -= self.speed_increment
            self.current_v = np.clip(self.current_v, -self.max_speed, 0)
        if key == keyboard.Key.left:
            self.current_w = self.turn_rate
        if key == keyboard.Key.right:
            self.current_w = -self.turn_rate

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
