#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Data collection agent

"""

import json
import os
import shutil
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
    transform_to_rpy_pos,
    wrap_angle,
    mask_centroid,
    gen_square_spiral,
)
from lac.perception.vision import FiducialLocalizer
from lac.localization.ekf import EKF, get_pose_measurement_tag, create_Q
from lac.localization.imu_dynamics import propagate_state
from lac.utils.visualization import overlay_tag_detections
from lac.utils.frames import apply_transform
from lac.utils.data_logger import DataLogger
import lac.params as params

DISPLAY_IMAGES = True  # Whether to display the camera views
TELEOP = False  # Whether to use teleop control or autonomous control
UPDATE_LOG_RATE = 100  # Update log file every 100 steps


def get_entry_point():
    return "LocalizationAgent"


class LocalizationAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys."""
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ For teleop """
        self.current_v = 0
        self.current_w = 0
        self.TELEOP = True

        self.IMG_WIDTH = 1280
        self.IMG_HEIGHT = 720
        self.DISPLAY = True
        self.max_steer = 1.0  # rad/s
        self.max_steer_delta = 0.6  # rad/s

        """ Perception modules """
        self.frame = 0
        self.image_process_rate = 1  # Hz

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.step = 0

        """ Waypoints """
        self.waypoints = gen_square_spiral(max_val=4.5, min_val=2.0, step=0.5)
        self.waypoint_idx = 0
        self.waypoint_threshold = 1.0  # meters

        """ Data logging """
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
        self.active_cameras = ["FrontLeft", "Right"]

        self.data_logger = DataLogger(self, "localization_agent", self.cameras)

        """ Localization """
        self.fid_localizer = FiducialLocalizer(self.cameras)
        self.measurement_history = []
        self.odometry_history = []
        # TODO: init Filter class
        init_rpy, init_pos = transform_to_rpy_pos(self.get_initial_transform())
        v0 = np.zeros(3)
        init_state = np.hstack((init_pos, v0, init_rpy)).T

        init_r = 0.001
        init_v = 0.01
        init_angle = 0.001
        P0 = np.diag(
            np.hstack(
                (
                    np.ones(3) * init_r * init_r,
                    np.ones(3) * init_v * init_v,
                    np.ones(3) * init_angle * init_angle,
                )
            )
        )
        self.Q_EKF = create_Q(params.DT, 0.03, 0.00005)
        self.ekf = EKF(init_state, P0, store=True)

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return True

    def sensors(self):
        """In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048)
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light."""

        sensors = {}
        for cam, config in self.cameras.items():
            sensors[getattr(carla.SensorPosition, cam)] = {
                "camera_active": config["active"],
                "light_intensity": config["light"],
                "width": "1280",
                "height": "720",
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

        current_pose = transform_to_numpy(self.get_transform())
        imu_data = self.get_imu_data()
        linear_speed = self.get_linear_speed()
        angular_speed = self.get_angular_speed()

        # TODO: filter predict with IMU
        a_k = imu_data[:3]
        omega_k = imu_data[3:]

        def dyn_func(x):
            return propagate_state(x, a_k, omega_k, params.DT, with_stm=True, use_numdiff=False)

        self.ekf.predict(self.step, dyn_func, self.Q_EKF)

        # Perception
        if self.image_available():
            FL_gray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
            R_gray = input_data["Grayscale"][carla.SensorPosition.Right]
            tag_detections = self.fid_localizer.detect(FL_gray)

            # TODO:
            # Detection fiducials
            fid_measurements = self.fid_localizer.estimate_rover_pose(
                FL_gray, "FrontLeft", current_pose
            )
            print("Fiducial measurements: ", fid_measurements)
            # EKF update

            if self.DISPLAY:
                # FL_rgb_PIL = Image.fromarray(FL_gray).convert("RGB")
                overlay = overlay_tag_detections(FL_gray, tag_detections)
                cv.imshow("Tag detections", overlay)
                cv.waitKey(1)

            self.data_logger.log_images(self.step, input_data)

        # TODO: EKF smoothing at some rate

        if self.TELEOP:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        # Data logging
        self.data_logger.log_data(self.step, control)
        if self.step % UPDATE_LOG_RATE == 0:
            self.data_logger.save_log()

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
