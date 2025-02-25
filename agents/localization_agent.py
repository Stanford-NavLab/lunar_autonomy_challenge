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
    wrap_angle,
    mask_centroid,
    gen_square_spiral,
)
from lac.perception.vision import FiducialLocalizer
from lac.utils.visualization import overlay_tag_detections
from lac.utils.frames import apply_transform
import lac.params as params


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

        """ Controller params """
        self.KP_STEER = 0.3
        self.KP_LINEAR = 0.1
        self.TARGET_SPEED = 0.2  # m/s
        self.steer_delta = 0.0

        """ Perception modules """
        self.frame = 0
        self.image_process_rate = 1  # Hz

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.step = 0

        """ Waypoints """
        self.waypoints = gen_square_spiral(max_val=4.5, min_val=2.0, step=0.5)
        self.waypoint_idx = 0
        self.waypoint_threshold = 1.0  # meters

        """ Localization """
        self.fid_localizer = FiducialLocalizer()
        self.measurement_history = []
        self.odometry_history = []
        # TODO: init Filter class

        """ Data logging """
        self.run_name = "localization_agent"
        self.log_file = "output/" + self.run_name + "/data_log.json"
        self.rock_points_file = "output/" + self.run_name + "/rock_points.npy"
        initial_rover_pose = transform_to_numpy(self.get_initial_position())
        lander_pose_rover = transform_to_numpy(self.get_initial_lander_position())
        lander_pose_world = initial_rover_pose @ lander_pose_rover
        self.out = {
            "initial_pose": initial_rover_pose.tolist(),
            "lander_pose_rover": lander_pose_rover.tolist(),
            "lander_pose_world": lander_pose_world.tolist(),
        }
        self.cameras = {
            "FrontLeft": {"active": True, "light": 1.0, "semantic": False},
            "FrontRight": {"active": False, "light": 0.0, "semantic": False},
            "BackLeft": {"active": False, "light": 0.0, "semantic": False},
            "BackRight": {"active": False, "light": 0.0, "semantic": False},
            "Left": {"active": False, "light": 0.0, "semantic": False},
            "Right": {"active": True, "light": 1.0, "semantic": False},
            "Front": {"active": False, "light": 0.0, "semantic": False},
            "Back": {"active": False, "light": 0.0, "semantic": False},
        }
        self.out["cameras_config"] = self.cameras
        self.out["use_fiducials"] = self.use_fiducials()
        self.frames = []

        if os.path.exists("output/" + self.run_name):
            shutil.rmtree("output/" + self.run_name)
        for cam, config in self.cameras.items():
            if config["active"]:
                os.makedirs("output/" + self.run_name + "/" + cam)
                if config["semantic"]:
                    os.makedirs("output/" + self.run_name + "/" + cam + "_semantic")

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
        self.set_front_arm_angle(radians(params.ARM_ANGLE_STATIC_DEG))
        self.set_back_arm_angle(radians(params.ARM_ANGLE_STATIC_DEG))

    def run_step(self, input_data):
        if self.step == 0:
            self.initialize()

        current_pose = transform_to_numpy(self.get_transform())
        imu_data = self.get_imu_data()
        linear_speed = self.get_linear_speed()
        angular_speed = self.get_angular_speed()

        # TODO: filter predict with IMU

        # Perception
        FL_gray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        if FL_gray is not None:
            R_gray = input_data["Grayscale"][carla.SensorPosition.Right]
            tag_detections = self.fid_localizer.detect(FL_gray)

            # TODO:
            # Detection fiducials
            # EKF update

            if self.DISPLAY:
                # FL_rgb_PIL = Image.fromarray(FL_gray).convert("RGB")
                overlay = overlay_tag_detections(FL_gray, tag_detections)
                cv.imshow("Tag detections", overlay)
                cv.waitKey(1)

            R_img = input_data["Grayscale"][carla.SensorPosition.Right]
            cv.imwrite(
                "output/" + self.run_name + "/Right/" + str(self.step) + ".png",
                R_img,
            )

        # TODO: EKF smoothing at some rate

        if self.TELEOP:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        # Data logging
        log_entry = {
            "step": self.step,
            "timestamp": time.time(),
            "mission_time": self.get_mission_time(),
            "current_power": self.get_current_power(),
            "pose": current_pose.tolist(),
            "imu": imu_data.tolist(),
            "control": {"v": self.current_v, "w": self.current_w},
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
        }
        self.frames.append(log_entry)
        self.step += 1
        if self.step % 100 == 0:
            with open(self.log_file, "w") as f:
                self.out["frames"] = self.frames
                json.dump(self.out, f, indent=4)

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
