#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Data collection agent

"""

import numpy as np
import carla
import cv2 as cv
import random
import os
import shutil
import json
import time
from math import radians
from pynput import keyboard
import plotly.graph_objects as go

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.util import transform_to_numpy
from lac.utils.data_logger import DataLogger
import lac.params as params

UPDATE_LOG_RATE = 100  # Update log file every 100 steps


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

        """ Attributes for teleop sensitivity and max speed. """
        self.max_speed = 0.3
        self.speed_increment = 0.1
        self.turn_rate = 0.6

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.step = 0
        self.rate = 1  # Sub-sample rate. Max rate is 10Hz

        # self.run_name = "data_collection"
        # self.log_file = "output/" + self.run_name + "/data_log.json"

        # # Initial rover pose in world frame
        # initial_rover_pose = transform_to_numpy(self.get_initial_position())
        # # Lander pose in rover frame at initialization
        # lander_pose_rover = transform_to_numpy(self.get_initial_lander_position())
        # # Lander pose in world frame (constant)
        # lander_pose_world = initial_rover_pose @ lander_pose_rover
        # print("Initial lander pose in world frame: ", lander_pose_world[:3, 3])
        # self.out = {
        #     "initial_pose": initial_rover_pose.tolist(),
        #     "lander_pose_rover": lander_pose_rover.tolist(),
        #     "lander_pose_world": lander_pose_world.tolist(),
        # }

        # self.frames = []
        self.log_images = True
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
        # self.out["cameras_config"] = self.cameras
        # self.out["use_fiducials"] = self.use_fiducials()

        self.data_logger = DataLogger(self, "data_collection", self.cameras)

        # if os.path.exists("output/" + self.run_name):
        #     shutil.rmtree("output/" + self.run_name)
        # for cam, config in self.cameras.items():
        #     if config["active"]:
        #         os.makedirs("output/" + self.run_name + "/" + cam)
        #         if config["semantic"]:
        #             os.makedirs("output/" + self.run_name + "/" + cam + "_semantic")

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
        # NOTE: alternatively manually check input_data
        return self.step % 2 == 0  # Image data is available every other step

    def initialize(self):
        # Move the arms out of the way
        self.set_front_arm_angle(radians(params.ARM_ANGLE_STATIC_DEG))
        self.set_back_arm_angle(radians(params.ARM_ANGLE_STATIC_DEG))

    def run_step(self, input_data):
        if self.step == 0:
            self.initialize()

        FL_img = input_data["Grayscale"][carla.SensorPosition.FrontLeft]

        self.step += 1  # Starts at 0 at init, equal to 1 on the first run_step call

        # current_pose = transform_to_numpy(self.get_transform())
        # imu_data = self.get_imu_data()
        # linear_speed = self.get_linear_speed()
        # angular_speed = self.get_angular_speed()

        """ We need to check that the sensor data is not None before we do anything with it. The data for each camera will be 
        None for every other simulation step, since the cameras operate at 10Hz while the simulator operates at 20Hz. """
        if self.image_available():
            self.data_logger.log_images(self.step, input_data)
            FL_img = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
            cv.imshow("Front left", FL_img)
            # R_img = input_data["Grayscale"][carla.SensorPosition.Right]
            # cv.imshow("Right", R_img)
            cv.waitKey(1)

            # if self.frame % self.rate == 0:
            #     if self.log_images:
            #         for cam, config in self.cameras.items():
            #             if config["active"]:
            #                 img = input_data["Grayscale"][getattr(carla.SensorPosition, cam)]
            #                 cv.imwrite(
            #                     "output/" + self.run_name + f"/{cam}/" + str(self.frame) + ".png",
            #                     img,
            #                 )
            #                 if config["semantic"]:
            #                     semantic_img = input_data["Semantic"][
            #                         getattr(carla.SensorPosition, cam)
            #                     ]
            #                     cv.imwrite(
            #                         "output/"
            #                         + self.run_name
            #                         + f"/{cam}_semantic/"
            #                         + str(self.frame)
            #                         + ".png",
            #                         semantic_img,
            #                     )

        # log_entry = {
        #     "frame": self.frame,
        #     "timestamp": time.time(),
        #     "mission_time": self.get_mission_time(),
        #     "current_power": self.get_current_power(),
        #     "pose": current_pose.tolist(),
        #     "imu": imu_data.tolist(),
        #     "control": {"v": self.current_v, "w": self.current_w},
        #     "linear_speed": linear_speed,
        #     "angular_speed": angular_speed,
        # }
        # self.frames.append(log_entry)
        # self.frame += 1

        control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        self.data_logger.log_data(self.step, (self.current_v, self.current_w))

        if self.step % UPDATE_LOG_RATE == 0:
            # with open(self.log_file, "w") as f:
            #     self.out["frames"] = self.frames
            #     json.dump(self.out, f, indent=4)
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
