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
import json
import time
from math import radians
from pynput import keyboard
import plotly.graph_objects as go

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.util import transform_to_numpy, to_blender_convention


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
        self.frame = 0
        self.rate = 1  # Sub-sample rate. Max rate is 10Hz

        self.run_name = "data_collection"

        self.log_file = "output/" + self.run_name + "/data_log.json"
        self.out = {
            "initial_pose": transform_to_numpy(self.get_initial_position()).tolist(),
            "lander_pose": transform_to_numpy(self.get_initial_lander_position()).tolist(),
        }
        # TODO: add camera and lights config

        self.frames = []

        if not os.path.exists("output/" + self.run_name):
            os.makedirs("output/" + self.run_name + "/front_left")
            os.makedirs("output/" + self.run_name + "/front_right")
            os.makedirs("output/" + self.run_name + "/front_left_semantic")
            os.makedirs("output/" + self.run_name + "/front_right_semantic")

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

        FL_img = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        FR_img = input_data["Grayscale"][carla.SensorPosition.FrontRight]
        FL_sem = input_data["Semantic"][carla.SensorPosition.FrontLeft]
        FR_sem = input_data["Semantic"][carla.SensorPosition.FrontRight]

        current_pose = transform_to_numpy(self.get_transform())
        # camera_to_robot_pose = transform_to_numpy(
        #     self.get_camera_position(carla.SensorPosition.FrontLeft)
        # )
        # camera_to_world_pose = current_pose @ camera_to_robot_pose
        imu_data = self.get_imu_data()

        """ We need to check that the sensor data is not None before we do anything with it. The data for each camera will be 
        None for every other simulation step, since the cameras operate at 10Hz while the simulator operates at 20Hz. """
        if FL_img is not None:
            cv.imshow("Left camera view", FL_img)
            cv.waitKey(1)

            if self.frame % self.rate == 0:
                cv.imwrite(
                    "output/" + self.run_name + "/front_left/" + str(self.frame) + ".png",
                    FL_img,
                )
                cv.imwrite(
                    "output/" + self.run_name + "/front_right/" + str(self.frame) + ".png",
                    FR_img,
                )
                cv.imwrite(
                    "output/" + self.run_name + "/front_left_semantic/" + str(self.frame) + ".png",
                    FL_sem,
                )
                cv.imwrite(
                    "output/" + self.run_name + "/front_right_semantic/" + str(self.frame) + ".png",
                    FR_sem,
                )
                # self.poses.append(to_blender_convention(camera_to_world_pose))
                log_entry = {
                    "frame": self.frame,
                    "timestamp": time.time(),
                    "mission_time": self.get_mission_time(),
                    "current_power": self.get_current_power(),
                    "pose": current_pose.tolist(),
                    "imu": imu_data.tolist(),
                    "control": {"v": self.current_v, "w": self.current_w},
                }
                self.frames.append(log_entry)

            self.frame += 1

        control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        if self.frame % 100 == 0:
            # fig = go.Figure(data=pose_traces(self.poses))
            # fig.write_html("output/" + self.run_name + "/poses.html")
            with open(self.log_file, "w") as f:
                self.out["frames"] = self.frames
                json.dump(self.out, f, indent=4)

        if self.frame >= 5000:
            self.mission_complete()

        return control

    def finalize(self):
        print("Running finalize")
        # self.writer.close()

        with open(self.log_file, "w") as f:
            json.dump(self.log_entries, f, indent=4)

        # Plot poses
        # fig = go.Figure(data=pose_traces(self.poses))
        # fig.write_html("output/" + self.run_name + "/poses.html")

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
