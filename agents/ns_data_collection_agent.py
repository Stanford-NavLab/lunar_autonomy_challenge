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

from lac.util import transform_to_numpy, to_blender_convention
from lac.plotting import pose_traces


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
        self.frame = 0
        self.rate = 5  # Sub-sample rate. Max rate is 10Hz

        self.run_name = "ns_data_collection"
        self.log_file = "output/" + self.run_name + "/data_log.json"

        # Initial rover pose in world frame
        initial_rover_pose = transform_to_numpy(self.get_initial_position())
        # Lander pose in rover frame at initialization
        lander_pose_rover = transform_to_numpy(self.get_initial_lander_position())
        # Lander pose in world frame (constant)
        lander_pose_world = initial_rover_pose @ lander_pose_rover
        print("Initial lander pose in world frame: ", lander_pose_world[:3, 3])
        self.out = {
            "initial_pose": initial_rover_pose.tolist(),
            "lander_pose_rover": lander_pose_rover.tolist(),
            "lander_pose_world": lander_pose_world.tolist(),
        }

        self.frames = []
        self.cameras = {
            "FrontLeft": {"active": True, "light": 0.0, "semantic": False},
            "FrontRight": {"active": True, "light": 0.0, "semantic": False},
            "BackLeft": {"active": False, "light": 0.0, "semantic": False},
            "BackRight": {"active": False, "light": 0.0, "semantic": False},
            "Left": {"active": True, "light": 0.0, "semantic": False},
            "Right": {"active": True, "light": 0.0, "semantic": False},
            "Front": {"active": False, "light": 0.0, "semantic": False},
            "Back": {"active": False, "light": 0.0, "semantic": False},
        }
        self.out["cameras_config"] = self.cameras
        self.out["use_fiducials"] = self.use_fiducials()

        if os.path.exists("output/" + self.run_name):
            shutil.rmtree("output/" + self.run_name)
        for cam, config in self.cameras.items():
            if config["active"]:
                os.makedirs("output/" + self.run_name + "/" + cam)
                if config["semantic"]:
                    os.makedirs("output/" + self.run_name + "/" + cam + "_semantic")

        """ Nerfstudio data logging """
        self.ns_transforms = "output/" + self.run_name + "/transforms.json"
        self.ns_frames = []
        IMAGE_WIDTH = 1280
        IMAGE_HEIGHT = 720
        HFOV = 1.22  # radians
        FX = IMAGE_WIDTH / (2 * np.tan(HFOV / 2))
        FY = FX
        self.ns_out = {
            "camera_model": "OPENCV",
            "fl_x": FX,
            "fl_y": FY,
            "cx": IMAGE_WIDTH / 2,
            "cy": IMAGE_HEIGHT / 2,
            "w": IMAGE_WIDTH,
            "h": IMAGE_HEIGHT,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
        }
        self.cam_poses = []
        self.cam_poses_blender = []

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return False

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

    def run_step(self, input_data):
        # Move the arms out of the way
        if self.frame == 0:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        FL_img = input_data["Grayscale"][carla.SensorPosition.FrontLeft]

        current_pose = transform_to_numpy(self.get_transform())
        imu_data = self.get_imu_data()
        linear_speed = self.get_linear_speed()
        angular_speed = self.get_angular_speed()

        """ We need to check that the sensor data is not None before we do anything with it. The data for each camera will be 
        None for every other simulation step, since the cameras operate at 10Hz while the simulator operates at 20Hz. """
        if FL_img is not None:
            cv.imshow("Front left", FL_img)
            # L_img = input_data["Grayscale"][carla.SensorPosition.Left]
            # cv.imshow("Left", L_img)
            cv.waitKey(1)

            if self.frame % self.rate == 0 and self.frame > 100:
                for cam, config in self.cameras.items():
                    if config["active"]:
                        img = input_data["Grayscale"][getattr(carla.SensorPosition, cam)]
                        cv.imwrite(
                            "output/" + self.run_name + f"/{cam}/" + str(self.frame) + ".png",
                            img,
                        )
                        rover_to_cam = self.get_camera_position(getattr(carla.SensorPosition, cam))
                        cam_pose = current_pose @ transform_to_numpy(rover_to_cam)
                        self.cam_poses.append(cam_pose)
                        cam_pose_blender = to_blender_convention(cam_pose)
                        self.cam_poses_blender.append(cam_pose_blender)
                        frame = {
                            "file_path": f"{cam}/" + str(self.frame) + ".png",
                            "transform_matrix": cam_pose_blender.tolist(),
                        }
                        self.ns_frames.append(frame)

        log_entry = {
            "frame": self.frame,
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
        self.frame += 1

        control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        if self.frame % 100 == 0:
            with open(self.log_file, "w") as f:
                self.out["frames"] = self.frames
                json.dump(self.out, f, indent=4)
            with open(self.ns_transforms, "w") as f:
                self.ns_out["frames"] = self.ns_frames
                json.dump(self.ns_out, f, indent=4)

        # if self.frame == 500:
        #     fig = go.Figure(data=pose_traces(self.cam_poses[::10]))
        #     fig.update_layout(height=900, width=1600, scene_aspectmode="data", title="Camera poses")
        #     fig.show()
        #     fig = go.Figure(data=pose_traces(self.cam_poses_blender[::10]))
        #     fig.update_layout(
        #         height=900, width=1600, scene_aspectmode="data", title="Camera poses blender"
        #     )
        #     fig.show()

        return control

    def finalize(self):
        print("Running finalize")

        with open(self.log_file, "w") as f:
            json.dump(self.out, f, indent=4)

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
