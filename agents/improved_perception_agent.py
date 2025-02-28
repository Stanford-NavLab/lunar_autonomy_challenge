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
    gen_square_spiral,
    cv_display_text,
)

from lac.perception.perception_utils import (
    get_camera_intrinsics, 
    get_homogenous_transform,
    get_extrinsic_left_to_right, 
    decompose_homogenous_transform,
    stereo_rectify,
    segment_image
)

from lac.perception.segmentation import Segmentation
from lac.perception.depth import DepthAnything
from lac.localization.imu_recovery import ImuEstimator


def get_entry_point():
    return "NavAgent"


class NavAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys."""
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ Time"""
        self.time = 0


        """ For teleop """
        self.current_v = 0
        self.current_w = 0
        self.TELEOP = False

        self.IMG_WIDTH = 1280
        self.IMG_HEIGHT = 720
        self.DISPLAY = False
        self.max_steer = 1.0  # rad/s
        self.max_steer_delta = 0.6  # rad/s

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

        """ Wheel contact mapping """
        self.wheel_rig = np.array(
            [
                [0.222, 0.203, -0.134],
                [0.222, -0.203, -0.134],
                [-0.222, 0.203, -0.134],
                [-0.222, 0.203, -0.134],
            ]
        )
        self.wheel_rig_coords = np.concatenate((self.wheel_rig.T, np.ones((1, 4))), axis=0)

        """ Waypoints """
        self.waypoints = gen_square_spiral(max_val=4.5, min_val=2.0, step=0.5)
        self.waypoint_idx = 0
        self.waypoint_threshold = 1.0  # meters

        """ IMU localization """
        self.imu_estimator = ImuEstimator(
            initial_pose=transform_to_numpy(self.get_initial_position()), dt=0.05
        )
        self.imu_start_frame = 50

        """ Data logging """
        self.run_name = "nav_agent"
        self.log_file = "output/" + self.run_name + "/data_log.json"
        initial_rover_pose = transform_to_numpy(self.get_initial_position())
        lander_pose_rover = transform_to_numpy(self.get_initial_lander_position())
        lander_pose_world = initial_rover_pose @ lander_pose_rover
        print("Initial lander pose in world frame: ", lander_pose_world[:3, 3])
        self.out = {
            "initial_pose": initial_rover_pose.tolist(),
            "lander_pose_rover": lander_pose_rover.tolist(),
            "lander_pose_world": lander_pose_world.tolist(),
        }
        self.frames = []
        if not os.path.exists("output/" + self.run_name):
            os.makedirs("output/" + self.run_name)

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
                "camera_active": True,
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



        # # Wait 50 frames for IMU to stabilize and motion to go to zero
        # while self.frame < self.imu_start_frame:
        #     control = carla.VehicleVelocityControl(0.0, 0.0)
        #     return control

        # if self.frame == self.imu_start_frame:
        #     self.imu_estimator.reset(transform_to_numpy(self.get_initial_position()))

        current_pose = transform_to_numpy(self.get_transform())
        rpy, pos = pose_to_rpy_pos(current_pose)
        heading = rpy[2]

        # Wheel contact mapping
        wheel_contact_points = current_pose @ self.wheel_rig_coords
        wheel_contact_points = wheel_contact_points[:3, :].T

        g_map = self.get_geometric_map()
        for point in wheel_contact_points:
            current_height = g_map.get_height(point[0], point[1])
            if current_height is None:  # Out of bounds
                continue
            if (current_height == -np.inf) or (current_height > point[2]):
                g_map.set_height(point[0], point[1], point[2])

        # Navigate to the next waypoint
        if np.linalg.norm(pos[:2] - self.waypoints[self.waypoint_idx]) < self.waypoint_threshold:
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.waypoints):
                self.mission_complete()
                self.waypoint_idx = 0
        #print(f"Waypoint {self.waypoint_idx + 1} / {len(self.waypoints)}")
        waypoint = self.waypoints[self.waypoint_idx]

        angle = np.arctan2(waypoint[1] - pos[1], waypoint[0] - pos[0])
        angle_diff = wrap_angle(angle - heading)
        nominal_steering = np.clip(self.KP_STEER * angle_diff, -self.max_steer, self.max_steer)
        

        # Perception
        steer_angle = 0
        steer_speed = 0
        self.time += 1/20
        left_key = carla.SensorPosition.FrontLeft
        right_key = carla.SensorPosition.FrontRight
        FL_gray_left = input_data["Grayscale"][left_key]
        FL_gray_right = input_data["Grayscale"][right_key]
        if FL_gray_left is not None:
            
            # Rectify stereo images
            K_left, D_left = get_camera_intrinsics(self.sensors()[left_key])
            K_right, D_right = get_camera_intrinsics(self.sensors()[right_key])   
            M_left2right = get_extrinsic_left_to_right(self, left_key, right_key)
            R_left2right, T_left2right = decompose_homogenous_transform(M_left2right)
            left_rect, right_rect = stereo_rectify(FL_gray_left, FL_gray_right, K_left, D_left, K_right, D_right, R_left2right, T_left2right)

            # Segmentation
            cx_left, cy_left = segment_image(self, left_rect)
            cx_right, cy_right = segment_image(self, right_rect)
                
            if cx_left is not None and cx_right is not None:
                # Rock coordinates in left camera frame
                disparity = cx_left - cx_right
                z_rock_left = K_left[0, 0] * abs(T_left2right[1]) / disparity
                x_rock_left = z_rock_left * (cx_left - K_left[0, 2]) / K_left[0, 0]
                y_rock_left = z_rock_left * (cy_left - K_left[1, 2]) / K_left[1, 1]
                rock_coords_left = np.array([x_rock_left, y_rock_left, z_rock_left])

                # Rock coordinates in rover frame
                transform_left = self.get_camera_position(left_key)
                rock_coords_rover = get_homogenous_transform(transform_left) @ np.concatenate((rock_coords_left, np.array([1])))
                
                print(f"Rock coordinates in rover frame: {rock_coords_rover}")
                
                Xr = rock_coords_rover[0]
                Yr = rock_coords_rover[1]
                Zr = rock_coords_rover[2]

                distance = np.sqrt(Xr**2 + Yr**2 + Zr**2)

                danger_zone = 4
                if distance < danger_zone:
                    print(f"Rock detected at {distance} meters")
                    sign = -1 if Yr < 0 else +1
                    k_avoid = 0.6
                    steer_angle = k_avoid * (danger_zone - distance)**2
                    steer_angle = steer_angle * sign
                    steer_speed = -0.1
        

        if self.TELEOP:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        else:
            control = carla.VehicleVelocityControl(
                self.TARGET_SPEED + steer_speed, nominal_steering + steer_angle
            )

        # Data logging
        imu_data = self.get_imu_data()
        linear_speed = self.get_linear_speed()
        angular_speed = self.get_angular_speed()
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
        if self.frame % 100 == 0:
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
        """This is the callback execlocity. If the key pressed is either the left or right arrow, this method will set a target angular
        velocity of 0.6 radians per suted when a key is pressed. If the key pressed is either the up or down arrow, this method will add
        or subtract target linear veecond."""

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
