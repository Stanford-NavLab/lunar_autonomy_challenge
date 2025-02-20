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
from lac.perception.segmentation import Segmentation
from lac.perception.depth import (
    DepthAnything,
    stereo_depth_from_segmentation,
    project_depths_to_world,
)
from lac.localization.imu_recovery import ImuEstimator
from lac.utils.visualization import overlay_mask, draw_steering_arc, overlay_stereo_rock_depths
from lac.utils.frames import apply_transform
import lac.params as params


def get_entry_point():
    return "NavAgent"


class NavAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys."""
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ For teleop """
        self.current_v = 0
        self.current_w = 0
        self.TELEOP = False

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
        self.segmentation = Segmentation()
        self.depth = DepthAnything()
        self.frame = 0
        self.image_process_rate = 1  # Hz

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.step = 0

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

        """ Rock mapping """
        self.all_rock_detections = []

        """ Waypoints """
        self.waypoints = gen_square_spiral(max_val=4.5, min_val=2.0, step=0.5)
        self.waypoint_idx = 0
        self.waypoint_threshold = 1.0  # meters

        """ Localization """
        self.imu_estimator = ImuEstimator(
            initial_pose=transform_to_numpy(self.get_initial_position()), dt=0.05
        )
        self.imu_start_frame = 50

        """ Data logging """
        self.run_name = "nav_agent"
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
            "FrontRight": {"active": True, "light": 0.0, "semantic": False},
            "BackLeft": {"active": False, "light": 0.0, "semantic": False},
            "BackRight": {"active": False, "light": 0.0, "semantic": False},
            "Left": {"active": False, "light": 0.0, "semantic": False},
            "Right": {"active": True, "light": 0.0, "semantic": False},
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

    def segmentation_steering(self, masks):
        """Compute a steering delta based on segmentation results to avoid rocks.

        TODO: account for offset due to operating on left camera
        """
        max_area = 0
        max_mask = None
        for mask in masks:
            mask_area = np.sum(mask)
            if mask_area > max_area:
                max_area = mask_area
                max_mask = mask
        steer_delta = 0
        if max_mask is not None and max_area > params.ROCK_MASK_AVOID_MIN_AREA:
            max_mask = max_mask.astype(np.uint8)
            cx, cy = mask_centroid(max_mask)
            x, y, w, h = cv.boundingRect(max_mask)
            offset = params.IMG_WIDTH // 2 - cx
            if offset > 0:  # Turn right
                steer_delta = -min(
                    self.max_steer_delta * ((x + w) - cx) / 100, self.max_steer_delta
                )
            else:  # Turn left
                steer_delta = min(self.max_steer_delta * (cx - x) / 100, self.max_steer_delta)
        return steer_delta

    def run_step(self, input_data):
        # Move the arms out of the way
        if self.step == 0:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        current_pose = transform_to_numpy(self.get_transform())
        rpy, pos = pose_to_rpy_pos(current_pose)
        heading = rpy[2]

        # Wheel contact mapping
        wheel_contact_points = apply_transform(current_pose, params.WHEEL_RIG_POINTS)

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
        print(f"Waypoint {self.waypoint_idx + 1} / {len(self.waypoints)}")
        waypoint = self.waypoints[self.waypoint_idx]

        angle = np.arctan2(waypoint[1] - pos[1], waypoint[0] - pos[0])
        angle_diff = wrap_angle(angle - heading)
        nominal_steering = np.clip(self.KP_STEER * angle_diff, -self.max_steer, self.max_steer)

        # Perception
        FL_gray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        FR_gray = input_data["Grayscale"][carla.SensorPosition.FrontRight]
        if (
            FL_gray is not None
            and (self.step - 1) % (params.FRAME_RATE // self.image_process_rate) == 0
        ):
            FL_rgb_PIL = Image.fromarray(FL_gray).convert("RGB")
            FR_rgb_PIL = Image.fromarray(FR_gray).convert("RGB")

            # Run segmentation
            left_seg_masks, left_seg_full_mask = self.segmentation.segment_rocks(FL_rgb_PIL)
            right_seg_masks, right_seg_full_mask = self.segmentation.segment_rocks(FR_rgb_PIL)

            # Stereo rock depth
            stereo_depth_results = stereo_depth_from_segmentation(
                left_seg_masks, right_seg_masks, params.STEREO_BASELINE, params.FL_X
            )
            rock_points_world = project_depths_to_world(
                stereo_depth_results, params.CAMERA_INTRINSICS, current_pose
            )
            for point in rock_points_world:
                g_map.set_rock(point[0], point[1], True)
                self.all_rock_detections.append(point)

            # Hazard avoidance
            self.steer_delta = self.segmentation_steering(left_seg_masks)

            if self.DISPLAY:
                overlay = overlay_mask(FL_gray, left_seg_full_mask)
                overlay = draw_steering_arc(overlay, nominal_steering, color=(255, 0, 0))
                overlay = overlay_stereo_rock_depths(overlay, stereo_depth_results)
                overlay = draw_steering_arc(
                    overlay, nominal_steering + self.steer_delta, color=(0, 255, 0)
                )
                # cv.imshow("Left camera view", FL_gray)
                cv.imshow("Rock segmentation", overlay)
                cv.waitKey(1)

            R_img = input_data["Grayscale"][carla.SensorPosition.Right]
            cv.imwrite(
                "output/" + self.run_name + "/Right/" + str(self.step) + ".png",
                R_img,
            )

        if self.TELEOP:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        else:
            control = carla.VehicleVelocityControl(
                self.TARGET_SPEED, nominal_steering + self.steer_delta
            )

        # Data logging
        imu_data = self.get_imu_data()
        linear_speed = self.get_linear_speed()
        angular_speed = self.get_angular_speed()
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
            np.save(self.rock_points_file, self.all_rock_detections)

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
