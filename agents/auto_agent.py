#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Full agent

"""

import carla
import cv2 as cv
import numpy as np
import json
from PIL import Image
import signal

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.util import (
    pose_to_pos_rpy,
    transform_to_numpy,
    transform_to_pos_rpy,
)
from lac.perception.segmentation import Segmentation
from lac.perception.depth import (
    stereo_depth_from_segmentation,
)
from lac.control.controller import waypoint_steering, rock_avoidance_steering
from lac.planning.planner import Planner
from lac.localization.imu_dynamics import propagate_state
from lac.utils.visualization import (
    overlay_mask,
    draw_steering_arc,
    overlay_stereo_rock_depths,
    overlay_tag_detections,
)
from lac.utils.data_logger import DataLogger
import lac.params as params


""" Agent parameters and settings """
TARGET_SPEED = 0.15  # [m/s]
IMAGE_PROCESS_RATE = 10  # [Hz]
EARLY_STOP_STEP = 3000  # Number of steps before stopping the mission (0 for no early stop)
USE_GROUND_TRUTH_NAV = False  # Whether to use ground truth pose for navigation

DISPLAY_IMAGES = False  # Whether to display the camera views
LOG_DATA = True  # Whether to log data


def get_entry_point():
    return "AutoAgent"


class AutoAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Controller variables"""
        self.current_v = 0.0
        self.current_w = 0.0

        """ Perception modules """
        self.segmentation = Segmentation()

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.step = 0

        """ Camera config """
        self.cameras = params.CAMERA_CONFIG_INIT
        self.cameras["FrontLeft"] = {
            "active": True,
            "light": 1.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        self.cameras["FrontRight"] = {
            "active": True,
            "light": 1.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        self.active_cameras = [cam for cam, config in self.cameras.items() if config["active"]]

        """ Planner """
        initial_pose = transform_to_numpy(self.get_initial_position())
        self.lander_pose = initial_pose @ transform_to_numpy(self.get_initial_lander_position())
        self.planner = Planner(initial_pose)

        """ Path planner """
        # TODO: initialize ArcPlanner

        """ Data logging """
        if LOG_DATA:
            agent_name = get_entry_point()
            self.data_logger = DataLogger(self, agent_name, self.cameras)
            self.ekf_result_file = f"output/{agent_name}/{params.DEFAULT_RUN_NAME}/ekf_result.npz"
            self.rock_detections_file = (
                f"output/{agent_name}/{params.DEFAULT_RUN_NAME}/rock_detections.json"
            )

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signal_received, frame):
        print("\nCtrl+C detected! Exiting mission")
        self.mission_complete()

    def initialize(self):
        # Move the arms out of the way
        self.set_front_arm_angle(params.ARM_ANGLE_STATIC_RAD)
        self.set_back_arm_angle(params.ARM_ANGLE_STATIC_RAD)

    def image_available(self):
        return self.step % 2 == 0  # Image data is available every other step

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return False

    def sensors(self):
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

    def run_step(self, input_data):
        if self.step == 0:
            self.initialize()
        self.step += 1  # Starts at 0 at init, equal to 1 on the first run_step call
        print("\nStep: ", self.step)

        ground_truth_pose = transform_to_numpy(self.get_transform())
        nav_pose = ground_truth_pose

        """ Waypoint navigation """
        waypoint, advanced = self.planner.get_waypoint(nav_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)
        nominal_steering = waypoint_steering(waypoint, nav_pose)

        """ Rock segmentation """
        if self.step % (params.FRAME_RATE // IMAGE_PROCESS_RATE) == 0:
            FL_gray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
            FR_gray = input_data["Grayscale"][carla.SensorPosition.FrontRight]
            FL_rgb_PIL = Image.fromarray(FL_gray).convert("RGB")
            FR_rgb_PIL = Image.fromarray(FR_gray).convert("RGB")

            # Run segmentation
            left_seg_masks, left_seg_full_mask = self.segmentation.segment_rocks(FL_rgb_PIL)
            right_seg_masks, right_seg_full_mask = self.segmentation.segment_rocks(FR_rgb_PIL)

            # Stereo rock depth
            stereo_depth_results = stereo_depth_from_segmentation(
                left_seg_masks, right_seg_masks, params.STEREO_BASELINE, params.FL_X
            )

            # Path planning
            # TODO: set self.current_v and self.current_w based on output of Arc Planner

            if DISPLAY_IMAGES:
                overlay = overlay_mask(FL_gray, left_seg_full_mask)
                overlay = draw_steering_arc(overlay, nominal_steering, color=(255, 0, 0))
                overlay = overlay_stereo_rock_depths(overlay, stereo_depth_results)
                overlay = draw_steering_arc(
                    overlay, nominal_steering + self.steer_delta, color=(0, 255, 0)
                )
                cv.imshow("Rock segmentation", overlay)
                cv.waitKey(1)

        control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        """ Data logging """
        if LOG_DATA:
            self.data_logger.log_data(self.step, control)

        print("\n-----------------------------------------------")

        return control

    def finalize(self):
        print("Running finalize")

        if LOG_DATA:
            self.data_logger.save_log()

        if DISPLAY_IMAGES:
            cv.destroyAllWindows()
