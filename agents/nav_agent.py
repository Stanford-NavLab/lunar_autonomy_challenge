#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Full agent

"""

import carla
import cv2 as cv
import numpy as np
from pynput import keyboard
from PIL import Image

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.util import (
    pose_to_pos_rpy,
    transform_to_numpy,
    transform_to_pos_rpy,
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
from lac.perception.vision import FiducialLocalizer
from lac.control.controller import waypoint_steering, segmentation_steering
from lac.planning.planner import Planner
from lac.localization.ekf import EKF, get_pose_measurement_tag, create_Q
from lac.localization.imu_dynamics import propagate_state
from lac.utils.visualization import overlay_mask, draw_steering_arc, overlay_stereo_rock_depths
from lac.utils.frames import apply_transform
from lac.utils.data_logger import DataLogger
import lac.params as params

""" Agent parameters and settings """
IMAGE_PROCESS_RATE = 10  # [Hz]

DISPLAY_IMAGES = True  # Whether to display the camera views
TELEOP = False  # Whether to use teleop control or autonomous control
UPDATE_LOG_RATE = 100  # Update log file every 100 steps


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

        """ Controller variables """
        self.steer_delta = 0.0

        """ Perception modules """
        self.segmentation = Segmentation()
        self.depth = DepthAnything()

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.step = 0

        """ Rock mapping """
        self.all_rock_detections = []

        """ Planner """
        initial_pose = transform_to_numpy(self.get_initial_position())
        self.lander_pose = initial_pose @ transform_to_numpy(self.get_initial_lander_position())
        self.planner = Planner(initial_pose)

        """ Localization """
        self.fid_localizer = FiducialLocalizer(self.cameras)
        # Initialize EKF
        init_pos, init_rpy = transform_to_pos_rpy(self.get_initial_position())
        v0 = np.zeros(3)
        init_state = np.hstack((init_pos, v0, init_rpy)).T
        self.Q_EKF = create_Q(params.DT, params.EKF_Q_SIGMA_A, params.EKF_Q_SIGMA_ANGLE)
        self.ekf = EKF(init_state, params.EKF_P0, store=True)

        """ Data logging """
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
        self.cameras["Right"] = {
            "active": True,
            "light": 1.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        self.active_cameras = [cam for cam, config in self.cameras.items() if config["active"]]

        run_name = get_entry_point()
        self.data_logger = DataLogger(self, run_name, self.cameras)
        self.ekf_result_file = f"output/{run_name}/ekf_result.npz"
        self.rock_points_file = f"output/{run_name}/rock_points.npy"

    def initialize(self):
        # Move the arms out of the way
        self.set_front_arm_angle(params.ARM_ANGLE_STATIC_RAD)
        self.set_back_arm_angle(params.ARM_ANGLE_STATIC_RAD)

    def image_available(self):
        return self.step % 2 == 0  # Image data is available every other step

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

        current_pose = transform_to_numpy(self.get_transform())
        pos, rpy = pose_to_pos_rpy(current_pose)

        """ EKF predict step """
        imu_data = self.get_imu_data()
        a_k = imu_data[:3]
        omega_k = imu_data[3:]

        def dyn_func(x):
            return propagate_state(x, a_k, omega_k, params.DT, with_stm=True, use_numdiff=False)

        self.ekf.predict(self.step, dyn_func, self.Q_EKF)

        """ Wheel contact mapping """
        wheel_contact_points = apply_transform(current_pose, params.WHEEL_RIG_POINTS)

        g_map = self.get_geometric_map()
        for point in wheel_contact_points:
            current_height = g_map.get_height(point[0], point[1])
            if current_height is None:  # Out of bounds
                continue
            if (current_height == -np.inf) or (current_height > point[2]):
                g_map.set_height(point[0], point[1], point[2])

        """ Waypoint navigation """
        waypoint = self.planner.get_waypoint(pos, print_progress=True)
        if waypoint is None:
            self.mission_complete()
        # waypoint = self.waypoints[self.waypoint_idx]
        # if np.linalg.norm(pos[:2] - waypoint) < params.WAYPOINT_REACHED_DIST_THRESHOLD:
        #     self.waypoint_idx += 1
        #     if self.waypoint_idx >= len(self.waypoints):
        #         self.mission_complete()
        #         self.waypoint_idx = 0
        # print(f"Waypoint {self.waypoint_idx + 1} / {len(self.waypoints)}")

        nominal_steering = waypoint_steering(waypoint, pos, rpy)

        """ Image processing """
        if self.image_available():
            images_gray = {}
            fid_measurements = []
            for cam in self.active_cameras:
                images_gray[cam] = input_data["Grayscale"][getattr(carla.SensorPosition, cam)]
                pose_measurements, detections = self.fid_localizer.estimate_rover_pose(
                    images_gray[cam], cam, self.lander_pose
                )
                for pose in pose_measurements.values():
                    meas = np.concatenate(pose_to_pos_rpy(pose))
                    fid_measurements.append(meas)

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
                rock_points_world = project_depths_to_world(
                    stereo_depth_results, current_pose, "FrontLeft", self.cameras
                )
                for point in rock_points_world:
                    g_map.set_rock(point[0], point[1], True)
                    self.all_rock_detections.append(point)

                # Hazard avoidance
                self.steer_delta = segmentation_steering(left_seg_masks)

            if DISPLAY_IMAGES:
                overlay = overlay_mask(FL_gray, left_seg_full_mask)
                overlay = draw_steering_arc(overlay, nominal_steering, color=(255, 0, 0))
                overlay = overlay_stereo_rock_depths(overlay, stereo_depth_results)
                overlay = draw_steering_arc(
                    overlay, nominal_steering + self.steer_delta, color=(0, 255, 0)
                )
                # cv.imshow("Left camera view", FL_gray)
                cv.imshow("Rock segmentation", overlay)
                cv.waitKey(1)

            self.data_logger.log_images(self.step, input_data)

        if TELEOP:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        else:
            control = carla.VehicleVelocityControl(
                params.TARGET_SPEED, nominal_steering + self.steer_delta
            )

        # Data logging
        self.data_logger.log_data(self.step, control)
        if self.step % UPDATE_LOG_RATE == 0:
            self.data_logger.save_log()
            np.save(self.rock_points_file, self.all_rock_detections)

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
