#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Localization agent

"""

import signal

import carla
import cv2 as cv
import lac.params as params
import numpy as np
from lac.control.controller import waypoint_steering
from lac.localization.ekf import EKF, create_Q, get_pose_measurement_tag
from lac.localization.imu_dynamics import propagate_state
from lac.perception.vision import FiducialLocalizer
from lac.planning.waypoint_planner import Planner
from lac.util import (
    pos_rpy_to_pose,
    pose_to_pos_rpy,
    transform_to_numpy,
    transform_to_pos_rpy,
)
from lac.utils.data_logger import DataLogger
from lac.utils.visualization import overlay_tag_detections
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from pynput import keyboard

""" Agent parameters and settings """
USE_FIDUCIALS = False

DISPLAY_IMAGES = True  # Whether to display the camera views
TELEOP = False  # Whether to use teleop control or autonomous control
USE_GROUND_TRUTH_NAV = True  # Whether to use ground truth pose for navigation


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

        """ Controller variables """
        self.steer_delta = 0.0

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.step = 0

        """ Camera config """
        self.cameras = params.CAMERA_CONFIG_INIT
        self.cameras["FrontLeft"] = {
            "active": True,
            "light": 0.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        if USE_FIDUCIALS:
            self.cameras["Right"] = {
                "active": True,
                "light": 0.0,
                "width": 1280,
                "height": 720,
                "semantic": False,
            }
        self.active_cameras = [cam for cam, config in self.cameras.items() if config["active"]]

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
        self.current_pose = initial_pose

        """ Data logging """
        agent_name = get_entry_point()
        self.data_logger = DataLogger(self, agent_name, self.cameras)
        self.ekf_result_file = f"output/{agent_name}/{params.DEFAULT_RUN_NAME}/ekf_result.npz"

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signal_received, frame):
        print("\nCtrl+C detected! Exiting mission")
        self.mission_complete()

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return USE_FIDUCIALS

    def sensors(self):
        """In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048)
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light.
        """

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
        print("\nStep: ", self.step)

        ground_truth_pose = transform_to_numpy(self.get_transform())

        """ EKF predict step """
        imu_data = self.get_imu_data()
        a_k = imu_data[:3]
        omega_k = imu_data[3:]

        def dyn_func(x):
            return propagate_state(x, a_k, omega_k, params.DT, with_stm=True, use_numdiff=False)

        self.ekf.predict(self.step, dyn_func, self.Q_EKF)

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

                if DISPLAY_IMAGES:
                    overlay = overlay_tag_detections(images_gray[cam], detections)
                    cv.imshow(cam, overlay)

            if DISPLAY_IMAGES:
                cv.waitKey(1)

            """ EKF update step """
            n_meas = len(fid_measurements)
            print("# measurements: ", n_meas)
            fid_measurements = np.array(fid_measurements).flatten()

            def meas_func(x):
                return get_pose_measurement_tag(x, n_meas)

            self.ekf.update(self.step, fid_measurements, meas_func)

            self.data_logger.log_images(self.step, input_data)

        if self.current_v == 0 and self.current_w == 0:
            self.ekf.zero_velocity_update(self.step)

        if self.step % params.EKF_SMOOTHING_INTERVAL == 0:
            self.ekf.smooth()

        ekf_result = self.ekf.get_results()
        ekf_state = ekf_result["xhat_smooth"][-1]
        self.current_pose = pos_rpy_to_pose(ekf_state[:3], ekf_state[-3:])
        print("Position error: ", np.linalg.norm(ekf_state[:3] - ground_truth_pose[:3, 3]))

        if USE_GROUND_TRUTH_NAV:
            nav_pose = ground_truth_pose
        else:
            nav_pose = self.current_pose

        """ Waypoint navigation """
        waypoint, advanced = self.planner.get_waypoint(nav_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)
        if advanced:
            self.data_logger.save_log()
            np.savez(self.ekf_result_file, **self.ekf.get_results())
        nominal_steering = waypoint_steering(waypoint, nav_pose)

        if TELEOP:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        else:
            control = carla.VehicleVelocityControl(params.TARGET_SPEED, nominal_steering)

        # Data logging
        self.data_logger.log_data(self.step, control)

        print("\n-----------------------------------------------")

        return control

    def finalize(self):
        print("Running finalize")
        self.data_logger.save_log()
        np.savez(self.ekf_result_file, **self.ekf.get_results())

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
