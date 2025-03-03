#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Mapping agent

"""

import carla
import cv2 as cv
import numpy as np
from pynput import keyboard
import signal

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.util import (
    pose_to_pos_rpy,
    transform_to_numpy,
    transform_to_pos_rpy,
)
from lac.planning.planner import Planner
from lac.perception.vision import FiducialLocalizer
from lac.localization.ekf import EKF, get_pose_measurement_tag, create_Q
from lac.localization.imu_dynamics import propagate_state
from lac.control.controller import waypoint_steering
from lac.mapping.mapper import Mapper
from lac.utils.dashboard import Dashboard
from lac.utils.visualization import overlay_tag_detections
from lac.utils.data_logger import DataLogger
import lac.params as params

""" Agent parameters and settings """
STOP_INTERVAL = 1e7  # Number of steps to stop and reset the velocity
STOP_DURATION_STEPS = 40  # Number of steps to stop for
USE_GROUND_TRUTH_NAV = False  # Whether to use ground truth pose for navigation

DISPLAY_IMAGES = True  # Whether to display the camera views
TELEOP = False  # Whether to use teleop control or autonomous control


def get_entry_point():
    return "MappingAgent"


class MappingAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys."""
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ For teleop """
        self.current_v = 0
        self.current_w = 0

        """ State variable for velocity reset stop """
        self.stop_reset_counter = 0

        """ Controller variables """
        self.steer_delta = 0.0

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
        self.cameras["Right"] = {
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

        """ Localization """
        self.fid_localizer = FiducialLocalizer(self.cameras)
        self.fiducials_last_seen = False
        self.fiducials_stop_counter = 0
        # Initialize EKF
        init_pos, init_rpy = transform_to_pos_rpy(self.get_initial_position())
        v0 = np.zeros(3)
        init_state = np.hstack((init_pos, v0, init_rpy)).T
        self.Q_EKF = create_Q(params.DT, params.EKF_Q_SIGMA_A, params.EKF_Q_SIGMA_ANGLE)
        self.ekf = EKF(init_state, params.EKF_P0, store=True)
        self.current_pose = initial_pose

        """ Mapping """
        self.mapper = Mapper(self.get_geometric_map())

        """ Data logging """
        agent_name = get_entry_point()
        self.data_logger = DataLogger(self, agent_name, self.cameras)
        self.ekf_result_file = f"output/{agent_name}/{params.DEFAULT_RUN_NAME}/ekf_result.npz"

        self.dashboard = Dashboard()
        self.dashboard.start(port=8050)
        self.gt_poses = [initial_pose]

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signal_received, frame):
        print("\nCtrl+C detected! Exiting mission")
        self.mission_complete()

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return True

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
        self.gt_poses.append(ground_truth_pose)

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

            if n_meas > 0:
                self.fiducials_last_seen = True
            elif n_meas == 0 and self.fiducials_last_seen:
                self.fiducials_last_seen = False
                if self.fiducials_stop_counter == 0:
                    self.fiducials_stop_counter = 40
                    self.stop_reset_counter = STOP_DURATION_STEPS
                else:
                    self.fiducials_stop_counter -= 1

            print("Fiducials last seen: ", self.fiducials_last_seen)

            self.data_logger.log_images(self.step, input_data)

        # if self.step % params.EKF_SMOOTHING_INTERVAL == 0:
        self.ekf.smooth()

        # ekf_result = self.ekf.get_results()
        # if self.step < params.EKF_SMOOTHING_INTERVAL:
        #     ekf_state = ekf_result["xhat"][-1]
        # else:
        #     ekf_state = ekf_result["xhat_smooth"][-1]
        # self.current_pose = pos_rpy_to_pose(ekf_state[:3], ekf_state[-3:])
        # print("Position error: ", np.linalg.norm(ekf_state[:3] - ground_truth_pose[:3, 3]))
        self.current_pose = self.ekf.get_pose(self.step)
        position_error = np.linalg.norm(self.current_pose[:3, 3] - ground_truth_pose[:3, 3])
        print("Position error: ", position_error)
        self.dashboard.update_metric(position_error)
        self.dashboard.update_pose_plot(self.gt_poses, self.ekf.get_smoothed_poses())

        if USE_GROUND_TRUTH_NAV:
            nav_pose = ground_truth_pose
        else:
            nav_pose = self.current_pose

        """ Waypoint navigation """
        waypoint, _ = self.planner.get_waypoint(nav_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)
        nominal_steering = waypoint_steering(waypoint, nav_pose)

        if TELEOP:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        else:
            if self.step % STOP_INTERVAL == 0:
                self.stop_reset_counter = STOP_DURATION_STEPS
            if self.stop_reset_counter > 0:
                self.stop_reset_counter -= 1
                print(" ======= STOPPING ======= ")
                control = carla.VehicleVelocityControl(0.0, 0.0)
                # NOTE: should probably wait a few steps to allow rover to stop before zeroing
                if self.stop_reset_counter == 0:
                    self.ekf.zero_velocity_update(self.step)
            else:
                print(params.TARGET_SPEED, nominal_steering)
                control = carla.VehicleVelocityControl(params.TARGET_SPEED, nominal_steering)

        """ Data logging """
        self.data_logger.log_data(self.step, control)
        print("\n-----------------------------------------------")

        return control

    def finalize(self):
        print("Running finalize")
        self.mapper.wheel_contact_update(self.ekf.get_smoothed_poses())
        self.mapper.finalize_heights()

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
