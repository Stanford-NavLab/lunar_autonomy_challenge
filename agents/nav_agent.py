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
from lunar_autonomy_challenge.lac.perception.segmentation_util import Segmentation
from lac.perception.depth import (
    stereo_depth_from_segmentation,
)
from lac.perception.vision import FiducialLocalizer
from lac.control.controller import waypoint_steering, rock_avoidance_steering
from lac.planning.planner import Planner
from lac.localization.ekf import EKF, get_pose_measurement_tag, create_Q
from lac.localization.imu_dynamics import propagate_state
from lac.mapping.mapper import Mapper
from lac.utils.visualization import (
    overlay_mask,
    draw_steering_arc,
    overlay_stereo_rock_depths,
    overlay_tag_detections,
)
from lac.utils.data_logger import DataLogger
import lac.params as params


""" Agent parameters and settings """
EVAL = False  # Whether running in evaluation mode (disable ground truth)
USE_FIDUCIALS = True

TARGET_SPEED = 0.15  # [m/s]
IMAGE_PROCESS_RATE = 10  # [Hz]
EARLY_STOP_STEP = 3000  # Number of steps before stopping the mission (0 for no early stop)
USE_GROUND_TRUTH_NAV = True  # Whether to use ground truth pose for navigation

DISPLAY_IMAGES = True  # Whether to display the camera views
LOG_DATA = True  # Whether to log data

if EVAL:
    USE_GROUND_TRUTH_NAV = False
    DISPLAY_IMAGES = False
    LOG_DATA = False


def get_entry_point():
    return "NavAgent"


class NavAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Controller variables"""
        self.steer_delta = 0.0

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
        if USE_FIDUCIALS:
            self.cameras["Right"] = {
                "active": True,
                "light": 1.0,
                "width": 1280,
                "height": 720,
                "semantic": False,
            }
        self.active_cameras = [cam for cam, config in self.cameras.items() if config["active"]]

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
        self.current_pose = initial_pose

        """ Mapping """
        self.mapper = Mapper(self.get_geometric_map())

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
        return USE_FIDUCIALS

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

        if EARLY_STOP_STEP != 0 and self.step >= EARLY_STOP_STEP:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        if not EVAL:
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
            if self.use_fiducials():
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

                    if DISPLAY_IMAGES and cam == "Right":
                        overlay = overlay_tag_detections(images_gray[cam], detections)
                        cv.imshow(cam, overlay)

                """ EKF update step """
                n_meas = len(fid_measurements)
                fid_measurements = np.array(fid_measurements).flatten()

                def meas_func(x):
                    return get_pose_measurement_tag(x, n_meas)

                self.ekf.update(self.step, fid_measurements, meas_func)

            if LOG_DATA:
                self.data_logger.log_images(self.step, input_data)

        # if self.step % params.EKF_SMOOTHING_INTERVAL == 0:
        #     self.ekf.smooth()
        self.ekf.smooth()

        # ekf_result = self.ekf.get_results()
        # ekf_state = ekf_result["xhat_smooth"][-1]
        # self.current_pose = pos_rpy_to_pose(ekf_state[:3], ekf_state[-3:])
        self.current_pose = self.ekf.get_pose(self.step)
        if not EVAL:
            print(
                "Position error: ",
                np.linalg.norm(self.current_pose[:3, 3] - ground_truth_pose[:3, 3]),
            )

        if USE_GROUND_TRUTH_NAV:
            nav_pose = ground_truth_pose
        else:
            nav_pose = self.current_pose

        """ Waypoint navigation """
        waypoint, advanced = self.planner.get_waypoint(nav_pose, print_progress=True)
        # if advanced:
        #     self.mapper.wheel_contact_update(self.ekf.get_smoothed_poses())
        #     self.mapper.finalize_heights()
        #     self.mapper.rock_projection_update(self.ekf.get_smoothed_poses(), self.cameras)
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
            self.mapper.add_rock_detections(self.step, stereo_depth_results)

            # Hazard avoidance
            self.steer_delta = rock_avoidance_steering(stereo_depth_results, self.cameras)

            if DISPLAY_IMAGES:
                overlay = overlay_mask(FL_gray, left_seg_full_mask, color=(0, 0, 1))
                overlay = draw_steering_arc(overlay, nominal_steering, color=(255, 0, 0))
                overlay = overlay_stereo_rock_depths(overlay, stereo_depth_results)
                overlay = draw_steering_arc(
                    overlay, nominal_steering + self.steer_delta, color=(0, 255, 0)
                )
                cv.imshow("Rock segmentation", overlay)
                cv.waitKey(1)

        target_speed = TARGET_SPEED
        if self.steer_delta != 0:
            target_speed = 0.1
        control = carla.VehicleVelocityControl(target_speed, nominal_steering + self.steer_delta)

        """ Data logging """
        if LOG_DATA:
            self.data_logger.log_data(self.step, control)

        print("\n-----------------------------------------------")

        return control

    def finalize(self):
        print("Running finalize")
        self.mapper.wheel_contact_update(self.ekf.get_smoothed_poses())
        self.mapper.finalize_heights()
        self.mapper.rock_projection_update(self.ekf.get_smoothed_poses(), self.cameras)

        if LOG_DATA:
            self.data_logger.save_log()
            np.savez(self.ekf_result_file, **self.ekf.get_results())

        if DISPLAY_IMAGES:
            cv.destroyAllWindows()
