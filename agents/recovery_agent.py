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
from lac.control.controller import waypoint_steering
from lac.planning.planner import Planner
from lac.localization.ekf import EKF, get_pose_measurement_tag, create_Q
from lac.localization.imu_dynamics import propagate_state
from lac.utils.visualization import (
    overlay_mask,
    draw_steering_arc,
    overlay_stereo_rock_depths,
)
from lac.utils.data_logger import DataLogger
from lac.utils.rerun_interface import Rerun
import lac.params as params


""" Agent parameters and settings """
TARGET_SPEED = 0.15  # [m/s]
IMAGE_PROCESS_RATE = 10  # [Hz]

DISPLAY_IMAGES = True  # Whether to display the camera views
LOG_DATA = True  # Whether to log data


def get_entry_point():
    return "RecoveryAgent"


class RecoveryAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Controller variables"""
        self.current_v = 0.0
        self.current_w = 0.0

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.step = 0

        """ Initialize a counter for backup maneuvers. """
        self.backup_counter = 0

        """ Initialize a counter for how long the rover is stuck. """
        self.stuck_counter = 0

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

        """ Localization """
        # Initialize EKF
        init_pos, init_rpy = transform_to_pos_rpy(self.get_initial_position())
        v0 = np.zeros(3)
        init_state = np.hstack((init_pos, v0, init_rpy)).T
        self.Q_EKF = create_Q(params.DT, params.EKF_Q_SIGMA_A, params.EKF_Q_SIGMA_ANGLE)
        self.ekf = EKF(init_state, params.EKF_P0, store=True)
        self.current_pose = initial_pose

        """ Data logging """
        if LOG_DATA:
            agent_name = get_entry_point()
            self.data_logger = DataLogger(self, agent_name, self.cameras)
            self.ekf_result_file = f"output/{agent_name}/{params.DEFAULT_RUN_NAME}/ekf_result.npz"
            self.rock_detections_file = (
                f"output/{agent_name}/{params.DEFAULT_RUN_NAME}/rock_detections.json"
            )

        Rerun.init_vo()
        # self.ekf_states = []
        self.gt_poses = [initial_pose]

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
    

    def run_nominal_step(self, input_data):
        self.step += 1  # Starts at 0 at init, equal to 1 on the first run_step call

        ground_truth_pose = transform_to_numpy(self.get_transform())
        nav_pose = ground_truth_pose

        """ Waypoint navigation """
        waypoint, advanced = self.planner.get_waypoint(nav_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)
        nominal_steering = waypoint_steering(waypoint, nav_pose)
        
        control = carla.VehicleVelocityControl(0.2, nominal_steering)
        return control
    
    def run_backup_maneuver(self, input_data):
        print("Running backup maneuver")
        frame_rate = params.FRAME_RATE
        self.backup_counter += 1
        if self.backup_counter <= frame_rate * 3: # Go backwards for 3 seconds
            control = carla.VehicleVelocityControl(-0.2, 0.0)
        elif self.backup_counter <= frame_rate * 4: # Rotate 90 degrees in 1 second
            control = carla.VehicleVelocityControl(0.0, -np.pi/2)
        elif self.backup_counter <= frame_rate * 5: # Move in an arc around the rock for 1 second
            control = carla.VehicleVelocityControl(0.2, np.pi/2)
        else:
            self.backup_counter = 0
            control = self.run_nominal_step(input_data)
        return control
    
    def check_stuck(self, rov_vel):
        # Agent is stuck if the velocity is less than 0.1 m/s
        # is_stuck = np.linalg.norm(ekf_cur_state[3:6]) < 0.1
        frame_rate = params.FRAME_RATE
        is_stuck = np.linalg.norm(rov_vel) < 0.05
        if is_stuck:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        return is_stuck and self.stuck_counter >= frame_rate # 1 second


    def run_step(self, input_data):  # This runs at 20 Hz
        if self.step == 0:
            self.initialize()
        print("\nStep: ", self.step)

        ground_truth_pose = transform_to_numpy(self.get_transform())
        self.gt_poses.append(ground_truth_pose)

        ''' WHEN USING EKF TO CHECK IF THE ROVER IS STUCK:
        # """ EKF predict step """
        # imu_data = self.get_imu_data()
        # a_k = imu_data[:3]
        # omega_k = imu_data[3:]

        # def dyn_func(x):
        #     return propagate_state(x, a_k, omega_k, params.DT, with_stm=True, use_numdiff=False)

        # self.ekf.predict(self.step, dyn_func, self.Q_EKF)
        # self.ekf.smooth()
        # self.current_pose = self.ekf.get_pose(self.step)
        # ekf_result = self.ekf.get_results()
        # ekf_cur_state = ekf_result["xhat_smooth"][-1]
        # self.ekf_states.append(ekf_cur_state) 
        # position_error = self.current_pose[:3, 3] - ground_truth_pose[:3, 3]
        '''

        gt_trajectory = np.array([pose[:3, 3] for pose in self.gt_poses])

        Rerun.log_3d_trajectory(
            self.step, gt_trajectory, trajectory_string="ground_truth", color=[0, 0, 255]
        )
        # Rerun.log_2d_seq_scalar("ground_truth_pose/x", self.step, ground_truth_pose[0, 3])
        # Rerun.log_2d_seq_scalar("ground_truth_pose/y", self.step, ground_truth_pose[1, 3])  
        # Rerun.log_2d_seq_scalar("ground_truth_pose/z", self.step, ground_truth_pose[2, 3])

        # Obtain velocity estimate from ground truth poses
        rov_vel = np.zeros(3)
        print("rov_vel: ", rov_vel)
        if self.step > 0:
            dt = params.DT
            rov_vel = (gt_trajectory[-1] - gt_trajectory[-2]) / dt
            print("rov_vel: ", rov_vel)
            print("rov_vel[0]: ", rov_vel[0])
            Rerun.log_2d_seq_scalar("trajectory_error/vx", self.step, rov_vel[0])
            Rerun.log_2d_seq_scalar("trajectory_error/vy", self.step, rov_vel[1])
            Rerun.log_2d_seq_scalar("trajectory_error/vz", self.step, rov_vel[2])
            Rerun.log_2d_seq_scalar("trajectory_error/v", self.step, np.linalg.norm(rov_vel))

        """ Waypoint navigation """
        # If agent is stuck, perform backup maneuver
        if self.backup_counter > 0 or self.check_stuck(rov_vel):
            print("Agent is stuck.")
            control = self.run_backup_maneuver(input_data)
        else:
            control = self.run_nominal_step(input_data)

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
