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
import pickle
from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.util import (
    pose_to_pos_rpy,
    transform_to_numpy,
    transform_to_pos_rpy,
)
from lac.perception.segmentation import UnetSegmentation
from lac.perception.depth import (
    stereo_depth_from_segmentation,
    compute_rock_coords_rover_frame,
    compute_rock_radii,
)
from lac.control.controller import waypoint_steering, ArcPlanner
from lac.planning.waypoint_planner import Planner
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
ENABLE_RERUN = True  # Whether to enable Rerun dashboard
LOG_DATA = True  # Whether to log data


def get_entry_point():
    return "RecoveryAgent"


class RecoveryAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Controller variables"""
        self.current_v = 0.0
        self.current_w = 0.0
        
        """ Perception modules """
        self.segmentation = UnetSegmentation()

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
        self.cameras["Front"] = {
            "active": True,
            "light": 1.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        self.cameras["Left"] = {
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
        self.planner = Planner(initial_pose, spiral_min=3.5, spiral_max=13.5, spiral_step=2.0)

        """ Path planner """
        self.arc_planner = ArcPlanner(arc_config=20, arc_duration=4.0)
        self.path_planner_statistics = {} 
        self.path_planner_statistics["collision detections"] = [] # frame number and current pose
        self.path_planner_statistics["planner_failure"] = []
        self.path_planner_statistics["time taken"] = 0
        self.path_planner_statistics["success"] = False
        self.first_time_stuck = True
        self.success = False
        """ Data logging """
        if LOG_DATA:
            agent_name = get_entry_point()
            self.data_logger = DataLogger(self, agent_name, self.cameras)
            self.path_planner_file = f"output/{agent_name}/{params.DEFAULT_RUN_NAME}/path_planner_stats.pkl"


        if ENABLE_RERUN:
            Rerun.init_vo()
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

    # No longer used:
    def run_nominal_step(self):
        ground_truth_pose = transform_to_numpy(self.get_transform())
        nav_pose = ground_truth_pose

        """ Waypoint navigation """
        waypoint, advanced = self.planner.get_waypoint(nav_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            self.success = True
            self.path_planner_statistics["success"] = True
            
            return carla.VehicleVelocityControl(0.0, 0.0)
        nominal_steering = waypoint_steering(waypoint, nav_pose)

        control = carla.VehicleVelocityControl(0.2, nominal_steering)
        return control

    def run_backup_maneuver(self):
        print("Running backup maneuver")
        frame_rate = params.FRAME_RATE
        self.backup_counter += 1
        if self.backup_counter <= frame_rate * 3:  # Go backwards for 3 seconds
            control = carla.VehicleVelocityControl(-0.2, 0.0)
        elif (
            self.backup_counter <= frame_rate * 5
        ):  # Rotate 90 deg/s for 2 seconds (overcorrecting because it isn't rotating in 1 second)
            control = carla.VehicleVelocityControl(0.0, np.pi / 2)
        elif (
            self.backup_counter <= frame_rate * 7
        ):  # Move in an arc around the rock for 2 seconds (overcorrecting because it isn't rotating in 1 second)
            control = carla.VehicleVelocityControl(0.2, -np.pi / 2)
        else:
            self.backup_counter = 0
            # control = self.run_nominal_step()
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
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
        return is_stuck and self.stuck_counter >= frame_rate  # 1 second

    def run_step(self, input_data):  # This runs at 20 Hz
        if self.step == 0:
            self.initialize()
        # Moved from nominal_step to run_step!!!
        self.step += 1  # Starts at 0 at init, equal to 1 on the first run_step call
        print("\nStep: ", self.step)

        # Obtain and save history of ground truth poses
        ground_truth_pose = transform_to_numpy(self.get_transform())
        nav_pose = ground_truth_pose
        self.gt_poses.append(ground_truth_pose)
        gt_trajectory = np.array([pose[:3, 3] for pose in self.gt_poses])

        if ENABLE_RERUN:
            Rerun.log_3d_trajectory(self.step, gt_trajectory, trajectory_string="ground_truth", color=[0, 0, 255])
            # Rerun.log_2d_seq_scalar("ground_truth_pose/x", self.step, ground_truth_pose[0, 3])
            # Rerun.log_2d_seq_scalar("ground_truth_pose/y", self.step, ground_truth_pose[1, 3])
            # Rerun.log_2d_seq_scalar("ground_truth_pose/z", self.step, ground_truth_pose[2, 3])

        # Obtain velocity estimate from ground truth poses
        rov_vel = np.zeros(3)
        print("rov_vel: ", rov_vel)
        if self.step > 1:
            dt = params.DT
            rov_vel = (gt_trajectory[-1] - gt_trajectory[-2]) / dt
            print("rov_vel: ", rov_vel)
            print("rov_vel[0]: ", rov_vel[0])
            if ENABLE_RERUN:
                Rerun.log_2d_seq_scalar("trajectory_error/vx", self.step, rov_vel[0])
                Rerun.log_2d_seq_scalar("trajectory_error/vy", self.step, rov_vel[1])
                Rerun.log_2d_seq_scalar("trajectory_error/vz", self.step, rov_vel[2])
                Rerun.log_2d_seq_scalar("trajectory_error/v", self.step, np.linalg.norm(rov_vel))

        """ Waypoint navigation """
        waypoint, advanced = self.planner.get_waypoint(nav_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        """ Rock segmentation """
        if self.image_available():
            # if self.step % (params.FRAME_RATE // IMAGE_PROCESS_RATE) == 0:  # This runs at 1 Hz
            FL_gray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
            FR_gray = input_data["Grayscale"][carla.SensorPosition.FrontRight]

            # Run segmentation
            left_seg_masks, left_seg_full_mask = self.segmentation.segment_rocks(FL_gray)
            right_seg_masks, right_seg_full_mask = self.segmentation.segment_rocks(FR_gray)

            # Stereo rock depth
            stereo_depth_results = stereo_depth_from_segmentation(
                left_seg_masks, right_seg_masks, params.STEREO_BASELINE, params.FL_X
            )
            rock_coords = compute_rock_coords_rover_frame(stereo_depth_results, self.cameras)
            rock_radii = compute_rock_radii(stereo_depth_results)

            # Path planning
            control, path, waypoint_local = self.arc_planner.plan_arc(waypoint, nav_pose, rock_coords, rock_radii)
            if control is None:
                control = self.run_backup_maneuver()
                self.path_planner_statistics["planner_failure"].append((self.step, ground_truth_pose))
                self.mission_complete()  # For now, end the mission, but in reality we probably want some tolerance
                return carla.VehicleVelocityControl(0.0, 0.0) 
            self.current_v, self.current_w = control
            print(f"Control: linear = {self.current_v}, angular = {self.current_w}")
            print(f"Waypoint_local: {waypoint_local}")

            if LOG_DATA:
                self.data_logger.log_images(self.step, input_data)

            if DISPLAY_IMAGES:
                overlay = overlay_mask(FL_gray, left_seg_full_mask, color=(0, 0, 1))
                overlay = draw_steering_arc(overlay, self.current_w, color=(255, 0, 0))
                overlay = overlay_stereo_rock_depths(overlay, stereo_depth_results)
                cv.imshow("Rock segmentation", overlay)
                cv.waitKey(1)

            """ Rerun visualization """
            if ENABLE_RERUN:
                gt_trajectory = np.array([pose[:3, 3] for pose in self.gt_poses])
                Rerun.log_3d_trajectory(
                    self.step, gt_trajectory, trajectory_string="ground_truth", color=[0, 120, 255]
                )
                print(f"path: {path.shape}")
                Rerun.log_2d_trajectory(topic="/local/path", frame_id=self.step, trajectory=path)
                if len(rock_coords) > 0:
                    # TODO: crop rocks within certain bounds
                    rock_centers = np.array(rock_coords)[:, :2]
                    print(f"Rock centers: {rock_centers.shape}")
                    Rerun.log_2d_obstacle_map(
                        topic="/local/obstacles",
                        frame_id=self.step,
                        centers=rock_centers,
                        radii=rock_radii,
                    )
            
        """ Control """
        if self.step < 100:  # Wait for arms to raise before moving
            carla_control = carla.VehicleVelocityControl(0.0, 0.0)
        # If agent is stuck, perform backup maneuver
        elif self.backup_counter > 0 or self.check_stuck(rov_vel):
            print("Agent is stuck.")
            if self.first_time_stuck:
                self.path_planner_statistics["collision detections"].append((self.step, ground_truth_pose))
                self.first_time_stuck = False
            carla_control = self.run_backup_maneuver()
        else:
            carla_control = carla.VehicleVelocityControl(self.current_v, self.current_w)
            self.first_time_stuck = True

        """ Data logging """
        if LOG_DATA:
            self.data_logger.log_data(self.step, carla_control)

        print("\n-----------------------------------------------")

        return carla_control

    def finalize(self):
        print("Running finalize")
        self.path_planner_statistics["time taken"] = self.step

        with open(self.path_planner_file, 'wb') as file:
            pickle.dump(self.path_planner_statistics, file)

            print("Dictionary saved to my_dict.pkl")

        if LOG_DATA:
            self.data_logger.save_log()

        if DISPLAY_IMAGES:
            cv.destroyAllWindows()
