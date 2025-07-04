#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Full agent

"""

import carla
import numpy as np
import cv2 as cv
import signal

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
from lac.planning.arc_planner import ArcPlanner
from lac.planning.waypoint_planner import WaypointPlanner
from lac.utils.visualization import (
    overlay_mask,
    draw_steering_arc,
    overlay_stereo_rock_depths,
)
from lac.utils.rerun_interface import Rerun
from lac.utils.data_logger import DataLogger
import lac.params as params


""" Agent parameters and settings """
TARGET_SPEED = 0.15  # [m/s]
IMAGE_PROCESS_RATE = 10  # [Hz]

ARM_RAISE_WAIT_FRAMES = 80

DISPLAY_IMAGES = True  # Whether to display the camera views
LOG_DATA = True  # Whether to log data


def get_entry_point():
    return "AutoAgent"


class AutoAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Controller variables"""
        self.current_v = 0.0
        self.current_w = 0.0

        """ Perception modules """
        self.segmentation = UnetSegmentation()

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
        self.initial_pose = transform_to_numpy(self.get_initial_position())
        self.lander_pose = self.initial_pose @ transform_to_numpy(
            self.get_initial_lander_position()
        )
        self.planner = WaypointPlanner(self.initial_pose)

        """ Path planner """
        self.arc_planner = ArcPlanner()

        """ Data logging """
        if LOG_DATA:
            agent_name = get_entry_point()
            self.data_logger = DataLogger(self, agent_name, self.cameras)
            self.ekf_result_file = f"output/{agent_name}/{params.DEFAULT_RUN_NAME}/ekf_result.npz"
            self.rock_detections_file = (
                f"output/{agent_name}/{params.DEFAULT_RUN_NAME}/rock_detections.json"
            )
        Rerun.init_vo()
        self.gt_poses = [self.initial_pose]

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

    def run_step(self, input_data):  # This runs at 20 Hz
        if self.step == 0:
            self.initialize()
        self.step += 1  # Starts at 0 at init, equal to 1 on the first run_step call
        print("\nStep: ", self.step)

        ground_truth_pose = transform_to_numpy(self.get_transform())
        self.gt_poses.append(ground_truth_pose)
        nav_pose = ground_truth_pose

        """ Waypoint navigation """
        waypoint, advanced = self.planner.get_waypoint(nav_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        """ Rock segmentation """
        if self.image_available():
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
            control, path, waypoint_local = self.arc_planner.plan_arc(
                waypoint, nav_pose, rock_coords, rock_radii
            )
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
            print(Rerun.blueprint)

        if self.step < ARM_RAISE_WAIT_FRAMES:  # Wait for arms to raise before moving
            control = carla.VehicleVelocityControl(0.0, 0.0)
        else:
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
