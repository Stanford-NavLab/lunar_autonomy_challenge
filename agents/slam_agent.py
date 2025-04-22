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
from lac.perception.depth import (
    stereo_depth_from_segmentation,
    compute_rock_coords_rover_frame,
    compute_rock_radii,
)
from lac.perception.segmentation import UnetSegmentation
from lac.planning.waypoint_planner import Planner
from lac.control.controller import ArcPlanner
from lac.slam.visual_odometry import StereoVisualOdometry
from lac.slam.feature_tracker import FeatureTracker
from lac.slam.slam import SLAM
from lac.control.controller import waypoint_steering
from lac.utils.data_logger import DataLogger
from lac.utils.rerun_interface import Rerun
from lac.utils.visualization import (
    overlay_mask,
    draw_steering_arc,
    overlay_stereo_rock_depths,
)
import lac.params as params

""" Agent parameters and settings """
USE_GROUND_TRUTH_NAV = True  # Whether to use ground truth pose for navigation
ARM_RAISE_WAIT_FRAMES = 100  # Number of frames to wait for the arms to raise
GRAPH_OPTIMIZE_RATE = 500  # Optimize graph every N steps

DISPLAY_IMAGES = True  # Whether to display the camera views
TELEOP = False  # Whether to use teleop control or autonomous control


def get_entry_point():
    return "SlamAgent"


class SlamAgent(AutonomousAgent):
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
        self.lander_pose = self.initial_pose @ transform_to_numpy(self.get_initial_lander_position())
        self.planner = Planner(self.initial_pose)
        self.arc_planner = ArcPlanner()

        """ SLAM """
        self.svo = StereoVisualOdometry(self.cameras)
        self.tracker = FeatureTracker(self.cameras)
        self.segmentation = UnetSegmentation()
        self.graph = SLAM()
        self.graph_idx = 0

        """ Data logging """
        agent_name = get_entry_point()
        self.data_logger = DataLogger(self, agent_name, self.cameras)
        self.ekf_result_file = f"output/{agent_name}/{params.DEFAULT_RUN_NAME}/ekf_result.npz"

        Rerun.init_vo()
        self.gt_poses = [self.initial_pose]
        self.svo_poses = []
        self.current_pose = self.initial_pose

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signal_received, frame):
        print("\nCtrl+C detected! Exiting mission")
        self.mission_complete()

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

        if USE_GROUND_TRUTH_NAV:
            nav_pose = ground_truth_pose
        else:
            nav_pose = self.current_pose

        """ Waypoint navigation """
        waypoint, _ = self.planner.get_waypoint(nav_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)
        # nominal_steering = waypoint_steering(waypoint, nav_pose)

        """ Image processing """
        if self.image_available():
            images_gray = {}
            for cam in self.active_cameras:
                images_gray[cam] = input_data["Grayscale"][getattr(carla.SensorPosition, cam)]

            # Stereo VO
            if self.step >= ARM_RAISE_WAIT_FRAMES:
                if self.step == ARM_RAISE_WAIT_FRAMES:
                    self.svo.initialize(self.initial_pose, images_gray["FrontLeft"], images_gray["FrontRight"])
                    self.tracker.initialize(self.initial_pose, images_gray["FrontLeft"], images_gray["FrontRight"])
                else:
                    self.svo.track(images_gray["FrontLeft"], images_gray["FrontRight"])
                    self.tracker.track_keyframe(self.current_pose, images_gray["FrontLeft"], images_gray["FrontRight"])

                self.graph.add_pose(self.graph_idx, self.initial_pose)
                self.graph.add_vision_factors(self.graph_idx, self.tracker)
                self.graph_idx += 1
                self.svo_poses.append(self.svo.get_pose())
                self.current_pose = self.svo.get_pose()

                # Run segmentation
                left_seg_masks, left_labels, left_pred = self.segmentation.segment_rocks(
                    images_gray["FrontLeft"], output_pred=True
                )
                right_seg_masks, right_labels = self.segmentation.segment_rocks(images_gray["FrontRight"])
                left_full_mask = np.clip(left_labels, 0, 1).astype(np.uint8)

                # Stereo rock depth
                stereo_depth_results = stereo_depth_from_segmentation(
                    left_seg_masks, right_seg_masks, params.STEREO_BASELINE, params.FL_X
                )
                rock_coords = compute_rock_coords_rover_frame(stereo_depth_results, self.cameras)
                rock_radii = compute_rock_radii(stereo_depth_results)

                control, path, waypoint_local = self.arc_planner.plan_arc(waypoint, nav_pose, rock_coords, rock_radii)
                if control is not None:
                    self.current_v, self.current_w = control
                else:
                    print("No safe paths found!")

                # if DISPLAY_IMAGES:
                #     overlay = overlay_mask(
                #         images_gray["FrontLeft"], left_full_mask, color=(0, 0, 1)
                #     )
                #     overlay = draw_steering_arc(overlay, self.current_w, color=(255, 0, 0))
                #     overlay = overlay_stereo_rock_depths(overlay, stereo_depth_results)
                #     cv.imshow(cam, overlay)
                #     cv.waitKey(1)

                overlay = overlay_mask(images_gray["FrontLeft"], left_full_mask, color=(0, 0, 1))
                overlay = draw_steering_arc(overlay, self.current_w, color=(255, 0, 0))
                overlay = overlay_stereo_rock_depths(overlay, stereo_depth_results)
                Rerun.log_img(overlay)

            self.data_logger.log_images(self.step, input_data)

        # if self.step % GRAPH_OPTIMIZE_RATE == 0:
        #     print("Optimizing graph...")
        #     window = list(range(0, self.graph_idx))
        #     self.graph.optimize(window, verbose=True)

        # Rerun logging
        position_error = self.current_pose[:3, 3] - ground_truth_pose[:3, 3]
        gt_trajectory = np.array([pose[:3, 3] for pose in self.gt_poses])
        svo_trajectory = np.array([pose[:3, 3] for pose in self.svo_poses])
        slam_poses = list(self.graph.poses.values())
        slam_trajectory = np.array([pose[:3, 3] for pose in slam_poses])
        Rerun.log_3d_trajectory(self.step, gt_trajectory, trajectory_string="ground_truth", color=[0, 0, 0])
        Rerun.log_3d_trajectory(self.step, svo_trajectory, trajectory_string="visual_odometry", color=[0, 0, 255])
        # Rerun.log_3d_trajectory(
        #     self.step, slam_trajectory, trajectory_string="slam", color=[0, 255, 0]
        # )
        Rerun.log_2d_seq_scalar("trajectory_error/err_x", self.step, position_error[0])
        Rerun.log_2d_seq_scalar("trajectory_error/err_y", self.step, position_error[1])
        Rerun.log_2d_seq_scalar("trajectory_error/err_z", self.step, position_error[2])

        landmark_points = np.array(list(self.graph.landmarks.values()))
        if len(landmark_points) > 0:
            Rerun.log_3d_points(
                landmark_points,
                topic="/world/landmarks",
                color=[255, 165, 0],
            )

        """ Control """
        if self.step < ARM_RAISE_WAIT_FRAMES:
            control = carla.VehicleVelocityControl(0.0, 0.0)
        elif TELEOP:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        else:
            # control = carla.VehicleVelocityControl(params.TARGET_SPEED, nominal_steering)
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        """ Data logging """
        self.data_logger.log_data(self.step, control)
        print("\n-----------------------------------------------")

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
