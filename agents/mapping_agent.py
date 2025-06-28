#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Mapping agent

"""

import carla
import cv2
import numpy as np
from pynput import keyboard
import signal

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.util import transform_to_numpy
from lac.perception.segmentation import UnetSegmentation, SemanticClasses
from lac.planning.waypoint_planner import WaypointPlanner
from lac.slam.visual_odometry import StereoVisualOdometry
from lac.slam.feature_tracker import FeatureTracker
from lac.control.steering import waypoint_steering
from lac.mapping.mapper import Mapper
from lac.utils.data_logger import DataLogger
from lac.utils.visualization import (
    overlay_mask,
    draw_steering_arc,
)
from lac.utils.rerun_interface import Rerun
import lac.params as params

""" Agent parameters and settings """
STOP_INTERVAL = 1e7  # Number of steps to stop and reset the velocity
STOP_DURATION_STEPS = 40  # Number of steps to stop for
USE_GROUND_TRUTH_NAV = False  # Whether to use ground truth pose for navigation

ARM_RAISE_WAIT_FRAMES = 80  # Number of frames to wait for the arms to raise

DISPLAY_IMAGES = True  # Whether to display the camera views
TELEOP = False  # Whether to use teleop control or autonomous control
LOG_DATA = True  # Whether to log data


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
        self.cameras["FrontRight"] = {
            "active": True,
            "light": 1.0,
            "width": 1280,
            "height": 720,
            "semantic": False,
        }
        self.active_cameras = [cam for cam, config in self.cameras.items() if config["active"]]

        """ Perception modules """
        self.segmentation = UnetSegmentation()

        """ Planner """
        self.initial_pose = transform_to_numpy(self.get_initial_position())
        self.lander_pose = self.initial_pose @ transform_to_numpy(
            self.get_initial_lander_position()
        )
        self.planner = WaypointPlanner(self.initial_pose)

        """ State variables """
        self.current_pose = self.initial_pose

        """ SLAM """
        self.svo = StereoVisualOdometry(self.cameras)
        self.svo_poses = [self.initial_pose]
        self.feature_tracker = FeatureTracker(self.cameras)
        self.ground_points = []
        self.rock_detections = {}
        self.rock_points = []
        self.lander_points = []

        """ Mapping """
        self.mapper = Mapper(self.get_geometric_map())

        """ Data logging """
        if LOG_DATA:
            agent_name = get_entry_point()
            self.data_logger = DataLogger(self, agent_name, self.cameras)

        Rerun.init_vo()
        self.gt_poses = [self.initial_pose]

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
        self.current_pose = ground_truth_pose

        """ Get current waypoint """
        waypoint, _ = self.planner.get_waypoint(self.current_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)
        nominal_steering = waypoint_steering(waypoint, self.current_pose)

        """ Image processing """
        if self.image_available():
            FL_gray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
            FR_gray = input_data["Grayscale"][carla.SensorPosition.FrontRight]

            if self.step >= ARM_RAISE_WAIT_FRAMES:
                # Run segmentation
                left_seg_masks, left_labels, left_pred = self.segmentation.segment_rocks(
                    FL_gray, output_pred=True
                )
                right_seg_masks, right_labels = self.segmentation.segment_rocks(FR_gray)
                left_full_mask = np.clip(left_labels, 0, 1).astype(np.uint8)

                # Feature matching depth
                feats_left, feats_right, matches, depths = self.feature_tracker.process_stereo(
                    FL_gray, FR_gray, return_matched_feats=True
                )

                # Extract ground points for height mapping
                kps_left = feats_left["keypoints"][0].cpu().numpy()
                ground_idxs = []
                rock_idxs = []
                lander_idxs = []
                for i, kp in enumerate(kps_left):
                    pred_class = left_pred[int(kp[1]), int(kp[0])]
                    if pred_class == SemanticClasses.ROCK.value:
                        rock_idxs.append(i)
                    elif pred_class == SemanticClasses.GROUND.value:
                        ground_idxs.append(i)
                    elif pred_class == SemanticClasses.LANDER.value:
                        lander_idxs.append(i)
                ground_kps = kps_left[ground_idxs]
                ground_depths = depths[ground_idxs]
                ground_points_world = self.feature_tracker.project_stereo(
                    self.current_pose, ground_kps, ground_depths
                )
                self.ground_points.append(ground_points_world)

                rock_kps = kps_left[rock_idxs]
                rock_depths = depths[rock_idxs]
                rock_points_world = self.feature_tracker.project_stereo(
                    self.current_pose, rock_kps, rock_depths
                )
                self.rock_points.append(rock_points_world)

                lander_kps = kps_left[lander_idxs]
                lander_depths = depths[lander_idxs]
                lander_points_world = self.feature_tracker.project_stereo(
                    self.current_pose, lander_kps, lander_depths
                )
                self.lander_points.append(lander_points_world)

                if DISPLAY_IMAGES:
                    overlay = overlay_mask(FL_gray, left_full_mask, color=(0, 0, 1))
                    overlay = draw_steering_arc(overlay, self.current_w, color=(255, 0, 0))
                    # overlay = overlay_stereo_rock_depths(overlay, stereo_depth_results)
                    cv2.imshow("Rock segmentation", overlay)
                    cv2.waitKey(1)

                """ Rerun visualization """
                Rerun.log_img(FL_gray)
                Rerun.log_3d_points(
                    np.concatenate(self.ground_points, axis=0),
                    topic="/world/ground_points",
                    color=[0, 0, 255],
                )
                Rerun.log_3d_points(
                    np.concatenate(self.lander_points, axis=0),
                    topic="/world/lander_points",
                    color=[0, 255, 0],
                )
                Rerun.log_3d_points(
                    np.concatenate(self.rock_points, axis=0),
                    topic="/world/rock_points",
                    color=[255, 0, 0],
                )

            if LOG_DATA:
                self.data_logger.log_images(self.step, input_data)

        """ Rerun visualization """
        gt_trajectory = np.array([pose[:3, 3] for pose in self.gt_poses])
        Rerun.log_3d_trajectory(
            self.step, gt_trajectory, trajectory_string="ground_truth", color=[0, 120, 255]
        )

        if TELEOP:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        else:
            if self.step < ARM_RAISE_WAIT_FRAMES:  # Wait for arms to raise before moving
                control = carla.VehicleVelocityControl(0.0, 0.0)
            else:
                control = carla.VehicleVelocityControl(params.TARGET_SPEED, nominal_steering)

        """ Data logging """
        if LOG_DATA:
            self.data_logger.log_data(self.step, control)
        print("\n-----------------------------------------------")

        return control

    def finalize(self):
        print("Running finalize")

        if LOG_DATA:
            self.data_logger.save_log()

        """In the finalize method, we should clear up anything we've previously initialized that might be taking up memory or resources.
        In this case, we should close the OpenCV window."""
        cv2.destroyAllWindows()

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
            cv2.destroyAllWindows()
