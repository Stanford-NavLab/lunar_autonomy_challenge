#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Full agent

"""

import carla
import cv2 as cv
import numpy as np
import signal

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.util import (
    pose_to_pos_rpy,
    transform_to_numpy,
    transform_to_pos_rpy,
)
from lac.perception.segmentation import UnetSegmentation, SemanticClasses
from lac.perception.depth import (
    stereo_depth_from_segmentation,
    project_pixel_to_world,
    compute_rock_coords_rover_frame,
    compute_rock_radii,
)
from lac.slam.visual_odometry import StereoVisualOdometry
from lac.slam.feature_tracker import FeatureTracker
from lac.control.controller import ArcPlanner
from lac.planning.planner import Planner
from lac.mapping.mapper import bin_points_to_grid, interpolate_heights
from lac.utils.visualization import (
    overlay_mask,
    draw_steering_arc,
    overlay_stereo_rock_depths,
)
from lac.utils.frames import apply_transform
from lac.utils.data_logger import DataLogger
import lac.params as params


""" Agent parameters and settings """
EVAL = True  # Whether running in evaluation mode (disable ground truth)
USE_FIDUCIALS = False
BACK_CAMERAS = True

EARLY_STOP_STEP = 0  # Number of steps before stopping the mission (0 for no early stop)
USE_GROUND_TRUTH_NAV = False  # Whether to use ground truth pose for navigation
ARM_RAISE_WAIT_FRAMES = 80  # Number of frames to wait for the arms to raise

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
        if BACK_CAMERAS:
            self.cameras["BackLeft"] = {
                "active": True,
                "light": 1.0,
                "width": 1280,
                "height": 720,
                "semantic": False,
            }
            self.cameras["BackRight"] = {
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
        self.planner = Planner(self.initial_pose, spiral_min=3.5, spiral_max=3.5, spiral_step=1.0)

        """ State variables """
        self.current_pose = self.initial_pose
        self.current_velocity = np.zeros(3)

        """ SLAM """
        self.svo = StereoVisualOdometry(self.cameras)
        self.svo_poses = [self.initial_pose]
        self.feature_tracker = FeatureTracker(self.cameras)
        self.ground_points = []
        self.rock_detections = {}
        self.rock_points = []

        """ Path planner """
        self.arc_planner = ArcPlanner()

        """ Data logging """
        if LOG_DATA:
            agent_name = get_entry_point()
            self.data_logger = DataLogger(self, agent_name, self.cameras)

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signal_received, frame):
        print("\nCtrl+C detected! Exiting mission")
        self.mission_complete()

    def initialize(self):
        # Move the arms out of the way
        self.set_front_arm_angle(params.ARM_ANGLE_STATIC_RAD)
        self.set_back_arm_angle(params.ARM_ANGLE_STATIC_RAD)

        # Initialize the map
        g_map = self.get_geometric_map()
        map_array = g_map.get_map_array()
        map_array[:, :, 2] = self.initial_pose[2, 3]  # Height
        map_array[:, :, 3] = 1.0  # Rocks
        # TODO: clear a patch of no rock around the initial pose and lander
        i, j = g_map.get_cell_indexes(self.initial_pose[0, 3], self.initial_pose[1, 3])
        r = int(params.ROVER_RADIUS / params.CELL_WIDTH)
        map_array[i - r : i + r, j - r : j + r, 3] = 0.0
        i, j = g_map.get_cell_indexes(0, 0)
        r = int(params.LANDER_WIDTH / (2 * params.CELL_WIDTH))
        map_array[i - r : i + r, j - r : j + r, 3] = 0.0

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

    def run_backup_maneuver(self):
        print("Running backup maneuver")
        frame_rate = params.FRAME_RATE
        self.backup_counter += 1
        if self.backup_counter <= frame_rate * 1.5:  # Go backwards for 3 seconds
            control = carla.VehicleVelocityControl(-0.2, 0.0)
        elif (
            self.backup_counter <= frame_rate * 3
        ):  # Rotate 90 deg/s for 1.5 seconds (overcorrecting because it isn't rotating in 1 second)
            control = carla.VehicleVelocityControl(0.0, np.pi / 4)
        elif self.backup_counter <= frame_rate * 9:
            # Go forward for 6 seconds
            control = carla.VehicleVelocityControl(0.2, 0.0)
        else:
            self.backup_counter = 0
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        return control

    def check_stuck(self):
        # Agent is stuck if the velocity is less than 0.1 m/s
        if self.step < ARM_RAISE_WAIT_FRAMES + 10:
            return False
        is_stuck = np.linalg.norm(self.current_velocity) < 0.75 * params.TARGET_SPEED
        if is_stuck:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        return is_stuck and self.stuck_counter >= params.FRAME_RATE  # 1 second

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

        if USE_GROUND_TRUTH_NAV:
            nav_pose = ground_truth_pose
        else:
            nav_pose = self.current_pose

        """ Waypoint navigation """
        waypoint, advanced = self.planner.get_waypoint(nav_pose, print_progress=True)
        if waypoint is None:
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        """ Image processing """
        if self.image_available():
            FL_gray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
            FR_gray = input_data["Grayscale"][carla.SensorPosition.FrontRight]

            if self.step >= ARM_RAISE_WAIT_FRAMES:
                # VO
                if self.step == ARM_RAISE_WAIT_FRAMES:
                    self.svo.initialize(self.current_pose, FL_gray, FR_gray)
                else:
                    self.svo.track(FL_gray, FR_gray)
                self.svo_poses.append(self.svo.get_pose())
                self.current_velocity = (
                    self.svo.get_pose()[:3, 3] - self.current_pose[:3, 3]
                ) / params.DT
                self.current_pose = self.svo.get_pose()

                # Run segmentation
                left_seg_masks, left_labels, left_pred = self.segmentation.segment_rocks(
                    FL_gray, output_pred=True
                )
                right_seg_masks, right_labels = self.segmentation.segment_rocks(FR_gray)
                left_full_mask = np.clip(left_labels, 0, 1).astype(np.uint8)

                # Stereo rock depth
                stereo_depth_results = stereo_depth_from_segmentation(
                    left_seg_masks, right_seg_masks, params.STEREO_BASELINE, params.FL_X
                )
                rock_coords = compute_rock_coords_rover_frame(stereo_depth_results, self.cameras)
                rock_radii = compute_rock_radii(stereo_depth_results)

                # Add points for rock mapping
                if self.step % 20 == 0:
                    rock_points_world = apply_transform(self.current_pose, rock_coords)
                    self.rock_points.append(rock_points_world)

                # Feature matching depth
                feats_left, feats_right, matches, depths = self.feature_tracker.process_stereo(
                    FL_gray, FR_gray, return_matched_feats=True
                )

                # Extract ground points for height mapping
                left_ground_mask = left_pred == SemanticClasses.GROUND.value
                kps_left = feats_left["keypoints"][0].cpu().numpy()
                ground_idxs = []
                for i, kp in enumerate(kps_left):
                    if left_ground_mask[int(kp[1]), int(kp[0])]:
                        ground_idxs.append(i)
                ground_kps = kps_left[ground_idxs]
                ground_depths = depths[ground_idxs]
                ground_points_world = self.feature_tracker.project_stereo(
                    self.current_pose, ground_kps, ground_depths
                )
                self.ground_points.append(ground_points_world)

                if BACK_CAMERAS:
                    BL_gray = input_data["Grayscale"][carla.SensorPosition.BackLeft]
                    BR_gray = input_data["Grayscale"][carla.SensorPosition.BackRight]

                    left_seg_masks, _, left_pred = self.segmentation.segment_rocks(
                        BL_gray, output_pred=True
                    )
                    right_seg_masks, _ = self.segmentation.segment_rocks(BR_gray)

                    back_stereo_depth_results = stereo_depth_from_segmentation(
                        left_seg_masks, right_seg_masks, params.STEREO_BASELINE, params.FL_X
                    )
                    back_rock_coords = compute_rock_coords_rover_frame(
                        back_stereo_depth_results, self.cameras, cam_name="BackLeft"
                    )

                    # Add points for rock mapping
                    if self.step % 20 == 0:
                        rock_points_world = apply_transform(self.current_pose, back_rock_coords)
                        self.rock_points.append(rock_points_world)

                    # Feature matching depth
                    feats_left, feats_right, matches, depths = self.feature_tracker.process_stereo(
                        BL_gray, BR_gray, return_matched_feats=True
                    )

                    # Extract ground points for height mapping
                    left_ground_mask = left_pred == SemanticClasses.GROUND.value
                    kps_left = feats_left["keypoints"][0].cpu().numpy()
                    ground_idxs = []
                    for i, kp in enumerate(kps_left):
                        if left_ground_mask[int(kp[1]), int(kp[0])]:
                            ground_idxs.append(i)
                    ground_kps = kps_left[ground_idxs]
                    ground_depths = depths[ground_idxs]
                    ground_points_world = self.feature_tracker.project_stereo(
                        self.current_pose, ground_kps, ground_depths, cam_name="BackLeft"
                    )
                    self.ground_points.append(ground_points_world)

                # Path planning
                control, path, waypoint_local = self.arc_planner.plan_arc(
                    waypoint, nav_pose, rock_coords, rock_radii
                )
                if control is not None:
                    self.current_v, self.current_w = control
                else:
                    print("No safe paths found!")

                if DISPLAY_IMAGES:
                    overlay = overlay_mask(FL_gray, left_full_mask, color=(0, 0, 1))
                    overlay = draw_steering_arc(overlay, self.current_w, color=(255, 0, 0))
                    overlay = overlay_stereo_rock_depths(overlay, stereo_depth_results)
                    cv.imshow("Rock segmentation", overlay)
                    cv.waitKey(1)

            if LOG_DATA:
                self.data_logger.log_images(self.step, input_data)

        """ Control """
        if self.step < ARM_RAISE_WAIT_FRAMES:  # Wait for arms to raise before moving
            control = carla.VehicleVelocityControl(0.0, 0.0)
        # If agent is stuck, perform backup maneuver
        elif self.backup_counter > 0 or self.check_stuck():
            print("Agent is stuck.")
            control = self.run_backup_maneuver()
        else:
            control = carla.VehicleVelocityControl(self.current_v, self.current_w)
            # control = carla.VehicleVelocityControl(params.TARGET_SPEED, 0.0)  # drive straight

        """ Data logging """
        if LOG_DATA:
            self.data_logger.log_data(self.step, control, self.current_pose)

        print("\n-----------------------------------------------")

        return control

    def finalize(self):
        print("Running finalize")

        # Set the map
        g_map = self.get_geometric_map()
        map_array = g_map.get_map_array()

        # Heights
        if len(self.ground_points) > 0:
            all_ground_points = np.concatenate(self.ground_points, axis=0)
            ground_grid = bin_points_to_grid(all_ground_points)
            map_array[:, :, 2] = ground_grid
            map_array[:] = interpolate_heights(map_array)

            # map_array[:, :, 3] = ground_grid == -np.inf
            map_array[:, :, 3] = 0.0
            rock_points = np.concatenate(self.rock_points, axis=0)
            xmin, xmax = np.min(map_array[:, :, 0]), np.max(map_array[:, :, 0])
            ymin, ymax = np.min(map_array[:, :, 1]), np.max(map_array[:, :, 1])
            nx, ny = map_array.shape[:2]
            for p in rock_points:
                i = int((p[0] - xmin) / (xmax - xmin) * nx)
                j = int((p[1] - ymin) / (ymax - ymin) * ny)
                if 0 <= i < nx and 0 <= j < ny:
                    map_array[i, j, 3] = 1.0

            # if LOG_DATA:
            #     np.save(
            #         f"output/{get_entry_point()}/{params.DEFAULT_RUN_NAME}/ground_points.npy",
            #         all_ground_points,
            #     )

        # Rocks
        # map_array[:, :, 3] = 0.0
        # for id, detections in self.rock_detections.items():
        #     points = np.array(detections["points"])
        #     avg_point = np.median(points, axis=0)
        #     radii = np.array(detections["radii"])
        #     avg_radius = np.median(radii)
        #     g_map.set_rock(avg_point[0], avg_point[1], True)

        if LOG_DATA:
            self.data_logger.save_log()

        if DISPLAY_IMAGES:
            cv.destroyAllWindows()
