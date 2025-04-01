"""Controller module for the LAC challenge."""

from math import pi
import numpy as np
import cv2 as cv

from lac.perception.depth import project_pixel_to_rover
from lac.control.dynamics import arc, dubins_traj
from lac.utils.frames import invert_transform_mat, apply_transform
from lac.util import mask_centroid, wrap_angle, pose_to_pos_rpy
import lac.params as params


def waypoint_steering(waypoint: np.ndarray, current_pose: np.ndarray) -> float:
    """Compute steering to point to a waypoint."""
    pos, rpy = pose_to_pos_rpy(current_pose)
    angle_to_waypoint = np.arctan2(waypoint[1] - pos[1], waypoint[0] - pos[0])
    angle_diff = wrap_angle(angle_to_waypoint - rpy[2])  # [rad]
    steering = np.clip(params.KP_STEER * angle_diff, -params.MAX_STEER, params.MAX_STEER)
    return steering


def segmentation_steering(masks: list[np.ndarray]) -> float:
    """Compute a steering delta based on segmentation results to avoid rocks.

    TODO: account for offset due to operating on left camera
    """
    max_area = 0
    max_mask = None
    for mask in masks:
        mask_area = np.sum(mask)
        if mask_area > max_area:
            max_area = mask_area
            max_mask = mask
    steer_delta = 0
    if max_mask is not None and max_area > params.ROCK_MASK_AVOID_MIN_AREA:
        max_mask = max_mask.astype(np.uint8)
        cx, _ = mask_centroid(max_mask)
        x, _, w, _ = cv.boundingRect(max_mask)
        offset = params.IMG_WIDTH / 2 - cx
        if offset > 0:  # Turn right
            steer_delta = -min(
                params.MAX_STEER_DELTA * ((x + w) - cx) / 100, params.MAX_STEER_DELTA
            )
        else:  # Turn left
            steer_delta = min(params.MAX_STEER_DELTA * (cx - x) / 100, params.MAX_STEER_DELTA)
    return steer_delta


def rock_avoidance_steering(depth_results: dict, cam_config: dict) -> float:
    """Compute a steering delta based on segmentation results to avoid rocks."""
    ROCK_AVOID_DIST = 4.0
    K_AVOID = 0.6

    rock_points_rover_frame = []
    mask_areas = []
    distances = []
    for rock in depth_results:
        rock_point_rover_frame = project_pixel_to_rover(
            rock["left_centroid"], rock["depth"], "FrontLeft", cam_config
        )
        distance = np.linalg.norm(rock_point_rover_frame)
        if distance < ROCK_AVOID_DIST:
            rock_points_rover_frame.append(rock_point_rover_frame)
            mask_areas.append(rock["left_mask"].sum())
            distances.append(distance)

    if len(mask_areas) == 0:
        return 0.0

    max_mask_area_idx = np.argmax(mask_areas)
    if mask_areas[max_mask_area_idx] < params.ROCK_MASK_AVOID_MIN_AREA:
        return 0.0

    rock_point = rock_points_rover_frame[max_mask_area_idx]
    distance = distances[max_mask_area_idx]
    if distance > ROCK_AVOID_DIST:
        return 0.0
    else:
        MAX_STEER_DELTA = 0.8
        steer_mag = min(MAX_STEER_DELTA, K_AVOID * (ROCK_AVOID_DIST - distance) ** 2)
        # steer_mag = K_AVOID * (ROCK_AVOID_DIST - distance) ** 2
        return -np.sign(rock_point[1]) * steer_mag


class ArcPlanner:
    """Arc planner"""

    def __init__(self):
        NUM_OMEGAS = 5
        MAX_OMEGA = 1  # [rad/s]
        ARC_DURATION = 2.0  # [s]
        NUM_ARC_POINTS = int(ARC_DURATION / params.DT)
        self.speeds = [params.TARGET_SPEED]  # [0.05, 0.1, 0.15, 0.2]  # [m/s]
        self.omegas = np.linspace(-MAX_OMEGA, MAX_OMEGA, NUM_OMEGAS)
        self.root_arcs = []
        self.candidate_arcs = []
        self.root_vw = []
        self.vw = []
        for v in self.speeds:
            for w in self.omegas:
                new_arc = dubins_traj(np.zeros(3), [v, w], NUM_ARC_POINTS, params.DT)
                self.root_arcs.append(new_arc)
                self.candidate_arcs.append(new_arc)
                self.root_vw.append((v, w))

        concatenated_arcs = []
        for count, root_arc in enumerate(self.root_arcs):
            last_state = root_arc[-1]  # Extract last state [x, y, theta]

            for v in self.speeds:
                for w in self.omegas:
                    new_arc = dubins_traj(last_state, [v, w], NUM_ARC_POINTS, params.DT)
                    concatenated_arcs.append(np.concatenate((root_arc, new_arc)))
                    self.vw.append(self.root_vw[count])

        self.candidate_arcs = concatenated_arcs
        self.np_candidate_arcs = np.array(self.candidate_arcs)

    def plan_arc(
        self,
        waypoint_global: np.ndarray,
        current_pose: np.ndarray,
        rock_coords: np.ndarray,
        rock_radii: list,
    ):
        """Plan an arc to a waypoint while avoiding rocks and lander

        Given rock coordinates and radii in rover local frame, select the path with lowest cost
        (distance to waypoint) that also avoids rocks.

        """
        # Transform global waypoint to local frame
        pose_inv = invert_transform_mat(current_pose)
        waypoint_local = pose_inv @ np.array([waypoint_global[0], waypoint_global[1], 0.0, 1.0])

        # Transform lander global position to rover local frame
        lander_local = apply_transform(pose_inv, params.LANDER_GLOBAL)

        # Obtain bounding box of lander
        # TODO: check against oriented lander box (instead of axis-aligned overbounding box)
        min_x = np.min(lander_local[:, 0])
        max_x = np.max(lander_local[:, 0])
        min_y = np.min(lander_local[:, 1])
        max_y = np.max(lander_local[:, 1])

        lander_bbox = np.array([min_x, max_x, min_y, max_y])

        # Distance to waypoint cost
        path_costs = np.linalg.norm(
            self.np_candidate_arcs[:, -1, :2] - (waypoint_local[:2]), axis=1
        )

        sorted_indices = np.argsort(path_costs)

        for i in sorted_indices:
            arc = self.np_candidate_arcs[i]
            valid = True
            for j in range(len(arc)):
                # Check in arc is inside lander's bounding box
                if (
                    # X bounds
                    arc[j][0] >= lander_bbox[0]
                    and arc[j][0] <= lander_bbox[1]
                    # Y bounds
                    and arc[j][1] >= lander_bbox[2]
                    and arc[j][1] <= lander_bbox[3]
                ):
                    path_costs[i] += 1000
                    valid = False
                    break
                # Check if arc is inside any rocks
                for rock, radius in zip(rock_coords, rock_radii):
                    if radius > params.ROCK_MIN_RADIUS:
                        if np.linalg.norm(arc[j][:2] - rock[:2]) - params.ROVER_RADIUS <= radius:
                            path_costs[i] += 1000
                            valid = False
                            break
            if valid:
                return self.vw[i], arc, waypoint_local

        # TODO: handle case if no paths are valid
        return None, None, None
