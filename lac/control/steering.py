"""Controller module for the LAC challenge."""

import numpy as np
import cv2 as cv

from lac.perception.depth import project_pixel_to_rover
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
