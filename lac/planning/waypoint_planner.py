"""Planner class

Generates waypoints for the agent to follow, and tracks the agent's progress.
 - TODO: modify waypoints generation based on starting location

"""

import numpy as np

from lac.params import WAYPOINT_REACHED_DIST_THRESHOLD

SPIRAL_MAX = 13.5  # [m]
SPIRAL_MIN = 3.5  # [m]
SPIRAL_STEP = 2.0  # [m]

# Clockwise order: top-left, top-right, bottom-right, bottom-left
DEFAULT_ORDER = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])


class WaypointPlanner:
    def __init__(
        self,
        initial_pose: np.ndarray,
        spiral_min: float = SPIRAL_MIN,
        spiral_max: float = SPIRAL_MAX,
        spiral_step: float = SPIRAL_STEP,
        repeat: int = 1,
    ):
        self.waypoints = gen_spiral(initial_pose, spiral_min, spiral_max, spiral_step, repeat)
        # self.waypoints = self.waypoints[:3]  # get the first two waypoints only
        self.waypoint_idx = 0

    def get_waypoint(self, pose: np.ndarray, print_progress: bool = False) -> np.ndarray | None:
        """Get the next waypoint for the agent to follow.

        Returns None if all waypoints have been reached. TODO: handle this better

        """
        advanced = False
        waypoint = self.waypoints[self.waypoint_idx]
        xy_position = pose[:2, 3]
        if np.linalg.norm(xy_position - waypoint) < WAYPOINT_REACHED_DIST_THRESHOLD:
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.waypoints):  # Finished the waypoints
                self.waypoint_idx = 0
                return None, True
            waypoint = self.waypoints[self.waypoint_idx]
            advanced = True
        if print_progress:
            print(f"Waypoint {self.waypoint_idx + 1}/{len(self.waypoints)}: {waypoint}")
        return waypoint, advanced


def get_starting_direction_order(initial_pose):
    signs = np.sign(initial_pose[:2, 3])
    start_index = np.argwhere((DEFAULT_ORDER == signs).all(axis=1)).flatten()[0]
    return np.roll(DEFAULT_ORDER, -start_index, axis=0)


def gen_spiral(
    initial_pose: np.ndarray,
    min_val: float = SPIRAL_MIN,
    max_val: float = SPIRAL_MAX,
    step: float = SPIRAL_STEP,
    repeat: int = 0,
):
    """
    Generate an Nx2 numpy array of 2D coordinates following a square spiral,
    ensuring that the last waypoint is repeated, and the next ring starts at
    the next diagonal position.

    Parameters:
      initial_pose (np.array): The initial pose of the rover.
      max_val (float): The half-side length of the outermost square.
      min_val (float): The half-side length of the innermost square.
      step (float): The decrement between successive squares.

    Returns:
      np.array: An (N x 2) numpy array containing the 2D coordinates.
    """
    points = []
    r = min_val
    direction_order = get_starting_direction_order(initial_pose)

    while r <= max_val + 1e-8:
        # Compute the four corners of the current square ring
        corners = r * direction_order

        # Add corners in order
        points.extend(corners)

        # # Repeat the first waypoint
        # points.append(corners[0])
        # Repeat the last `repeat` waypoints
        if repeat > 0:
            points.extend(corners[:repeat])
            direction_order = np.roll(direction_order, -repeat, axis=0)

        # Cycle the direction
        # direction_order = np.roll(direction_order, -1, axis=0)
        r += step

    return np.array(points)


def gen_square_spiral(max_val, min_val, step):
    """
    Generate an Nx2 numpy array of 2D coordinates following a square spiral.

    Parameters:
      initial_pose (np.array): The initial pose of the rover.
      max_val (float): The half-side length of the outermost square.
      min_val (float): The half-side length of the innermost square.
      step (float): The decrement between successive squares.

    Returns:
      np.array: An (N x 2) numpy array containing the 2D coordinates.
    """
    points = []
    r = max_val
    # Use a small tolerance to account for floating point comparisons.
    while r >= min_val - 1e-8:
        # Order: top-left, top-right, bottom-right, bottom-left.
        points.append([-r, r])  # top-left
        points.append([r, r])  # top-right
        points.append([r, -r])  # bottom-right
        points.append([-r, -r])  # bottom-left
        r -= step
    return np.array(points)


def gen_square_spiral_inside_out(min_val, max_val, step):
    """
    Generate an Nx2 numpy array of 2D coordinates following a square spiral.

    Parameters:
      initial_pose (np.array): The initial pose of the rover.
      max_val (float): The half-side length of the outermost square.
      min_val (float): The half-side length of the innermost square.
      step (float): The decrement between successive squares.

    Returns:
      np.array: An (N x 2) numpy array containing the 2D coordinates.
    """
    points = []
    r = min_val
    # Use a small tolerance to account for floating point comparisons.
    while r <= max_val + 1e-8:
        # Order: top-left, top-right, bottom-right, bottom-left.
        points.append([-r, r])  # top-left
        points.append([r, r])  # top-right
        points.append([r, -r])  # bottom-right
        points.append([-r, -r])  # bottom-left
        r += step
    return np.array(points)
