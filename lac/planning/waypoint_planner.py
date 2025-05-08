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

WAYPOINT_TIMEOUT = 5000  # [steps] timeout to give up on a waypoint


class WaypointPlanner:
    def __init__(self, initial_pose: np.ndarray):
        # self.waypoints = gen_spiral(initial_pose, spiral_min, spiral_max, spiral_step, repeat)
        self.waypoints = gen_loops(initial_pose, extra_closure=True)
        # self.waypoints = gen_triangle_loops(initial_pose)
        self.waypoint_idx = 0
        self.last_waypoint_step = 0

    def get_waypoint(
        self, step: int, pose: np.ndarray, print_progress: bool = False
    ) -> np.ndarray | None:
        """Get the next waypoint for the agent to follow.

        Returns None if all waypoints have been reached. TODO: handle this better

        """
        advanced = False
        waypoint = self.waypoints[self.waypoint_idx]
        xy_position = pose[:2, 3]

        # Check waypoint timeout
        if step - self.last_waypoint_step > WAYPOINT_TIMEOUT:
            print(f"WAYPOINT TIMEOUT ON {self.waypoint_idx + 1}/{len(self.waypoints)}")
            advanced = True

        # Check if the waypoint has been reached
        if np.linalg.norm(xy_position - waypoint) < WAYPOINT_REACHED_DIST_THRESHOLD:
            advanced = True

        if advanced:
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.waypoints):  # Finished the waypoints
                self.waypoint_idx = 0
                return None, True
            waypoint = self.waypoints[self.waypoint_idx]
            self.last_waypoint_step = step

        if print_progress:
            print(f"Waypoint {self.waypoint_idx + 1}/{len(self.waypoints)}: {waypoint}")

        return waypoint, advanced


def get_starting_direction_order(initial_pose):
    signs = np.sign(initial_pose[:2, 3])
    start_index = np.argwhere((DEFAULT_ORDER == signs).all(axis=1)).flatten()[0]
    return np.roll(DEFAULT_ORDER, -start_index, axis=0), start_index


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
    direction_order, _ = get_starting_direction_order(initial_pose)

    while r <= max_val + 1e-8:
        # Compute the four corners of the current square ring
        corners = r * direction_order

        # Add corners in order
        points.extend(corners)

        # Repeat the last `repeat` waypoints
        if repeat > 0:
            points.extend(corners[:repeat])
            direction_order = np.roll(direction_order, -repeat, axis=0)

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


def gen_loops(initial_pose: np.ndarray, loop_width: float = 7.0, extra_closure: bool = False):
    """
    Generate an Nx2 numpy array of 2D coordinates following a flower pattern.

    Parameters:
      initial_pose (np.array): The initial pose of the rover.
      max_val (float): The half-side length of the outermost square.
      min_val (float): The half-side length of the innermost square.
      step (float): The decrement between successive squares.

    Returns:
      np.array: An (N x 2) numpy array containing the 2D coordinates.
    """
    W = loop_width / 2  # half-width of a square loop

    points = []

    # Add center loop around lander with order based on initial pose
    direction_order, start_index = get_starting_direction_order(initial_pose)
    center_loop = W * direction_order
    points.append(center_loop)

    if extra_closure:
        points.append(center_loop[:2])

    # Corner loops
    # Top-left
    petal_1 = np.array([[-W, W], [-W, 3 * W], [-3 * W, 3 * W], [-3 * W, W], [-W, W]])
    # Top-right
    petal_2 = np.array([[W, W], [3 * W, W], [3 * W, 3 * W], [W, 3 * W], [W, W]])
    # Bottom-right
    petal_3 = np.array([[W, -W], [W, -3 * W], [3 * W, -3 * W], [3 * W, -W], [W, -W]])
    # Bottom-left
    petal_4 = np.array([[-W, -W], [-3 * W, -W], [-3 * W, -3 * W], [-W, -3 * W], [-W, -W]])
    # Concatenate petals

    petals = np.array([petal_1, petal_2, petal_3, petal_4])
    shift = -start_index
    if extra_closure:
        shift -= 2
    petals = np.roll(petals, shift, axis=0)
    petals = np.concatenate(petals, axis=0)
    points.append(petals)

    if extra_closure:
        points.append(center_loop[2])

    points = np.vstack(points)
    return points


def gen_triangle_loops(initial_pose: np.ndarray, loop_width: float = 7.0):
    """ """
    W = loop_width

    points = []

    # Add center loop around lander with order based on initial pose
    direction_order, start_index = get_starting_direction_order(initial_pose)
    center_loop = (W / 2) * direction_order
    points.append(center_loop)  # [0, 1, 2, 3]
    points.append(center_loop[:2])  # [0, 1]

    # Side points for (+,+) quadrant (top-right), which is the first side to be added if start index is 0
    side_points = W * np.array([[0, -1], [1, -1], [1, 0], [0, -1]])
    R = np.array([[0, 1], [-1, 0]])  # 90-deg clockwise rotation
    # Rotate the side points based on start index
    for j in range(start_index):
        side_points = side_points @ R.T

    for i in range(4):
        # Get the corner point in the current quadrant (offset by 1)
        corner = center_loop[(i + 1) % 4]
        x, y = np.sign(corner)
        # Corner
        corner_points = W * np.array([[x, 0], [x, y], [0, 0], [0, y], [x, y], [0, 0]])
        points.append(corner + corner_points)

        # Side
        points.append(corner + side_points)
        side_points = side_points @ R.T

    points = np.vstack(points)
    return points


def gen_loops_lander_lc(initial_pose):
    """
    Same as 5 loops above, but with stop and turn to loop at the lander at each corner.
    """
    pass
