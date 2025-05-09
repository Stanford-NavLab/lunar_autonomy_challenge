"""Functions for waypoint generation."""

import numpy as np

# Clockwise order: top-left, top-right, bottom-right, bottom-left
DEFAULT_ORDER = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])


def get_starting_direction_order(initial_pose):
    signs = np.sign(initial_pose[:2, 3])
    start_index = np.argwhere((DEFAULT_ORDER == signs).all(axis=1)).flatten()[0]
    return np.roll(DEFAULT_ORDER, -start_index, axis=0), start_index


def gen_spiral(
    initial_pose: np.ndarray,
    min_val: float,
    max_val: float,
    step: float,
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


def gen_five_loops(initial_pose: np.ndarray, loop_width: float = 7.0, extra_closure: bool = False):
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


def gen_nine_loops(initial_pose: np.ndarray, loop_width: float = 7.0):
    """ """
    points = []
    W = loop_width / 2  # half-width of a square loop

    # Generate the waypoints assuming top-left start

    # Center loop
    center_loop = W * DEFAULT_ORDER
    points.append(center_loop)
    points.append(center_loop[:2])

    # Side loops
    points.append(np.array([[3 * W, W], [3 * W, -W], [W, -W], [-W, -W]]))  # Right
    points.append(np.array([[-3 * W, -W], [-3 * W, W], [-W, W], [W, W], [W, -W]]))  # Left
    points.append(np.array([[W, -3 * W], [-W, -3 * W], [-W, -W], [-W, W]]))  # Bottom
    points.append(np.array([[-W, 3 * W], [W, 3 * W], [W, W], [W, -W]]))  # Top

    # Corner loops
    # - Bottom-right
    points.append(np.array([[W, -3 * W], [3 * W, -3 * W], [3 * W, -W], [W, -W], [-W, -W]]))
    # - Bottom-left
    points.append(np.array([[-3 * W, -W], [-3 * W, -3 * W], [-W, -3 * W], [-W, -W], [-W, W]]))
    # - Top-left
    points.append(np.array([[-W, 3 * W], [-3 * W, 3 * W], [-3 * W, W], [-W, W], [W, W]]))
    # - Top-right
    points.append(np.array([[3 * W, W], [3 * W, 3 * W], [W, 3 * W], [W, W], [W, -W]]))

    points = np.vstack(points)

    # Rotate the whole trajectory based on the starting index
    _, start_index = get_starting_direction_order(initial_pose)
    R = np.array([[0, 1], [-1, 0]])  # 90-deg clockwise rotation
    for j in range(start_index):
        points = points @ R.T

    return points


def gen_triangle_loops(
    initial_pose: np.ndarray, loop_width: float = 7.0, additional_loops: bool = False
):
    W = loop_width

    points = []

    # Add center loop around lander with order based on initial pose
    direction_order, start_index = get_starting_direction_order(initial_pose)
    center_loop = (W / 2) * direction_order
    points.append(center_loop)  # [0, 1, 2, 3]
    points.append(center_loop[:2])  # [0, 1]

    # Side points for (+,+) quadrant (top-right), which is the first side to be added if start index is 0
    if additional_loops:
        side_points = W * np.array([[0, -1], [1, -1], [0, 0], [1, 0], [1, -1], [0, 0], [0, -1]])
    else:
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
