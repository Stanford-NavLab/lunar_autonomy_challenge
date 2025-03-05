"""Planner class

Generates waypoints for the agent to follow, and tracks the agent's progress.
 - TODO: modify waypoints generation based on starting location

"""

import numpy as np

from lac.util import gen_square_spiral
from lac.params import WAYPOINT_REACHED_DIST_THRESHOLD

SPIRAL_MAX = 4.25  # [m]
SPIRAL_MIN = 2.75  # [m]
SPIRAL_STEP = 0.5  # [m]


class Planner:
    def __init__(self, initial_pose: np.ndarray):
        # TODO: generate waypoints based on starting pose
        self.waypoints = gen_square_spiral(SPIRAL_MAX, SPIRAL_MIN, SPIRAL_STEP)
        self.waypoint_idx = 0

    def get_waypoint(self, pose: np.ndarray, print_progress: bool = False) -> np.ndarray | None:
        """Get the next waypoint for the agent to follow.

        Returns None if all waypoints have been reached.

        """
        advanced = False
        waypoint = self.waypoints[self.waypoint_idx]
        xy_position = pose[:2, 3]
        if np.linalg.norm(xy_position - waypoint) < WAYPOINT_REACHED_DIST_THRESHOLD:
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.waypoints):
                self.waypoint_idx = 0
                return None
            waypoint = self.waypoints[self.waypoint_idx]
            advanced = True
        if print_progress:
            print(f"Waypoint {self.waypoint_idx + 1}/{len(self.waypoints)}: {waypoint}")
        return waypoint, advanced
