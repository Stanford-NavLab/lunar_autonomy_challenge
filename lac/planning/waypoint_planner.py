"""Waypoint Planner class

Generates waypoints for the agent to follow, and tracks the agent's progress.

"""

import numpy as np
from rich import print

from lac.planning.waypoint_generation import (
    gen_spiral,
    gen_five_loops,
    gen_nine_loops,
    gen_triangle_loops,
)
from lac.params import WAYPOINT_REACHED_DIST_THRESHOLD

SPIRAL_MIN = 3.5  # [m]
SPIRAL_MAX = 5.5  # [m]
SPIRAL_STEP = 0.35  # [m]

# Clockwise order: top-left, top-right, bottom-right, bottom-left
DEFAULT_ORDER = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])

WAYPOINT_TIMEOUT = 2000  # [steps] timeout to give up on a waypoint


class WaypointPlanner:
    def __init__(
        self,
        initial_pose: np.ndarray,
        trajectory_type: str = "five_loops",
        waypoint_reached_threshold: float = WAYPOINT_REACHED_DIST_THRESHOLD,
    ):
        """
        trajectory_type: str = "spiral", "five_loops", "nine_loops", "triangles"

        """
        match trajectory_type:
            case "spiral":
                self.waypoints = gen_spiral(initial_pose, SPIRAL_MIN, SPIRAL_MAX, SPIRAL_STEP)
            case "five_loops":
                self.waypoints = gen_five_loops(initial_pose, extra_closure=True)
            case "nine_loops":
                self.waypoints = gen_nine_loops(initial_pose)
            case "triangles":
                self.waypoints = gen_triangle_loops(initial_pose, additional_loops=False)
            case _:
                raise ValueError(f"Unknown trajectory type: {trajectory_type}")

        self.waypoint_idx = 0
        self.last_waypoint_step = 0
        self.waypoint_reached_threshold = waypoint_reached_threshold

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
            print(f"[bold red]WAYPOINT TIMEOUT ON {self.waypoint_idx + 1}/{len(self.waypoints)}")
            advanced = True

        # Check if the waypoint has been reached
        distance = np.linalg.norm(xy_position - waypoint)
        if distance < self.waypoint_reached_threshold:
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
            print(f"Distance to waypoint: {distance:.2f} m")

        return waypoint, advanced
