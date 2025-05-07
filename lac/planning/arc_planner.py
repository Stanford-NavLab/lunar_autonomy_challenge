"""Sampling-based Arc Path Planner

Similar to Direct-Window-Approach (DWA)

"""

import numpy as np

from lac.control.dynamics import dubins_traj
from lac.utils.frames import invert_transform_mat, apply_transform
import lac.params as params


class ArcPlanner:
    """Arc planner"""

    def __init__(
        self,
        arc_config: int | tuple[int, int] = 31,
        arc_duration: float | tuple[float, float] = 8.0,
        max_omega: float | tuple[float, float] = 0.8,
    ):
        """Initialize the arc planner

        Parameters
        ----------
        arc_config : int | tuple[int, int]
            Number of omega values to sample for the first and second arcs
            (if is_branch is True)
        arc_duration : float | tuple[float, float]
            Duration of the arc in seconds
            (if is_branch is True, this is the duration of the first arc)
        max_omega : float | tuple[float, float]
            Maximum angular velocity in rad/s
            (if is_branch is True, this is the maximum angular velocity of the first arc)

        """
        MAX_OMEGA = max_omega  # [rad/s]
        ARC_DURATION = arc_duration  # [s]
        NUM_ARC_POINTS = int(ARC_DURATION / params.DT)
        self.speeds = [params.TARGET_SPEED]  # [0.05, 0.1, 0.15, 0.2]  # [m/s]
        self.root_arcs = []
        self.candidate_arcs = []
        self.root_vw = []
        self.vw = []
        self.scale = 0.5

        if isinstance(arc_config, int):
            self.is_branch = False
            NUM_OMEGAS_1 = arc_config
            self.omegas1 = np.linspace(-MAX_OMEGA, MAX_OMEGA, NUM_OMEGAS_1)
        else:
            self.is_branch = True
            NUM_OMEGAS_1 = arc_config[0]
            NUM_OMEGAS_2 = arc_config[1]
            self.omegas1 = np.linspace(-MAX_OMEGA, MAX_OMEGA, NUM_OMEGAS_1)
            self.omegas2 = np.linspace(-MAX_OMEGA, MAX_OMEGA, NUM_OMEGAS_2)

        for v in self.speeds:
            for w in self.omegas1:
                new_arc = dubins_traj(np.zeros(3), [v, w], NUM_ARC_POINTS, params.DT)
                self.root_arcs.append(new_arc)
                self.candidate_arcs.append(new_arc)
                self.root_vw.append((v, w * 1 / (self.scale)))

        if self.is_branch:
            concatenated_arcs = []
            for count, root_arc in enumerate(self.root_arcs):
                last_state = root_arc[-1]  # Extract last state [x, y, theta]

                for v in self.speeds:
                    for w in self.omegas2:
                        new_arc = dubins_traj(last_state, [v, w], NUM_ARC_POINTS, params.DT)
                        concatenated_arcs.append(np.concatenate((root_arc, new_arc)))
                        self.vw.append(self.root_vw[count])

            self.candidate_arcs = concatenated_arcs
        else:
            self.vw = self.root_vw
        self.np_candidate_arcs = np.array(self.candidate_arcs)

    def plan_arc(
        self,
        waypoint_global: np.ndarray,
        current_pose: np.ndarray,
        rock_data: dict,
        current_velocity: float,
    ) -> tuple:
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
                for rock, radius in zip(rock_data["centers"], rock_data["radii"]):
                    if radius > params.ROCK_MIN_RADIUS:
                        if np.linalg.norm(arc[j][:2] - rock[:2]) - params.ROVER_RADIUS <= radius:
                            path_costs[i] += 1000
                            valid = False
                            break
            if valid:
                return self.vw[i], arc, waypoint_local

        # If no valid path is found, return None
        return None, None, None
