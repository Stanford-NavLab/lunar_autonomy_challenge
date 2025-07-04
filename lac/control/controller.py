"""Controller module for the LAC challenge."""

from math import pi
import numpy as np
import cv2 as cv

from lac.perception.depth import project_pixel_to_rover
from lac.control.dynamics import arc, dubins_traj
from lac.utils.frames import invert_transform_mat, apply_transform
from lac.util import mask_centroid, wrap_angle, pose_to_pos_rpy
import lac.params as params


from collections import deque
import numpy as np


class ArcPlanner:
    def __init__(
        self,
        arc_config: int | tuple[int, int] = 21,
        arc_duration: float | tuple[float, float] = 4.0,
        max_omega: float | tuple[float, float] = 1,
        max_queue_size: int = 5,
        step_interval: int = 10,
    ):
        # Arc generation (unchanged, omitted for brevity)...
        self.step_interval = step_interval
        self.max_queue_size = max_queue_size
        self.rock_history_queue = deque(maxlen=max_queue_size)

    def update_rock_history(self, rock_coords: np.ndarray, rock_radii: list, pose):
        """Store rocks and their associated pose every `step_interval` frames."""
        # Save both rock data and the associated pose
        self.rock_history_queue.append((rock_coords.copy(), rock_radii.copy(), pose))

    def get_combined_rock_map(self, current_pose: np.ndarray):
        """Transform all historical rocks into the current frame using R @ p + t instead of homogeneous multiplication."""
        combined_coords = []
        combined_radii = []

        for coords, radii, stored_pose in self.rock_history_queue:
            # Compute transform from stored_pose → current_pose
            T_relative = np.linalg.inv(stored_pose) @ current_pose
            rotation_diff = T_relative[:3, :3]
            translation_diff = T_relative[:3, 3]

            for rock in coords:
                transformed_rock = rotation_diff @ rock + translation_diff
                combined_coords.append(transformed_rock)

            combined_radii.extend(radii)

        if combined_coords:
            return np.array(combined_coords), combined_radii
        else:
            return np.zeros((0, 3)), []

    def plan_arc(
        self,
        step: int,
        waypoint_global: np.ndarray,
        current_pose: np.ndarray,
        rock_coords: np.ndarray,
        rock_radii: list,
        poses: list[np.ndarray],
    ):
        """Plan path using rolling rock memory projected into the current frame."""
        # Update memory queue with new data and pose
        self.update_rock_history(step, rock_coords, rock_radii, poses)

        # Combine + transform rocks to current frame
        rock_coords, rock_radii = self.get_combined_rock_map(current_pose)

        # -- original planning logic from here on --
        pose_inv = invert_transform_mat(current_pose)
        waypoint_local = pose_inv @ np.array([waypoint_global[0], waypoint_global[1], 0.0, 1.0])
        lander_local = apply_transform(pose_inv, params.LANDER_GLOBAL)
        min_x, max_x = np.min(lander_local[:, 0]), np.max(lander_local[:, 0])
        min_y, max_y = np.min(lander_local[:, 1]), np.max(lander_local[:, 1])
        lander_bbox = np.array([min_x, max_x, min_y, max_y])

        path_costs = np.linalg.norm(self.np_candidate_arcs[:, -1, :2] - waypoint_local[:2], axis=1)
        sorted_indices = np.argsort(path_costs)

        for i in sorted_indices:
            arc = self.np_candidate_arcs[i]
            valid = True
            for j in range(len(arc)):
                if (
                    lander_bbox[0] <= arc[j][0] <= lander_bbox[1]
                    and lander_bbox[2] <= arc[j][1] <= lander_bbox[3]
                ):
                    path_costs[i] += 1000
                    valid = False
                    break
                for rock, radius in zip(rock_coords, rock_radii):
                    if radius > params.ROCK_MIN_RADIUS:
                        if np.linalg.norm(arc[j][:2] - rock[:2]) - params.ROVER_RADIUS <= radius:
                            path_costs[i] += 1000
                            valid = False
                            break
            if valid:
                return self.vw[i], arc, waypoint_local

        return None, None, None
