"""Sampling-based Arc Path Planner

Similar to Direct-Window-Approach (DWA)

"""

import numpy as np

from lac.control.dynamics import dubins_traj
from lac.utils.frames import invert_transform_mat, apply_transform
import lac.params as params


from collections import deque
import numpy as np
from lac.utils.plotting import (
    plot_points_rover_frame,
    plot_path_rover_frame,
    plot_rocks_rover_frame,
)


class TemporalArcPlanner:
    def __init__(
        self,
        arc_config: int | tuple[int, int] = 31,
        arc_duration: float | tuple[float, float] = 8.0,
        max_omega: float | tuple[float, float] = 0.8,
        max_queue_size: int = 3,
        step_interval: int = 10,
    ):
        # Arc generation (unchanged, omitted for brevity)...
        self.step_interval = step_interval
        self.max_queue_size = max_queue_size
        self.rock_history_queue = deque(maxlen=max_queue_size)
        MAX_OMEGA = max_omega  # [rad/s]
        ARC_DURATION = arc_duration  # [s]
        NUM_ARC_POINTS = int(ARC_DURATION / params.DT)
        self.speeds = [params.TARGET_SPEED]  # [0.05, 0.1, 0.15, 0.2]  # [m/s]
        self.root_arcs = []
        self.candidate_arcs = []
        self.root_vw = []
        self.vw = []
        self.scale = 0.5
        self.min_depth = 0.5

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

    def update_rock_history(self, rock_coords: np.ndarray, rock_radii: list, pose, depth):
        """Store rocks and their associated pose every `step_interval` frames."""
        # Save both rock data and the associated pose
        close_rock_coords = []
        close_rock_radii = []

        for i, rock in enumerate(depth):
            if rock["depth"] < self.min_depth:
                close_rock_coords.append(rock_coords[i])
                close_rock_radii.append(rock_radii[i])

        if close_rock_coords:
            self.rock_history_queue.append((np.array(close_rock_coords), close_rock_radii, pose))

    def get_combined_rock_map(self, current_pose: np.ndarray):
        """Transform all historical rocks into the current frame."""
        combined_coords = []
        combined_radii = []

        for coords, radii, stored_pose in self.rock_history_queue:
            # Compute transform from stored_pose → current_pose
            T_relative = np.linalg.inv(stored_pose) @ current_pose

            for rock in coords:
                rock_homog = np.append(rock, 1)  # [x, y, z, 1]
                transformed_rock = (T_relative @ rock_homog)[:3]  # back to [x, y, z]
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
        current_velocity: float,
        depth,
    ):
        """Plan path using rolling rock memory projected into the current frame."""
        # Update memory queue with new data and pose
        self.update_rock_history(rock_coords, rock_radii, current_pose, depth)

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
                    # path_costs[i] += 1000
                    valid = False
                    break
                for rock, radius in zip(rock_coords, rock_radii):
                    if radius > params.ROCK_MIN_RADIUS:
                        if np.linalg.norm(arc[j][:2] - rock[:2]) - params.ROVER_RADIUS <= radius:
                            # path_costs[i] += 1000
                            valid = False
                            break
            if valid:
                return self.vw[i], arc, waypoint_local

        return None, None, None

    def plot_rocks(self, combined_map, arcs, best_arc, step, show=False):
        # Create folder if it doesn't exist

        # Plot rocks
        fig = plot_rocks_rover_frame(combined_map[0], combined_map[1], color="red")

        # Plot all arcs
        for arc in arcs:
            fig = plot_path_rover_frame(arc, fig=fig)

        # Plot best arc in green
        if best_arc is not None:
            fig = plot_path_rover_frame(best_arc, color="green", fig=fig)

        # Save the figure to HTML
        save_path = f"results/planner_graphs_temporal/rock_plot_{step}.html"
        fig.write_html(save_path)

        if show:
            fig.show()
