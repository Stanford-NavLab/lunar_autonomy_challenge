import os
import random
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lac.plotting import plot_surface

###################################
# Environment Definition
###################################


class FineGrid:
    def __init__(self, N=27, cell_size=0.15, mu_elev=0.0, sigma_elev=1.0):
        self.N = N
        self.cell_size = cell_size
        self.grid_size = int(N // cell_size)  # grid discretization
        # Store two grids:
        # (1) Elevation of each cell (in m)
        # (2) How well each cell is mapped from not mapped (0.0) to perfectly mapped (1.0)
        self.param_names = ["elevation", "is_mapped"]
        self.num_params = len(self.param_names)
        self.grid = self.create_grid(mu_elev, sigma_elev)

    def create_grid(self, mu_elev, sigma_elev):
        """Create a grid given elevation mean [m] and elevation confidence [m]."""
        np.random.seed(4)
        grid = np.zeros((self.grid_size, self.grid_size, self.num_params))
        elevation_noise = np.random.normal(
            mu_elev, sigma_elev, size=(self.grid_size, self.grid_size)
        )
        # elevation_noise = elevation_noise.reshape(self.grid_size, self.grid_size, 1)
        grid[:, :, 0] += elevation_noise
        return grid

    def get_elevation(self, x, y):
        return self.grid[x, y, 0]

    def get_is_mapped(self, x, y):
        return self.grid[x, y, 1]

    def add_rock(rock_x, rock_y, radius):
        # TODO: Make it so that we can add rocks into the elevation grid
        pass

    def update_is_mapped(
        self, rov_x, rov_y, rov_heading, cameras_on, lights_intensity, h_fov, max_range
    ):
        """
        Updates the is_mapped grid.
        IMPROVEMENTS TO BE MADE:
        Add camera spatial location wrt to rover.
        Add vertical field of view.
        No difference between the left and right cameras of the stereo pairs.
        Mapping confidence is a bit arbitrarily modeled right now. Maybe do exponential decay with distance?
        INPUTS:
        rov_x, rov_y: rover location
        rov_heading: rover heading in radians (0 rad means rover is facing to the right)
        cameras_on: 8-element boolean NumPy array
        lights_intensity: 8-element NumPy array with values from 0 (off) to 1.0 (max intensity)
        h_fov: camera's horizontal field of view in radians
        max_range: camera's maximum depth range with max light intensity
        INDEX --> CAMERA TYPE:
        0 --> forward facing stereo pair (left)
        1 --> forward facing stereo pair (right)
        2 --> rear facing stereo pair (left)
        3 --> rear facing stereo pair (right)
        4 --> left facing mid-chassis camera
        5 --> right facing mid-chassis camera
        6 --> camera at the end of the front arm
        7 --> camera at the end of the rear arm
        """
        camera_heading = rov_heading + np.array(
            [0, 0, np.pi, np.pi, np.pi / 2, -np.pi / 2, 0, np.pi]
        )  # rad
        for idx, c in enumerate(cameras_on):
            if c != 0 and lights_intensity[idx] != 0:
                c_heading = camera_heading[idx]
                intensity = lights_intensity[idx]
                for r in np.linspace(0, max_range * intensity, num=20):
                    for theta in np.linspace(-h_fov / 2, h_fov / 2, num=20):
                        # Convert from polar to Cartesian
                        dx = r * np.cos(c_heading + theta)
                        dy = r * np.sin(c_heading + theta)
                        # Convert to grid indices
                        x_idx = int((rov_x + dx) / self.cell_size)
                        y_idx = int((rov_y + dy) / self.cell_size)
                        if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                            # 0.5 is arbitrarily used
                            mapping_confidence = max(
                                0, 0.5 * intensity * (1 - r / (max_range * intensity))
                            )
                            updated_confidence = self.grid[x_idx, y_idx, 1] + mapping_confidence
                            self.grid[x_idx, y_idx, 1] = min(1.0, updated_confidence)

    # VISUALIZATIONS
    def plot_elevation_2d(self) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.imshow(self.grid[:, :, 0], cmap="terrain")
        plt.show()

    def plot_elevation_3d(self) -> go.Figure:
        """Create a 3D surface plot of the elevation grid."""
        x = np.linspace(0, self.N, self.grid_size)
        y = np.linspace(0, self.N, self.grid_size)
        X, Y = np.meshgrid(x, y)
        Z = self.grid[:, :, 0]
        fig = plot_surface(X, Y, Z)
        return fig


class BatteryModel:
    def __init__(self, initial_charge=283):
        self.charge = initial_charge
        self.current_power_draw = 0
        self.last_update_time = 0

    def update_power_draw(self, cameras_on, lights_intensity):
        """
        Update power draw according to LAC specifications.
        IMPROVEMENTS TO BE MADE:
        Does not include excavator arm or arm brake.
        Power draw due to wheels can be further improved.
        INPUTS:
        cameras_on: 8-element boolean NumPy array
        lights_intensity: 8-element NumPy array with values from 0 (off) to 1.0 (max intensity)
        """
        # Hard-coded power draws (constant during mission)
        comp_load = 8
        non_comp_load = 2
        imu_load = 0.15
        constant_power = comp_load + non_comp_load + imu_load

        # Hard-coded depletion rates (in W)
        wheels_load = 50  # for all wheels driving straight with linear speed 0.3 m/s
        camera_load = 3  # for one active camera
        lights_load = 9.8  # for one active light at full intensity (linearly scales with intensity)
        wheels_power = wheels_load  # make this higher fidelity!!!
        camera_power = camera_load * np.sum(cameras_on)
        light_power = 0
        for lights in lights_intensity:
            light_power += lights_load * lights
        self.current_power_draw = constant_power + wheels_power + camera_power + light_power  # W

    def consume(self, elapsed_time):
        hours_elapsed = elapsed_time / 3600  # h
        energy_used = self.current_power_draw * hours_elapsed  # Wh
        self.charge -= energy_used
        self.charge = max(self.charge, 0)


class Rover:
    def __init__(
        self,
        coarse_grid,
        grid_size,
        time_remaining=3600,
        cameras_on=np.zeros(8, dtype=bool),
        lights_intensity=np.zeros(9, dtype=int),
    ):
        self.coarse_grid = coarse_grid
        self.start_x = grid_size // 2
        self.start_y = grid_size // 2
        self.heading = 0
        self.time_remaining = time_remaining  # s
        self.battery = BatteryModel()
        self.cameras_on = cameras_on
        self.lights_intensity = lights_intensity

    def move(self, action):
        # TODO: Move the rover
        pass

    def set_camera(self, idx, state):
        # TODO: Turn a specific camera on or off.
        pass

    def set_light(self, idx, intensity):
        # TODO: Set the intensity of a specific light.
        pass


### CODE FROM PREVIOUS PROJECT. STILL NEED TO UPDATE/FIX
class RoverEnv:
    def __init__(
        self,
        Nx=27,  # m
        Ny=27,  # m
        cell_size=0.15,  # m
        rov_size=0.5,  # m
        max_time=3600,  # s
        initial_battery=283,  # Wh
        cameras_on=3,
        lights_on=3,
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.cell_size = cell_size
        self.rov_size = rov_size
        self.max_time = max_time
        self.initial_battery = initial_battery
        self.cameras_on = cameras_on
        self.lights_on = lights_on

        self.start_x = Nx // 2
        self.start_y = Ny // 2
        self.lander_pos = (self.start_x, self.start_y)
        self.action_space = 5
        self.observation_space_shape = (5,)
        self.total_cells = Nx * Ny

        # Array to count how many times each cell is visited
        self.visitation_counts = np.zeros((Nx, Ny), dtype=int)

        self.reset()

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y

        self.battery = self.initial_battery
        self.time_remaining = self.max_time
        self.visited = np.zeros((self.Nx, self.Ny), dtype=bool)
        self.visited[self.x, self.y] = True
        self.visitation_counts.fill(0)
        self.visitation_counts[self.x, self.y] = 1
        return self._get_state()

    def _get_state(self):
        fraction_visited = np.sum(self.visited) / self.total_cells
        return np.array(
            [self.x, self.y, self.battery, fraction_visited, self.time_remaining], dtype=float
        )

    def step(self, action):
        reward = 0.0
        done = False

        # Movement actions
        old_x, old_y = self.x, self.y
        if action == 0:  # up
            if self.x > 0:
                self.x -= 1
        elif action == 1:  # down
            if self.x < self.Nx - 1:
                self.x += 1
        elif action == 2:  # left
            if self.y > 0:
                self.y -= 1
        elif action == 3:  # right
            if self.y < self.Ny - 1:
                self.y += 1
        elif action == 4:  # recharge
            # If the rover is in the vicinity of the lander, it can recharge the battery
            if abs(self.x - self.lander_pos[0]) + abs(self.y - self.lander_pos[1]) <= 5:
                before_battery = self.battery
                self.battery = min(100, self.battery + self.recharge_rate)
                self.time_remaining -= 100 - before_battery
                # Reward for recharging
                if before_battery < 30:
                    reward += 30.0
                elif before_battery > 70:
                    reward -= 5.0
                else:
                    reward += 0.5
            else:
                reward -= 10.0  # penalty for trying to recharge away from lander

        # Battery drain if moved
        if action in [0, 1, 2, 3]:
            self.battery -= self.battery_drain
            if self.battery < 0:
                self.battery = 0

        # Reward for going toward the goal when the battery is low
        dist_to_lander = abs(self.x - self.lander_pos[0]) + abs(self.y - self.lander_pos[1])
        if self.battery < (dist_to_lander + 10) and action in [0, 1, 2, 3]:
            if dist_to_lander < abs(old_x - self.lander_pos[0]) + abs(old_y - self.lander_pos[1]):
                reward += 5.0
            else:
                reward -= 30.0

        # If battery depleted and tried to move
        if self.battery == 0 and action in [0, 1, 2, 3]:
            reward -= 100.0
            done = True

        # Check if we visited a new cell
        if not self.visited[self.x, self.y]:
            self.visited[self.x, self.y] = True
            reward += 10.0  # extrinsic reward for visiting a new cell
        else:
            reward -= 20.0  # penalty for revisiting a cell

        # Count-based intrinsic motivation:
        self.visitation_counts[self.x, self.y] += 1
        v_count = self.visitation_counts[self.x, self.y]
        intrinsic_reward = 1.0 / np.sqrt(v_count)
        reward += intrinsic_reward

        # Small step cost
        reward -= 0.005

        # Decrement time
        self.time_remaining -= 1
        if self.time_remaining <= 0:
            done = True

        return self._get_state(), reward, done, {}
