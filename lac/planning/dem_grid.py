"""Minimal DEM utility for frame-aware indexing and height queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DEMGrid:
    """Lightweight DEM wrapper for centered-grid world conversions."""

    z: np.ndarray
    cell_size: float
    swap_xy: bool = False

    def __post_init__(self):
        z_arr = np.asarray(self.z)
        if z_arr.ndim != 2:
            raise ValueError(f"DEM z must be 2D, got shape {z_arr.shape}")
        self.z = z_arr.astype(np.float32, copy=False)

    @classmethod
    def from_map_array(
        cls, map_arr: np.ndarray, cell_size: float, swap_xy: bool = False
    ) -> "DEMGrid":
        arr = np.asarray(map_arr)
        if arr.ndim == 2:
            z = arr
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            z = arr[:, :, 2]
        else:
            raise ValueError(f"map_arr must be (H,W) or (H,W,C>=3), got {arr.shape}")
        return cls(z=z, cell_size=float(cell_size), swap_xy=swap_xy)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.z.shape

    def _apply_frame(self, x_m: float, y_m: float) -> Tuple[float, float]:
        if self.swap_xy:
            return float(y_m), float(x_m)
        return float(x_m), float(y_m)

    def world_to_grid(self, x_m: float, y_m: float, clip: bool = True) -> Tuple[int, int]:
        xq, yq = self._apply_frame(x_m, y_m)
        h, w = self.shape
        gx = int(round(xq / self.cell_size + (w - 1) / 2.0))
        gy = int(round(yq / self.cell_size + (h - 1) / 2.0))
        if clip:
            gx = int(np.clip(gx, 0, w - 1))
            gy = int(np.clip(gy, 0, h - 1))
        return gx, gy

    def sample_height(self, x_m: float, y_m: float) -> float:
        gx, gy = self.world_to_grid(x_m, y_m, clip=True)
        return float(self.z[gy, gx])

    def lift_xy_to_xyz(self, xy: np.ndarray, z_offset_m: float = 0.0) -> np.ndarray:
        xy_arr = np.asarray(xy, dtype=np.float64)
        if xy_arr.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        if xy_arr.ndim != 2 or xy_arr.shape[1] != 2:
            raise ValueError(f"xy must be shape (N,2), got {xy_arr.shape}")

        xyz = np.zeros((len(xy_arr), 3), dtype=np.float64)
        for i, (x_m, y_m) in enumerate(xy_arr):
            xyz[i, 0] = float(x_m)
            xyz[i, 1] = float(y_m)
            xyz[i, 2] = self.sample_height(float(x_m), float(y_m)) + float(z_offset_m)
        return xyz

    def mesh_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return DEM x/y grids in the configured world frame."""
        h, w = self.shape
        x_coords_col = (np.arange(w, dtype=np.float32) - (w - 1) / 2.0) * float(self.cell_size)
        y_coords_row = (np.arange(h, dtype=np.float32) - (h - 1) / 2.0) * float(self.cell_size)
        if self.swap_xy:
            x_grid = np.tile(y_coords_row[:, None], (1, w))
            y_grid = np.tile(x_coords_col[None, :], (h, 1))
        else:
            x_grid = np.tile(x_coords_col[None, :], (h, 1))
            y_grid = np.tile(y_coords_row[:, None], (1, w))
        return x_grid, y_grid
