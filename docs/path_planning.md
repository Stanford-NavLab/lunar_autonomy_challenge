# Terrain-Aware Path Planning Pipeline

This document summarizes the DEM planning and smoothing pipeline implemented in:

- `lac/planning/dem_planner.py`
- `lac/planning/path_smoother.py`

The goal is to generate a drivable trajectory from a lunar height map, then track it with the controller.

## 1) DEM Planner (`dem_planner.py`)

### Purpose

`dem_planner.py` finds a low-cost path on a height grid (`z[y, x]`) using terrain difficulty, optional lander keepout, and A*/Theta* search.

### Inputs

- `z`: DEM as either:
  - `(H, W)` height array in meters, or
  - `(H, W, C)` map where channel `2` is height
- `cell_size`: meters per grid cell (default from `lac.params.CELL_WIDTH`, usually `0.15`)
- `start_xy`, `goal_xy`: world coordinates (meters) or grid indices
- `input_is_grid`: whether start/goal are grid indices
- `initial_pose` (optional): `4x4` pose or `[x, y, yaw]` used to bias initial path heading
- `DEMPlannerParams`: planner and cost settings

### Terrain Features

`compute_features(...)` computes:

- **Slope**: `||grad z||` in `m/m`
- **Roughness**: local `std(z)` in an `N x N` window
- **Step height**: max absolute height jump to local neighbors

### Traversal Cost and Blocking

`build_costmap(...)` creates:

- `costmap`:
  - `base + w_s*clamp(slope) + w_r*clamp(roughness) + w_d*clamp(step)`
- `blocked` mask:
  - cells blocked if slope/step exceed hard thresholds

### Lander Keepout

If `use_lander_keepout=True`, planner blocks a rectangle around lander footprint:

- footprint from `lac.params.LANDER_GLOBAL`
- extra margin from `lander_buffer_m` (default `1.0 m`)
- center from `lander_center_xy_m`

### Search

`astar(...)` supports:

- 8- or 4-connectivity
- A* and Theta* (`use_theta_star=True`)
- edge cost:
  - `distance * 0.5*(cost[u] + cost[v])`
- LOS check (Bresenham) for Theta\* shortcuts

### Path Post-Processing

`smooth_path(...)` in planner performs:

- collinear waypoint removal
- randomized LOS shortcut smoothing
- optional spline resampling

If `initial_pose` is provided, planner also blends the first section of path toward initial yaw (`initial_heading_window_m`).

### Outputs

`plan_path_dem(...)` returns:

- `path_world`: list of `(x_m, y_m)` waypoints
- `total_cost`: scalar path cost
- `debug` dict containing:
  - features (`slope`, `roughness`, `step`)
  - `costmap`, `blocked`
  - `path_grid`, `start_grid`, `goal_grid`
  - diagnostics and failure reasons

---

## 2) Path Smoother (`path_smoother.py`)

### Purpose

`PathSmoother` converts a polyline path into a dynamically feasible trajectory:

- `xyt` samples (`x, y, theta`)
- linear speed `v`
- angular speed `w`
- timestamps `t`

### Key Data Structures

- `Path2D`:
  - `xy`: `(N,2)` path in world meters
  - `meta`: optional metadata/debug
- `Trajectory2D`:
  - `t`: `(M,)`
  - `xyt`: `(M,3)`
  - `v`: `(M,)`
  - `w`: `(M,)`

### Processing Steps

1. **Resample path** at fixed spacing (`ds`) with spline or linear interpolation
2. **Optional initial heading blend** from `initial_pose` (`initial_heading_window_m`)
3. Compute heading and curvature from arc-length derivatives
4. Sample terrain slope/roughness under the path (bilinear on DEM feature maps)
5. Build speed limits from:
   - terrain severity
   - curvature/lateral acceleration limit
   - hard terrain thresholds (force stop)
6. Apply forward/backward acceleration and deceleration constraints
7. Enforce angular-rate feasibility (`|w| = |kappa * v| <= max_omega`)
8. Integrate time from segment lengths and average speed

### Inputs

- `path`: `Path2D` or `(N,2)` array
- `heightmap`: DEM as `(H,W)` or `(H,W,C>=3)` where channel `2` is height
- `initial_pose` (optional): `4x4` pose or `[x, y, yaw]`
- `PathSmootherConfig`: dynamic and terrain constraints

### Output

- `Trajectory2D(t, xyt, v, w)`

---

## 3) End-to-End Usage

Typical flow:

1. Call `plan_path_dem(...)` on DEM
2. Wrap path in `Path2D`
3. Call `PathSmoother.smooth(...)`
4. Pass `Trajectory2D` to `TrajectoryTracker.compute_command(...)`

Used this way, the planner handles global terrain-aware routing while the smoother and tracker handle local dynamic feasibility and execution.
