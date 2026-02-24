# LAC Frame, Sign, and Coordinate Conventions

This document is the working source of truth for coordinate frames, rotation signs, and transform conventions used in this repository.

It combines:
- LAC challenge documentation (API + geometry appendices)
- Current implementation details in `lac/*`
- Lessons learned from docking/debug runs

## 1) High-level simulator conventions (from challenge docs)

- Coordinate systems are right-handed.
- Global and robot local frames are both right-handed.
- Robot local axes:
  - `+x`: forward
  - `+y`: left
  - `+z`: up
- Positive yaw in the simulator API is **clockwise** (challenge API note on angular speed).

Practical implication: be explicit when mapping between:
- math-style unicycle control (often CCW-positive)
- simulator command convention (clockwise-positive)

## 2) Lander and robot geometry conventions

From challenge docs and `docs/geometry.json`:
- Lander charging antenna is on `+Y_L` side.
- Lander antenna orientation is approximately `(0, +1, 0)` in lander frame.
- Locator tag (ID 69) is the charging fiducial near the antenna.
- Rover charging receiver is near `(0, -0.208, 0.346)` in rover frame with orientation `(0, -1, 0)`.

## 3) Camera and vision frame conventions in code

### 3.1 Camera/body frame used in control and planning
- Rover/body frame convention in code is:
  - `+x` forward, `+y` left, `+z` up

### 3.2 OpenCV frame
- OpenCV frame is treated as:
  - `+z` forward, `+x` right, `+y` down
- Conversion helpers live in `lac/utils/frames.py`.
- `FiducialDockingPoseEstimator` converts PnP output to rover camera/body conventions before control.

## 4) Transform composition conventions used here

- `T_A_B` means transform from frame `B` into frame `A`.
- Docking controller update path uses:
  - measurement input: `T_cam_tag`
  - camera extrinsic: `T_base_cam`
  - composition: `T_base_tag = T_base_cam @ T_cam_tag`

From `T_base_tag`, planar control quantities are:
- `x = T_base_tag[0,3]`
- `y = T_base_tag[1,3]`
- `psi = atan2(R[1,0], R[0,0])`

Errors:
- `e_x = x - x_des`
- `e_y = y - y_des`
- `alpha = atan2(e_y, max(e_x, eps))`
- `e_psi = wrap(psi - psi_des)`
- `rho = hypot(e_x, e_y)`

## 5) Docking controller sign rules

File: `lac/control/fiducial_docking_controller.py`

- Angular commands are generated in a canonical internal form and then multiplied by `yaw_cmd_sign`.
- `yaw_cmd_sign` is configurable in `docking.json` under `docking.controller`.
  - `+1.0`: keep canonical sign
  - `-1.0`: flip all controller-generated yaw commands

This knob exists specifically to handle simulator/controller sign mismatches without rewriting equations.

## 6) Search/reacquire behavior notes

In `agents/docking_agent.py`:
- If no estimate: agent uses `search_pointing` (world-point-based yaw command).
- If estimate reappears: a settle timer currently holds still for `settle_steps`.
- Intermittent tag visibility can cause oscillatory loops:
  - reacquire -> settle -> rotate -> lose -> search -> reacquire

Mitigation knobs:
- `docking_estimate_hold_steps`
- `docking.settle_steps`
- controller `yaw_cmd_sign`
- controller yaw gains/limits (`k_alpha`, `w_max`, etc.)

## 7) Visualization target consistency

The static docking target box and control target must come from the same config source.

Use `docking.target` in `configs/docking.json`:
- `target_point_lander_m`
- `tag_point_lander_m`
- `tag_yaw_lander_rad`
- `psi_des_rad`

`DockingAgent` derives controller target (`x_des`, `y_des`, `psi_des`) and target-box pose from this same block.

## 8) Wheelbase dimensions for rover box visualization

Rover and target boxes are sized from wheel geometry:
- `params.WHEEL_RIG_POINTS` extents in x/y
- z size from `WHEEL_DIAMETER` (with a floor for visibility)

This keeps visual alignment checks tied to real rover footprint data.

## 9) Debugging checklist (recommended)

When docking diverges:
1. Check JSONL debug log (`logs/docking_debug_*.jsonl`) for:
   - long `found=False` windows
   - `alpha` trend during `rotate`
   - `w_cmd` saturation sign
2. If `alpha` magnitude does not decrease in `rotate`, test `yaw_cmd_sign`.
3. Reduce `settle_steps` if reacquire/dropout loops dominate.
4. Confirm `docking.target` matches intended physical/virtual target side.
5. Confirm camera used by mode (`FrontLeft` vs `Right`) matches expected tag visibility.

## 10) Known lessons learned

- Half-cell and axis mapping mistakes can silently create apparent planner/trajectory frame mismatches.
- Static and dynamic target definitions drifting apart causes misleading visualization.
- Sign mismatches in yaw control can look like "controller instability" but are often deterministic convention errors.
- Explicit per-tick JSONL telemetry is faster to debug than terminal-only logs.
