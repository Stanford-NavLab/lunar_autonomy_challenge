import numpy as np

from lac.params import CELL_WIDTH
from lac.planning.dem_planner import DEMPlannerParams, plan_path_dem
from lac.planning.path_smoother import Path2D, PathSmoother, PathSmootherConfig


def main() -> None:
    map_arr = np.load(
        "/home/shared/data_raw/Lunar/LAC/maps/competition/Moon_Map_01_preset_0.dat",
        allow_pickle=True,
    )
    # Map is HxWx4 with channels [x, y, z, rock]. We use z only.
    z = map_arr[:, :, 2].astype(np.float32)

    params = DEMPlannerParams(
        roughness_window=5,
        w_s=5.0,
        w_r=2.0,
        w_d=10.0,
        s_max=0.6,
        r_max=0.1,
        d_max=0.15,
        hard_slope_max=0.9,
        hard_step_max=0.22,
        use_lander_keepout=True,
        lander_buffer_m=1.0,
        lander_center_xy_m=(0.0, 0.0),
        do_spline=True,
    )

    path_world, total_cost, debug = plan_path_dem(
        z,
        cell_size=CELL_WIDTH,
        start_xy=(4.0, 4.0),  # world meters
        goal_xy=(12.0, 12.0),  # world meters
        params=params,
        use_theta_star=True,
        do_smooth=True,
        input_is_grid=False,
    )

    if not path_world:
        print("No valid path found.")
        return

    smoother_cfg = PathSmootherConfig(
        cell_size=CELL_WIDTH,
        ds=0.10,
        v_nominal=0.2,
        v_max=0.6,
        max_omega=1.2,
    )
    smoother = PathSmoother(smoother_cfg)
    trajectory = smoother.smooth(Path2D(xy=np.asarray(path_world), meta=debug), z)

    print(f"Path waypoints: {len(path_world)}")
    print(f"Total cost: {total_cost:.3f}")
    print(f"Start/goal grid: {debug['start_grid']} -> {debug['goal_grid']}")
    print(f"Trajectory samples: {len(trajectory.t)}")
    print(f"Trajectory duration [s]: {trajectory.t[-1]:.2f}")
    print(f"Max |omega| [rad/s]: {np.max(np.abs(trajectory.w)):.3f}")

    try:
        import plotly.graph_objects as go

        path_xy = np.asarray(path_world, dtype=np.float64)
        traj_xy = trajectory.xyt[:, :2]

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=z,
                colorscale="Viridis",
                colorbar={"title": "z [m]"},
                name="DEM",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=path_xy[:, 0] / CELL_WIDTH,
                y=path_xy[:, 1] / CELL_WIDTH,
                mode="lines+markers",
                name="Planned path",
                line={"color": "red", "width": 3},
                marker={"size": 4},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=traj_xy[:, 0] / CELL_WIDTH,
                y=traj_xy[:, 1] / CELL_WIDTH,
                mode="lines",
                name="Smoothed trajectory",
                line={"color": "cyan", "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[debug["start_grid"][0], debug["goal_grid"][0]],
                y=[debug["start_grid"][1], debug["goal_grid"][1]],
                mode="markers",
                name="Start/Goal",
                marker={"size": 10, "color": ["white", "magenta"]},
            )
        )
        fig.update_layout(
            title="DEM with planned and smoothed paths",
            xaxis_title="x [grid col]",
            yaxis_title="y [grid row]",
            yaxis={"scaleanchor": "x", "scaleratio": 1},
            template="plotly_dark",
        )
        fig.show()
    except Exception as exc:
        print(f"Plotly visualization skipped: {exc}")


if __name__ == "__main__":
    main()
