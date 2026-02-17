import numpy as np

import lac.params as lac_params
from lac.params import CELL_WIDTH
from lac.planning.dem_planner import DEMPlannerParams, plan_path_dem
from lac.planning.path_smoother import Path2D, PathSmoother, PathSmootherConfig
from lac.utils.plotting import plot_lander_2d


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
    start_xy = (-12.0, -12.0)
    goal_xy = (12.0, -12.0)

    path_world, total_cost, debug = plan_path_dem(
        z,
        cell_size=CELL_WIDTH,
        start_xy=start_xy,  # world meters
        goal_xy=goal_xy,  # world meters
        params=params,
        use_theta_star=True,
        do_smooth=True,
        input_is_grid=False,
    )

    if not path_world:
        print("No valid path found. Diagnostics:")
        for reason in debug.get("failure_reasons", []):
            print(f"  - {reason}")
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
        import plotly.express as px

        path_xy = np.asarray(path_world, dtype=np.float64)
        traj_xy = trajectory.xyt[:, :2]
        traj_v = trajectory.v
        map_extent_m = float(lac_params.MAP_EXTENT)
        x_world = (np.arange(z.shape[1], dtype=np.float64) - z.shape[1] / 2.0) * CELL_WIDTH
        y_world = (np.arange(z.shape[0], dtype=np.float64) - z.shape[0] / 2.0) * CELL_WIDTH

        lander_xy = lac_params.LANDER_GLOBAL[:, :2]
        keepout_x_min_m = (
            float(np.min(lander_xy[:, 0])) - params.lander_buffer_m + params.lander_center_xy_m[0]
        )
        keepout_x_max_m = (
            float(np.max(lander_xy[:, 0])) + params.lander_buffer_m + params.lander_center_xy_m[0]
        )
        keepout_y_min_m = (
            float(np.min(lander_xy[:, 1])) - params.lander_buffer_m + params.lander_center_xy_m[1]
        )
        keepout_y_max_m = (
            float(np.max(lander_xy[:, 1])) + params.lander_buffer_m + params.lander_center_xy_m[1]
        )

        fig = go.Figure()
        fig = plot_lander_2d(fig=fig, color="rgba(255, 215, 0, 0.30)")
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x_world,
                y=y_world,
                colorscale="Viridis",
                colorbar={"title": "z [m]", "x": 1.02, "y": 0.77, "len": 0.42},
                name="DEM",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=path_xy[:, 0],
                y=path_xy[:, 1],
                mode="lines+markers",
                name="Planned path",
                line={"color": "red", "width": 3},
                marker={"size": 4},
            )
        )
        if len(traj_xy) > 1:
            v_min = float(np.min(traj_v))
            v_max = float(np.max(traj_v))
            v_span = max(v_max - v_min, 1e-9)
            segment_speed = 0.5 * (traj_v[:-1] + traj_v[1:])
            segment_norm = (segment_speed - v_min) / v_span
            segment_colors = px.colors.sample_colorscale("Turbo", segment_norm.tolist())

            for i in range(len(traj_xy) - 1):
                fig.add_trace(
                    go.Scatter(
                        x=traj_xy[i : i + 2, 0],
                        y=traj_xy[i : i + 2, 1],
                        mode="lines",
                        line={"color": segment_colors[i], "width": 3},
                        hovertemplate=f"v ~ {segment_speed[i]:.3f} m/s<extra></extra>",
                        name="Smoothed trajectory",
                        showlegend=(i == 0),
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=traj_xy[:, 0],
                    y=traj_xy[:, 1],
                    mode="markers",
                    marker={
                        "size": 0.01,
                        "color": traj_v,
                        "colorscale": "Turbo",
                        "cmin": v_min,
                        "cmax": v_max,
                        "showscale": True,
                        "colorbar": {"title": "v [m/s]", "x": 1.02, "y": 0.23, "len": 0.42},
                    },
                    hoverinfo="skip",
                    name="Smoothed speed",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[start_xy[0], goal_xy[0]],
                y=[start_xy[1], goal_xy[1]],
                mode="markers",
                name="Start/Goal",
                marker={"size": 10, "color": ["white", "magenta"]},
            )
        )
        fig.add_shape(
            type="rect",
            x0=keepout_x_min_m,
            x1=keepout_x_max_m,
            y0=keepout_y_min_m,
            y1=keepout_y_max_m,
            line={"color": "orange", "width": 2, "dash": "dash"},
            fillcolor="rgba(255,165,0,0.12)",
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers",
                marker={"size": 10, "color": "rgba(255, 215, 0, 0.65)", "symbol": "square"},
                name="Lander",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[
                    keepout_x_min_m,
                    keepout_x_max_m,
                    keepout_x_max_m,
                    keepout_x_min_m,
                    keepout_x_min_m,
                ],
                y=[
                    keepout_y_min_m,
                    keepout_y_min_m,
                    keepout_y_max_m,
                    keepout_y_max_m,
                    keepout_y_min_m,
                ],
                mode="lines",
                line={"color": "orange", "width": 2, "dash": "dash"},
                name="Lander keepout",
            )
        )
        fig.update_layout(
            title="DEM with planned path, keepout zone, and velocity-coded smooth trajectory",
            xaxis={"title": "x [m]", "range": [-map_extent_m, map_extent_m]},
            yaxis={
                "title": "y [m]",
                "range": [-map_extent_m, map_extent_m],
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            legend={
                "x": 0.01,
                "y": 0.99,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(0,0,0,0.45)",
            },
            margin={"l": 60, "r": 140, "t": 60, "b": 60},
            template="plotly_dark",
        )
        fig.show()
    except Exception as exc:
        print(f"Plotly visualization skipped: {exc}")


if __name__ == "__main__":
    main()
