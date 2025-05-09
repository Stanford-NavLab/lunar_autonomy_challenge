"""Plotting functions"""

import os
import shutil
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import imageio.v2 as imageio

from lac.params import LANDER_GLOBAL, LANDER_HEIGHT, HEIGHT_ERROR_TOLERANCE, MAP_EXTENT

# ==================================================================================================
#                                    2D Plotting Functions
# ==================================================================================================


def plot_heatmap(data: np.ndarray, fig=None, colorscale="Viridis", no_axes=False):
    """Plot a 2D heatmap."""
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Heatmap(z=data, colorscale=colorscale))
    if no_axes:
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig.update_layout(width=800, height=800, xaxis=dict(scaleanchor="y"))
    return fig


def plot_heightmaps(ground_map: np.ndarray, agent_map: np.ndarray):
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Ground truth map", "Agent Map", "Error"],
        horizontal_spacing=0.1,
    )
    error = ground_map[:, :, 2] - agent_map[:, :, 2]
    fig.add_trace(
        go.Heatmap(z=ground_map[:, :, 2], colorscale="Viridis", colorbar=dict(x=0.27)), row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=agent_map[:, :, 2], colorscale="Viridis", colorbar=dict(x=0.63)), row=1, col=2
    )
    fig.add_trace(go.Heatmap(z=error, colorscale="Viridis", colorbar=dict(x=1.0)), row=1, col=3)
    fig.update_layout(
        width=1400,  # Adjust the figure width
        height=505,  # Adjust the figure height
        xaxis=dict(scaleanchor="y"),
        xaxis2=dict(scaleanchor="y2"),
        xaxis3=dict(scaleanchor="y3"),
    )
    fig.show()


def plot_height_error(ground_map: np.ndarray, agent_map: np.ndarray):
    fig = go.Figure()
    error = np.abs(ground_map[:, :, 2] - agent_map[:, :, 2])
    fig.add_trace(
        go.Heatmap(
            z=(error < HEIGHT_ERROR_TOLERANCE).astype(np.uint8),
            x=ground_map[:, :, 0][:, 0],
            y=ground_map[:, :, 1][0, :],
            colorscale=["red", "green"],
            showscale=False,
        ),
    )
    fig.update_layout(
        width=900,  # Adjust the figure width
        height=900,  # Adjust the figure height
        xaxis=dict(scaleanchor="y"),
        xaxis_title="X",
        yaxis_title="Y",
    )
    return fig


def plot_rock_results(ground_map: np.ndarray, agent_map: np.ndarray):
    fig = go.Figure()
    result = np.zeros_like(ground_map[:, :, 3])
    result[(ground_map[:, :, 3] == 0) & (agent_map[:, :, 3] == 0)] = 0  # TN
    result[(ground_map[:, :, 3] == 0) & (agent_map[:, :, 3] == 1)] = 1  # FP
    result[(ground_map[:, :, 3] == 1) & (agent_map[:, :, 3] == 0)] = 2  # FN
    result[(ground_map[:, :, 3] == 1) & (agent_map[:, :, 3] == 1)] = 3  # TP
    label_names = ["TN", "FP", "FN", "TP"]
    colors = ["gray", "red", "yellow", "green"]  # gray, red, yellow, green
    fig.add_trace(
        go.Heatmap(
            z=result,
            x=ground_map[:, :, 0][:, 0],
            y=ground_map[:, :, 1][0, :],
            colorscale=colors,
            showscale=False,
        )
    )
    # Add fake scatter traces for legend
    for i, name in enumerate(label_names):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers", marker=dict(size=10, color=colors[i]), name=name
            )
        )
    fig.update_layout(
        width=800,  # Adjust the figure width
        height=800,  # Adjust the figure height
        xaxis=dict(scaleanchor="y"),
        xaxis_title="X",
        yaxis_title="Y",
    )
    return fig


def plot_rock_maps(ground_map: np.ndarray, agent_map: np.ndarray):
    """
    rock_map is NxNx4 array where 4th channel is rock presence (0 or 1)

    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Ground truth map", "Agent Map", "Result"],
        horizontal_spacing=0.05,
    )
    fig.add_trace(
        go.Heatmap(
            z=ground_map[:, :, 3],
            x=ground_map[:, :, 0][:, 0],
            y=ground_map[:, :, 1][0, :],
            colorscale="Viridis",
            colorbar=dict(x=0.27),
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=agent_map[:, :, 3],
            x=ground_map[:, :, 0][:, 0],
            y=ground_map[:, :, 1][0, :],
            colorscale="Viridis",
            colorbar=dict(x=0.63),
            showscale=False,
        ),
        row=1,
        col=2,
    )
    result = np.zeros_like(ground_map[:, :, 3])
    result[(ground_map[:, :, 3] == 0) & (agent_map[:, :, 3] == 0)] = 0  # TN
    result[(ground_map[:, :, 3] == 0) & (agent_map[:, :, 3] == 1)] = 1  # FP
    result[(ground_map[:, :, 3] == 1) & (agent_map[:, :, 3] == 0)] = 2  # FN
    result[(ground_map[:, :, 3] == 1) & (agent_map[:, :, 3] == 1)] = 3  # TP
    label_names = ["TN", "FP", "FN", "TP"]
    colors = ["gray", "red", "yellow", "green"]  # gray, red, yellow, green
    fig.add_trace(
        go.Heatmap(
            z=result,
            x=ground_map[:, :, 0][:, 0],
            y=ground_map[:, :, 1][0, :],
            colorscale=colors,
            showscale=False,
        ),
        row=1,
        col=3,
    )
    # Add fake scatter traces for legend
    for i, name in enumerate(label_names):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers", marker=dict(size=10, color=colors[i]), name=name
            )
        )
    fig.update_layout(
        width=1400,  # Adjust the figure width
        height=555,  # Adjust the figure height
        xaxis=dict(scaleanchor="y"),
        xaxis2=dict(scaleanchor="y2"),
        xaxis3=dict(scaleanchor="y3"),
        xaxis_title="X",
        yaxis_title="Y",
    )
    fig.show()


def plot_waypoints(
    waypoints: np.ndarray,
    fig: go.Figure | None = None,
    arrow_scale: float = 0.2,
    arrow_size: int = 12,
    show_endpoint: bool = False,
):
    if fig is None:
        fig = go.Figure()

    # Path segments
    fig.add_trace(
        go.Scatter(
            x=waypoints[:, 0],
            y=waypoints[:, 1],
            mode="markers+lines",
            line=dict(color="blue", width=2),
            hovertext=np.arange(len(waypoints)),
            showlegend=False,
        )
    )

    # Arrows
    # Compute segment directions
    dx, dy = np.diff(waypoints[:, 0]), np.diff(waypoints[:, 1])
    seg_len = np.hypot(dx, dy)
    seg_len[seg_len == 0] = 1e-9  # avoid division by zero

    # Compute arrowhead positions and orientations
    arrow_x = waypoints[:-1, 0] + dx * arrow_scale
    arrow_y = waypoints[:-1, 1] + dy * arrow_scale
    angles = 90 - np.degrees(np.arctan2(dy, dx))  # Rotate from +Y

    fig.add_trace(
        go.Scatter(
            x=arrow_x,
            y=arrow_y,
            mode="markers",
            showlegend=False,
            hoverinfo="skip",
            marker=dict(
                symbol="triangle-up",
                size=arrow_size,
                angle=angles,
                color="blue",
                line=dict(width=0),
            ),
        )
    )

    if show_endpoint:
        # Add the last waypoint as a marker
        fig.add_trace(
            go.Scatter(
                x=[waypoints[-1, 0]],
                y=[waypoints[-1, 1]],
                mode="markers",
                marker=dict(color="green", size=arrow_size),
                name="Endpoint",
                showlegend=False,
            )
        )

    fig.update_layout(
        width=600,
        height=600,
        xaxis=dict(scaleanchor="y"),
        xaxis_range=[-MAP_EXTENT, MAP_EXTENT],
        yaxis_range=[-MAP_EXTENT, MAP_EXTENT],
    )
    return fig


def save_waypoints_gif(waypoints: np.ndarray, name: str, path: str, dt: int = 5.0):
    """
    Save a gif of the waypoints.

    Returns:
        None
    """
    image_folder = f"{path}/{name}"
    output_gif = f"{path}/{name}.gif"

    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)
    os.makedirs(image_folder)

    for i in range(len(waypoints)):
        fig = plot_waypoints(waypoints[: i + 1], show_endpoint=True)
        fig.write_image(f"{image_folder}/{i:03}.png")

    # Sort filenames to ensure correct order
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]
    )

    # Read and write images into a GIF
    with imageio.get_writer(output_gif, mode="I", duration=dt) as writer:
        for filename in image_files:
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            writer.append_data(image)


def plot_rocks_rover_frame(rock_points, rock_radii, waypoint=None, fig=None, color="red", **kwargs):
    """Plot rocks with radii in the rover frame."""
    if fig is None:
        fig = go.Figure()

    # Ensure inputs are NumPy arrays
    rock_points = np.asarray(rock_points)[:, :2]
    rock_radii = np.array(rock_radii, dtype=float)  # Convert to NumPy array explicitly

    if rock_points.shape[1] != 2:
        raise ValueError("rock_points must be an (N, 2) array of (x, y) coordinates.")
    if len(rock_radii) != rock_points.shape[0]:
        raise ValueError("rock_radii must have the same length as rock_points.")

    # Scatter plot for rock centers
    fig.add_trace(
        go.Scatter(
            x=rock_points[:, 1],
            y=rock_points[:, 0],
            mode="markers",
            marker=dict(color=color, size=5),
            name="Rock Centers",
            **kwargs,
        )
    )
    if waypoint is not None:
        fig.add_trace(
            go.Scatter(
                x=[waypoint[1]],
                y=[waypoint[0]],
                mode="markers",
                marker=dict(color="blue", size=10),
                name="Waypoint",
                **kwargs,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=rock_points[:, 1],
            y=rock_points[:, 0],
            mode="markers",
            marker=dict(color=color, size=5),
            name="Rock Centers",
            **kwargs,
        )
    )

    # Create parametric circles for all rocks in a single trace
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    all_x = []
    all_y = []
    for (x, y), r in zip(rock_points, rock_radii):
        if r is not None:  # Ensure radius is valid
            all_x.extend(x + r * circle_x)
            all_y.extend(y + r * circle_y)
            all_x.append(None)  # Break between circles
            all_y.append(None)

    fig.add_trace(
        go.Scatter(
            x=all_y,  # Swapping x and y to match rover frame convention
            y=all_x,
            mode="lines",
            line=dict(color=color, width=1),
            name="Rock Boundaries",
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="Y axis (Left)",
            autorange="reversed",
            zerolinewidth=3,
            zerolinecolor="gray",
            tickmode="linear",
            dtick=1,  # Ensure uniform spacing
        ),
        yaxis=dict(
            title="X axis (Forward)",
            scaleanchor="x",
            zerolinewidth=3,
            zerolinecolor="gray",
            tickmode="linear",
            dtick=1,  # Ensure uniform spacing
        ),
        showlegend=True,
    )
    return fig


def plot_points_rover_frame(points: np.ndarray, fig=None, color="red", **kwargs):
    """Plot points in the rover frame."""
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=points[:, 1],
            y=points[:, 0],
            mode="markers",
            marker=dict(color=color, size=5),
            **kwargs,
        )
    )
    fig.update_layout(
        xaxis=dict(
            title="Y axis (Left)",
            autorange="reversed",
            zerolinewidth=3,
            zerolinecolor="gray",
            tickmode="linear",
            dtick=1,  # Ensure uniform spacing
        ),
        yaxis=dict(
            title="X axis (Forward)",
            scaleanchor="x",
            zerolinewidth=3,
            zerolinecolor="gray",
            tickmode="linear",
            dtick=1,  # Ensure uniform spacing
        ),
    )
    return fig


def plot_path_rover_frame(
    path: np.ndarray, fig=None, color="blue", linewidth=2, waypoint=None, **kwargs
):
    """Plot points in the rover frame."""
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=path[:, 1],
            y=path[:, 0],
            mode="lines",
            line=dict(color=color, width=linewidth),
            **kwargs,
        )
    )
    if waypoint is not None:
        fig.add_trace(
            go.Scatter(
                x=[waypoint[1]],
                y=[waypoint[0]],
                mode="markers",
                marker=dict(color="blue", size=10),
                name="Waypoint",
                **kwargs,
            )
        )
        # Create parametric circles for all rocks in a single trac
    if color == "green":
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)

        all_x = []
        all_y = []

        for count, (x, y) in enumerate(path[:, :2]):
            if count % 10 == 0:
                all_x.extend(x + 0.5 * circle_x)
                all_y.extend(y + 0.5 * circle_y)
                all_x.append(None)  # Break between circles
                all_y.append(None)

        fig.add_trace(
            go.Scatter(
                x=all_y,  # Swapping x and y to match rover frame convention
                y=all_x,
                mode="lines",
                line=dict(color=color, width=1),
                name="Rover Boundaries",
            )
        )

    fig.update_layout(
        xaxis=dict(
            title="Y axis (Left)",
            autorange="reversed",
            zerolinewidth=3,
            zerolinecolor="gray",
            tickmode="linear",
            dtick=1,  # Ensure uniform spacing
        ),
        yaxis=dict(
            title="X axis (Forward)",
            scaleanchor="x",
            zerolinewidth=3,
            zerolinecolor="gray",
            tickmode="linear",
            dtick=1,  # Ensure uniform spacing
        ),
    )
    return fig


# ==================================================================================================
#                                    3D Plotting Functions
# ==================================================================================================


def plot_surface(
    grid: np.ndarray, fig=None, colorscale="Viridis", no_axes=False, showscale=True, **kwargs
):
    """
    grid is NxNx3 array representing the coordinates and elevation data for a surface plot.

    """
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=grid[:, :, 0],
            y=grid[:, :, 1],
            z=grid[:, :, 2],
            colorscale=colorscale,
            showscale=showscale,
            **kwargs,
        )
    )
    fig.update_layout(width=1600, height=900, scene_aspectmode="data")
    if no_axes:
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
            )
        )
    return fig


def plot_rock_map(grid: np.ndarray, fig=None, no_axes=False, **kwargs):
    """
    grid is NxNx4 array where 4th channel is rock presence (0 or 1)

    """
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=grid[:, :, 0],
            y=grid[:, :, 1],
            z=grid[:, :, 2],
            surfacecolor=grid[:, :, 3].astype(int),
            colorscale=[[0, "gray"], [1, "red"]],
            showscale=False,
            hoverinfo="none",
            **kwargs,
        )
    )
    fig.update_layout(width=1600, height=900, scene_aspectmode="data")
    if no_axes:
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
            )
        )
    return fig


def plot_path_3d(
    path: np.ndarray,
    fig=None,
    color="red",
    markersize=3,
    linewidth=3,
    markers=True,
    **kwargs,
):
    """Plot a 3D path with optional markers."""
    if fig is None:
        fig = go.Figure()
    if markers:
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2],
                mode="markers+lines",
                marker=dict(size=markersize, color=color),
                line=dict(color=color, width=linewidth),
                hovertext=np.arange(len(path)),
                **kwargs,
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2],
                mode="lines",
                line=dict(color=color, width=linewidth),
                hovertext=np.arange(len(path)),
                **kwargs,
            )
        )
    return fig


def plot_3d_points(points: np.ndarray, fig=None, color="blue", markersize=3, name=None):
    """Plot 3D points."""
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=markersize, color=color),
            name=name,
        )
    )
    fig.update_layout(width=1200, height=900, scene_aspectmode="data")
    return fig


def plot_semantic_points(
    points: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    fig=None,
    markersize=3,
    **kwargs,
):
    """Plot 3D points with semantic labels.

    labels: 1D array of labels
    names: list of names for each label (0, 1, 2, ...)

    """
    if fig is None:
        fig = go.Figure()
    colors = px.colors.qualitative.Plotly[: len(names)]
    for i, name in enumerate(names):
        mask = labels == i
        fig.add_trace(
            go.Scatter3d(
                x=points[mask, 0],
                y=points[mask, 1],
                z=points[mask, 2],
                mode="markers",
                marker=dict(size=markersize, color=colors[i]),
                name=name,
            )
        )
    fig.update_layout(width=1200, height=900, scene_aspectmode="data")
    return fig


def pose_trace(
    pose: np.ndarray | tuple, name: str = "", line_style: str = "solid", line_width: int = 5
):
    """Create a plotly trace to visualize a pose

    RGB vectors are used to represent the X, Y, Z axes of the rotation matrix

    Parameters
    ----------
    pose : 4x4 np.ndarray or tuple

    Returns
    -------
    traces : list
        List of plotly traces for displaying the pose

    """
    # If pose is tuple
    if isinstance(pose, tuple):
        # Unpack pose into rotation matrix R and translation vector t
        R, t = pose
        t = np.array(t)  # Ensure t is a numpy array
    else:  # If pose is a 4x4 matrix
        R = pose[:3, :3]
        t = pose[:3, 3]

    # Define arrow colors for each axis (RGB)
    colors = ["red", "green", "blue"]

    # Define the unit vectors from the columns of R
    axis_vectors = [R[:, 0], R[:, 1], R[:, 2]]

    # Create traces for each axis (X, Y, Z)
    traces = []
    for i, vec in enumerate(axis_vectors):
        arrow_start = t
        arrow_end = t + vec  # Arrow points in the direction of the column of R

        # Create an arrow trace for the axis
        if name == "":
            trace = go.Scatter3d(
                x=[arrow_start[0], arrow_end[0]],
                y=[arrow_start[1], arrow_end[1]],
                z=[arrow_start[2], arrow_end[2]],
                mode="lines",
                marker=dict(size=4),
                line=dict(color=colors[i], width=5),
                showlegend=False,
            )
        else:
            AXES_NAMES = ["X", "Y", "Z"]
            trace = go.Scatter3d(
                x=[arrow_start[0], arrow_end[0]],
                y=[arrow_start[1], arrow_end[1]],
                z=[arrow_start[2], arrow_end[2]],
                mode="lines",
                marker=dict(size=4),
                line=dict(color=colors[i], width=line_width, dash=line_style),
                name=name + f"_{AXES_NAMES[i]}",
                showlegend=True,
            )
        traces.append(trace)

    return traces


def pose_traces(pose_list: list | np.ndarray, **kwargs):
    """Create traces for a sequence of poses

    Parameters
    ----------
    pose_list : list of tuples
        List of poses, where each pose is a tuple (R, t)

    Returns
    -------
    all_traces : list
        List of plotly traces for displaying all poses

    """
    all_traces = []

    for pose in pose_list:
        traces = pose_trace(pose, **kwargs)
        all_traces.extend(traces)

    return all_traces


def plot_poses(poses: list | np.ndarray, fig=None, no_axes=False, **kwargs):
    """poses is a list of 4x4 arrays or an Nx4x4 array"""
    if fig is None:
        fig = go.Figure()
    if no_axes:
        positions = np.array([pose[:3, 3] for pose in poses])
        fig = plot_path_3d(positions, fig=fig, **kwargs)
    else:
        fig.add_traces(pose_traces(poses, **kwargs))
    fig.update_layout(width=1600, height=900, scene_aspectmode="data")
    return fig


def plot_loop_closures(trajectory, loop_closures: list, fig=None, **kwargs):
    """Plot loop closures

    trajectory: sequence of poses or positions
    loop_closures: list of tuples

    """
    if fig is None:
        fig = go.Figure()
    for i, j in loop_closures:
        fig.add_trace(
            go.Scatter3d(
                x=[trajectory[i][0, 3], trajectory[j][0, 3]],
                y=[trajectory[i][1, 3], trajectory[j][1, 3]],
                z=[trajectory[i][2, 3], trajectory[j][2, 3]],
                mode="markers+lines",
                marker=dict(color="red", size=5),
                line=dict(color="red", width=5),
                name=f"LC {i}-{j}",
                **kwargs,
            )
        )
    fig.update_layout(width=1600, height=900, scene_aspectmode="data")
    return fig


def plot_lander_3d(lander_elevation: float, fig=None, color="silver"):
    """Plot the lander as a box in 3D."""
    if fig is None:
        fig = go.Figure()

    vertices = np.vstack([LANDER_GLOBAL, LANDER_GLOBAL + np.array([0, 0, LANDER_HEIGHT])])
    vertices[:, 2] += lander_elevation

    # Define the triangular faces of the box
    faces = [
        [0, 1, 2],
        [0, 2, 3],  # Bottom face
        [4, 5, 6],
        [4, 6, 7],  # Top face
        [0, 1, 5],
        [0, 5, 4],  # Side face 1
        [1, 2, 6],
        [1, 6, 5],  # Side face 2
        [2, 3, 7],
        [2, 7, 6],  # Side face 3
        [3, 0, 4],
        [3, 4, 7],  # Side face 4
    ]

    # Create 3D mesh plot
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=[face[0] for face in faces],
            j=[face[1] for face in faces],
            k=[face[2] for face in faces],
            color=color,
            opacity=0.5,
            name="Lander",
            showlegend=True,
        )
    )

    return fig


def vector_trace(
    start: np.ndarray, end: np.ndarray, color: str = "blue", name: str = "", head_size: float = 0.1
) -> go.Scatter3d:
    """
    Create an arrow trace for a vector
    Input:
        start - start point of the vector
        end - end point of the vector
        color - color of the vector
        name - name of the vector
        head_size - size of the arrow head
    Output:
        list of traces for the vector (tail and head)
    """
    rel_vec = end - start
    rel_vec_unit = rel_vec / np.linalg.norm(rel_vec)
    head_base = end - head_size * rel_vec_unit

    tail_trace = go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode="lines+markers",
        marker=dict(size=5, color=color),
        name=name,
    )
    head_trace = go.Cone(
        x=[head_base[0]],
        y=[head_base[1]],
        z=[head_base[2]],
        u=[rel_vec_unit[0]],
        v=[rel_vec_unit[1]],
        w=[rel_vec_unit[2]],
        showscale=False,
        colorscale=[[0, color], [1, color]],
        sizemode="absolute",
        sizeref=head_size,  # Adjust size of the arrow head
    )

    return [tail_trace, head_trace]


def plot_reference_frames(poses: list[np.ndarray], pose_names: list[str]) -> go.Figure:
    """Generate a figure of the various frames of reference."""

    fig_poses = go.Figure()
    for i, pose in enumerate(poses):
        pose_traces = pose_trace((pose[:3, :3], pose[:3, 3]), name=pose_names[i])
        for trace in pose_traces:
            fig_poses.add_trace(trace)

    fig_poses.update_layout(height=700, width=1200, scene_aspectmode="data")
    return fig_poses


def plot_mesh(mesh, show_edges=True, textured=False):
    """
    Plots the vertices and edges of a PyTorch3D mesh using Plotly's Mesh3d.

    Args:
        mesh: A PyTorch3D Meshes object.
        title: Title of the plot.
    """
    verts = mesh.verts_packed().clone().detach().cpu().numpy()
    faces = mesh.faces_packed().clone().detach().cpu().numpy()

    # Extract vertices
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    # Create Mesh3d plot
    mesh_plot = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="lightblue",
        opacity=0.50,
        name="Mesh",
    )

    # Create scatter plot for vertices
    if textured and mesh.textures:
        colors = mesh.textures.verts_features_packed().clone().detach().cpu().numpy()
        # TODO: handle TexturesUV
    else:
        colors = "lightblue"
    vertices_plot = go.Scatter3d(
        x=x, y=y, z=z, mode="markers", marker=dict(color=colors, size=2), name="Vertices"
    )

    data = [mesh_plot, vertices_plot]

    # Create edges
    if show_edges:
        edges = set()
        for face in faces:
            for i in range(3):
                edge = tuple(sorted((face[i], face[(i + 1) % 3])))
                edges.add(edge)

        edge_x, edge_y, edge_z = [], [], []
        for i, j in edges:
            edge_x += [x[i], x[j], None]
            edge_y += [y[i], y[j], None]
            edge_z += [z[i], z[j], None]

        edge_plot = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="black", width=1),
            name="Edges",
        )
        data.append(edge_plot)

    fig = go.Figure(data=data)
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    return fig
