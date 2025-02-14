"""Plotting functions"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

##### ------------------- 2D ------------------- #####


def plot_heatmap(data, fig=None, colorscale="Viridis", no_axes=False):
    """Plot a 2D heatmap."""
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Heatmap(z=data, colorscale=colorscale))
    fig.update_layout(width=1200, height=900)
    if no_axes:
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


##### ------------------- 3D ------------------- #####


def plot_surface(x, y, z, fig=None, colorscale="Viridis", no_axes=False, showscale=True, **kwargs):
    """
    x, y, z are 2D arrays representing the coordinates and elevation data for a surface plot.

    """
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=colorscale, showscale=showscale, **kwargs))
    fig.update_layout(width=1600, height=900, scene_aspectmode="data")
    if no_axes:
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
            )
        )
    return fig


def plot_path_3d(
    x, y, z, fig=None, color="red", markersize=3, linewidth=3, markers=True, name=None, **kwargs
):
    """Plot a 3D path with optional markers."""
    if fig is None:
        fig = go.Figure()
    if markers:
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+lines",
                marker=dict(size=markersize, color=color),
                line=dict(color=color, width=linewidth),
                name=name,
                **kwargs,
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color=color, width=linewidth),
                name=name,
                **kwargs,
            )
        )
    return fig


def plot_3d_points(points, fig=None, color="blue", markersize=3, name=None):
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


def pose_trace(pose, name: str = "", line_style: str = "solid", line_width: int = 5):
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
                mode="lines+markers",
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
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color=colors[i], width=line_width, dash=line_style),
                name=name + f"_{AXES_NAMES[i]}",
                showlegend=True,    
            )
        traces.append(trace)

    return traces


def pose_traces(pose_list):
    """Create traces for a list of poses

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
        traces = pose_trace(pose)
        all_traces.extend(traces)

    return all_traces


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
