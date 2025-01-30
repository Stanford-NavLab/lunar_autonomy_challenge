"""Plotting functions"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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


def plot_3d_points(points, fig=None, color="blue", markersize=3):
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
        )
    )
    fig.update_layout(width=1200, height=900, scene_aspectmode="data")
    return fig


def pose_trace(pose):
    """Create a plotly trace to visualize a pose

    RGB vectors are used to represent the X, Y, Z axes of the rotation matrix

    Parameters
    ----------
    pose : tuple
        Pose tuple (R, t) where R is a 3x3 rotation matrix and t is a translation

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
        trace = go.Scatter3d(
            x=[arrow_start[0], arrow_end[0]],
            y=[arrow_start[1], arrow_end[1]],
            z=[arrow_start[2], arrow_end[2]],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color=colors[i], width=5),
            showlegend=False,
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
