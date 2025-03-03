import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from threading import Thread, Lock
import time
import queue
from collections import deque

from lac.utils.plotting import plot_poses


class Dashboard:
    def __init__(self, update_interval=1.0, history_length=100):
        """
        Initializes a Dash-based Agent Dashboard with a 3D trajectory plot and position error metric.

        Args:
            update_interval (float): Interval (seconds) for updating the UI.
            history_length (int): Number of error values to keep in history for plotting.
        """
        self.update_interval = update_interval
        self.pose_queue = queue.Queue()
        self.metric_queue = queue.Queue()
        self.history_length = history_length
        self.error_history = deque(maxlen=history_length)
        self.error_times = deque(maxlen=history_length)
        self.start_time = time.time()

        # Current state with thread safety
        self.lock = Lock()
        self.gt_poses = None
        self.est_poses = None
        self.latest_error = 0.0
        self.plot_data_changed = False

        # User interaction tracking
        self.last_user_interaction = 0
        self.interaction_cooldown = 2.0  # seconds to wait after user interaction

        # Flag to indicate if dashboard is running
        self.is_running = False

        # Create the Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()

    def setup_layout(self):
        """Sets up the Dash app layout"""
        self.app.layout = html.Div(
            [
                html.H1("🚀 Agent Navigation Dashboard"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("3D Trajectory Plot"),
                                dcc.Graph(
                                    id="trajectory-plot",
                                    style={"height": "70vh"},
                                    config={
                                        "displayModeBar": True,
                                        "scrollZoom": True,
                                        "doubleClick": "reset",
                                    },
                                ),
                                dcc.Store(id="last-interaction-time", data=0),
                                dcc.Store(id="camera-state", data=None),
                            ],
                            className="eight columns",
                        ),
                        html.Div(
                            [
                                html.H3("Position Error"),
                                html.Div(
                                    [
                                        html.Div(id="current-error", className="error-value"),
                                        dcc.Graph(id="error-history-plot"),
                                    ]
                                ),
                            ],
                            className="four columns",
                        ),
                    ],
                    className="row",
                ),
                # Button to manually refresh the 3D plot
                html.Button("Refresh 3D Plot", id="refresh-button", n_clicks=0),
                # Two separate intervals for different components
                dcc.Interval(
                    id="error-interval",
                    interval=int(self.update_interval * 1000),  # in milliseconds
                    n_intervals=0,
                ),
                dcc.Interval(
                    id="trajectory-interval",
                    interval=2000,  # Check every 2 seconds if update is needed
                    n_intervals=0,
                ),
            ]
        )

        # Define callbacks for trajectory plot - only updates when needed
        self.app.callback(
            Output("trajectory-plot", "figure"),
            [Input("trajectory-interval", "n_intervals"), Input("refresh-button", "n_clicks")],
            [
                State("trajectory-plot", "figure"),
                State("last-interaction-time", "data"),
                State("camera-state", "data"),
            ],
        )(self.update_trajectory_plot)

        # Define callbacks for error metrics - updates more frequently
        self.app.callback(
            [Output("current-error", "children"), Output("error-history-plot", "figure")],
            [Input("error-interval", "n_intervals")],
        )(self.update_error_metrics)

        # Callback to track user interactions with the plot
        self.app.callback(
            Output("last-interaction-time", "data"),
            [Input("trajectory-plot", "relayoutData")],
            [State("last-interaction-time", "data")],
        )(self.track_interaction)

        # Callback to save camera state
        self.app.callback(
            Output("camera-state", "data"),
            [Input("trajectory-plot", "relayoutData")],
            [State("camera-state", "data")],
        )(self.save_camera_state)

    def track_interaction(self, relayout_data, current_time):
        """Tracks when user interacts with the plot"""
        if relayout_data and any(k.startswith("scene") for k in relayout_data):
            return time.time()
        return current_time

    def save_camera_state(self, relayout_data, current_state):
        """Saves the camera state when user moves the view"""
        if relayout_data and any(k.startswith("scene") for k in relayout_data):
            # Filter only the camera-related properties
            camera_data = {k: v for k, v in relayout_data.items() if k.startswith("scene")}
            return camera_data
        return current_state

    def update_pose_plot(self, ground_truth_poses, estimated_poses):
        """
        Updates the 3D pose plot with new ground truth and estimated positions.
        """
        with self.lock:
            self.gt_poses = ground_truth_poses
            self.est_poses = estimated_poses
            self.plot_data_changed = True

    def update_metric(self, value):
        """
        Updates the displayed metric (e.g., position error).
        """
        with self.lock:
            self.latest_error = value
            current_time = time.time() - self.start_time
            self.error_history.append(value)
            self.error_times.append(current_time)

    def update_trajectory_plot(
        self, n_intervals, n_clicks, current_figure, last_interaction_time, camera_state
    ):
        """
        Callback to update the trajectory plot - only updates when needed and not during user interaction.
        """
        # Check if user is currently interacting with the plot
        user_inactive = (
            (time.time() - last_interaction_time) > self.interaction_cooldown
            if last_interaction_time
            else True
        )

        # Only update if data changed and user is not interacting, or if refresh button clicked
        with self.lock:
            data_changed = self.plot_data_changed
            self.plot_data_changed = False
            gt_poses = self.gt_poses
            est_poses = self.est_poses

        # If no need to update, keep current figure
        if not data_changed and not n_clicks:
            return current_figure

        # If user is interacting and this isn't a manual refresh, keep current figure
        if not user_inactive and not n_clicks:
            with self.lock:
                self.plot_data_changed = True  # Flag for update on next interval
            return current_figure

        # Create new figure
        fig_trajectory = go.Figure()

        if gt_poses is not None:
            fig_trajectory = plot_poses(
                gt_poses, fig=fig_trajectory, color="blue", no_axes=True, name="Ground Truth"
            )

        if est_poses is not None:
            fig_trajectory = plot_poses(
                est_poses, fig=fig_trajectory, color="orange", no_axes=True, name="Estimated"
            )

        # Apply layout
        fig_trajectory.update_layout(
            title="3D Trajectory",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        )

        # Apply saved camera state if available
        if camera_state and not n_clicks:  # Don't apply camera state on manual refresh
            fig_trajectory.update_layout(**camera_state)

        return fig_trajectory

    def update_error_metrics(self, n_intervals):
        """
        Callback to update just the error metrics.
        """
        with self.lock:
            error_history = list(self.error_history)
            error_times = list(self.error_times)
            latest_error = self.latest_error

        # Format current error display
        current_error_display = html.Div(
            [html.H4("Current Error:"), html.H2(f"{latest_error:.3f}", style={"color": "red"})]
        )

        # Create error history plot
        fig_error = go.Figure()
        if len(error_history) > 0:
            fig_error.add_trace(
                go.Scatter(
                    x=error_times,
                    y=error_history,
                    mode="lines",
                    line=dict(color="red", width=2),
                    name="Position Error",
                )
            )

        fig_error.update_layout(
            title="Error History",
            xaxis_title="Time (s)",
            yaxis_title="Error",
            margin=dict(l=30, r=10, b=30, t=30),
            height=300,
        )

        return current_error_display, fig_error

    def start(self, port=8050, debug=False):
        """
        Starts the dashboard in a separate thread.
        """
        if not self.is_running:
            self.is_running = True
            self.dashboard_thread = Thread(target=self._run_server, args=(port, debug))
            self.dashboard_thread.daemon = True
            self.dashboard_thread.start()
            print(f"Dashboard started at http://localhost:{port}")

    def _run_server(self, port, debug):
        """
        Run the Dash server in the thread.
        """
        self.app.run_server(port=port, debug=debug, use_reloader=False)

    def stop(self):
        """
        Stops the dashboard server.
        """
        self.is_running = False
