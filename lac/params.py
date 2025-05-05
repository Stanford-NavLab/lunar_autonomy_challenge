"""Shared constants and parameters

TODO: keep only constants here, and move parameters to config json

"""

import numpy as np
import json
import os
from pathlib import Path


TEAM_CODE_ROOT = Path(os.path.abspath(__file__)).parent.parent
# "/workspace/team_code" - Location in the docker container

LAC_BASE_PATH = TEAM_CODE_ROOT.parent

DEFAULT_RUN_NAME = "default_run"

"""-------------------------------------- Constants --------------------------------------"""
LUNAR_GRAVITY = np.array([0.0, 0.0, 1.6220])  # m/s^2

FRAME_RATE = 20  # fps of the simulation
DT = 1 / FRAME_RATE  # time step

STEREO_BASELINE = 0.162  # meters
IMG_FOV_RAD = 1.22173  # [rad] (70 degrees)
MAX_IMG_WIDTH = 2448
MAX_IMG_HEIGHT = 2048

SCENE_MAX_X = 20.0  # [m]
SCENE_MIN_X = -20.0  # [m]
SCENE_MAX_Y = 20.0  # [m]
SCENE_MIN_Y = -20.0  # [m]
SCENE_MAX_Z = 10.0  # [m]  (could probably reduce this, based on max height of lander)
SCENE_MIN_Z = 0.0  # [m]
SCENE_BBOX = np.array(
    [[SCENE_MIN_X, SCENE_MIN_Y, SCENE_MIN_Z], [SCENE_MAX_X, SCENE_MAX_Y, SCENE_MAX_Z]]
)

# Lander dimensions
LANDER_WIDTH = 3.0  # [m]  (approximate)
LANDER_HEIGHT = 3.0  # [m]  (approximate)
LANDER_GLOBAL = np.array(
    [
        [-LANDER_WIDTH / 2, LANDER_WIDTH / 2, 0.0],  # top left
        [LANDER_WIDTH / 2, LANDER_WIDTH / 2, 0.0],  # top right
        [LANDER_WIDTH / 2, -LANDER_WIDTH / 2, 0.0],  # bottom right
        [-LANDER_WIDTH / 2, -LANDER_WIDTH / 2, 0.0],  # bottom left
    ]
)

ROVER_RADIUS = 0.75  # [m]

CELL_WIDTH = 0.15  # [m] width of each cell in the map
MAP_EXTENT = 13.5  # [m] extent of the map in x and y directions
MAP_SIZE = 180  # number of cells in each direction
HEIGHT_ERROR_TOLERANCE = 0.05  # [m] tolerance for height error

"""-------------------------------------- Data Structures --------------------------------------"""

# GEOMETRY_DICT = json.load(open(os.path.expanduser(LAC_BASE_PATH + "/docs/geometry.json")))
current_file = Path(__file__)
current_dir = current_file.parent
GEOMETRY_DICT = json.load(open(current_dir / "../docs/geometry.json"))

# These angles are listed clockwise (starting with 0 at lander +Y-axis)
TAG_GROUP_BEARING_ANGLES = {
    "a": 135,
    "b": 45,
    "c": 315,
    "d": 225,
}
TAG_LOCATIONS = {}
for group, group_vals in GEOMETRY_DICT["lander"]["fiducials"].items():
    for tag, tag_vals in group_vals.items():
        TAG_LOCATIONS[tag_vals["id"]] = {
            "center": np.array([tag_vals["x"], tag_vals["y"], tag_vals["z"]]),
            "bearing": TAG_GROUP_BEARING_ANGLES[group],
            "size": tag_vals["size"],
        }
locator_tag = GEOMETRY_DICT["lander"]["locator"]
TAG_LOCATIONS[locator_tag["id"]] = {
    "center": np.array([locator_tag["x"], locator_tag["y"], locator_tag["z"]]),
    "bearing": 0,
    "size": locator_tag["size"],
}

# Bottom of wheel points in robot frame
WHEEL_RIG_POINTS = np.array(
    [
        [0.222, 0.203, -0.134],
        [0.222, -0.203, -0.134],
        [-0.222, 0.203, -0.134],
        [-0.222, 0.203, -0.134],
    ]
)

CAMERA_CONFIG_INIT = {
    "FrontLeft": {
        "active": False,
        "light": 0.0,
        "width": 1280,
        "height": 720,
        "semantic": False,
    },
    "FrontRight": {
        "active": False,
        "light": 0.0,
        "width": 1280,
        "height": 720,
        "semantic": False,
    },
    "BackLeft": {
        "active": False,
        "light": 0.0,
        "width": 1280,
        "height": 720,
        "semantic": False,
    },
    "BackRight": {
        "active": False,
        "light": 0.0,
        "width": 1280,
        "height": 720,
        "semantic": False,
    },
    "Left": {
        "active": False,
        "light": 0.0,
        "width": 1280,
        "height": 720,
        "semantic": False,
    },
    "Right": {
        "active": False,
        "light": 0.0,
        "width": 1280,
        "height": 720,
        "semantic": False,
    },
    "Front": {
        "active": False,
        "light": 0.0,
        "width": 1280,
        "height": 720,
        "semantic": False,
    },
    "Back": {
        "active": False,
        "light": 0.0,
        "width": 1280,
        "height": 720,
        "semantic": False,
    },
}


"""-------------------------------------- Parameters --------------------------------------"""
ARM_ANGLE_STATIC_RAD = 1.0472  # [rad] (60 degrees)

# TODO: these should be settable parameters
IMG_WIDTH = 1280
IMG_HEIGHT = 720
FL_X = IMG_WIDTH / (2 * np.tan(IMG_FOV_RAD / 2))  # Horizontal focal length
FL_Y = FL_X  # Vertical focal length  (same as horizontal because square pixels)
CAMERA_INTRINSICS = np.array([[FL_X, 0, IMG_WIDTH / 2], [0, FL_Y, IMG_HEIGHT / 2], [0, 0, 1]])

# Controller parameters
KP_STEER = 0.3
KP_LINEAR = 0.1
TARGET_SPEED = 0.2  # [m/s]
MAX_STEER = 1.2  # [rad/s]
MAX_STEER_DELTA = 1.0  # [rad/s]
WAYPOINT_REACHED_DIST_THRESHOLD = 1.5  # distance threshold for moving to next waypoint [m]

# Maximum area of a rock segmentation mask in pixels, anything larger is ignored
ROCK_MASK_MAX_AREA = 50000

# Minimum area of a rock segmentation mask in pixels to be considered for obstacle avoidance
ROCK_MASK_AVOID_MIN_AREA = 1000
ROCK_MIN_RADIUS = 0.08  # [m] minimum radius of a rock to be considered for obstacle avoidance

ROCK_AVOID_DIST = 2.0  # [m] distance to avoid rocks
ROCK_BRIGHTNESS_THRESHOLD = 50  # [0-255] pixel threshold for segmentation to be consider rock

MAX_DEPTH = 56.5  # Maximum depth value in meters (40 * sqrt(2))


# EKF parameters
EKF_SMOOTHING_INTERVAL = 10  # Run smoothing every N steps
# TODO: document/organize these parameters
EKF_INIT_R = 0.001  # Initial position std
EKF_INIT_V = 0.01  # Initial velocity std
EKF_INIT_ANGLE = 0.001  # Initial euler angle std
EKF_P0 = np.diag(
    np.hstack(
        (
            np.ones(3) * EKF_INIT_R * EKF_INIT_R,
            np.ones(3) * EKF_INIT_V * EKF_INIT_V,
            np.ones(3) * EKF_INIT_ANGLE * EKF_INIT_ANGLE,
        )
    )
)
EKF_Q_SIGMA_A = 0.03  # Acceleration std
EKF_Q_SIGMA_ANGLE = 0.00005  # Angle std
EKF_R_SIGMAS = np.array([0.25, 0.25, 0.25, 0.05, 0.05, 0.2])  # Measurement stds
