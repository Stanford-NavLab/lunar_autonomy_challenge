"""Shared constants and parameters"""

import numpy as np
import json
import os


LAC_BASE_PATH = (
    os.getenv("LAC_BASE_PATH")
    if os.getenv("LAC_BASE_PATH")
    else os.path.expanduser("~/LunarAutonomyChallenge")
)

"""-------------------------------------- Constants --------------------------------------"""
LUNAR_GRAVITY = np.array([0.0, 0.0, 1.6220])  # m/s^2

FRAME_RATE = 20  # fps of the simulation
DT = 1 / FRAME_RATE  # time step

STEREO_BASELINE = 0.162  # meters
IMG_FOV_RAD = 1.22173  # [rad] (70 degrees)
MAX_IMG_WIDTH = 2448
MAX_IMG_HEIGHT = 2048

"""-------------------------------------- Data Structures --------------------------------------"""

GEOMETRY_DICT = json.load(open(os.path.expanduser(LAC_BASE_PATH + "/docs/geometry.json")))

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
    "FrontLeft": {"active": False, "light": 0.0, "width": 1280, "height": 720, "semantic": False},
    "FrontRight": {"active": False, "light": 0.0, "width": 1280, "height": 720, "semantic": False},
    "BackLeft": {"active": False, "light": 0.0, "width": 1280, "height": 720, "semantic": False},
    "BackRight": {"active": False, "light": 0.0, "width": 1280, "height": 720, "semantic": False},
    "Left": {"active": False, "light": 0.0, "width": 1280, "height": 720, "semantic": False},
    "Right": {"active": False, "light": 0.0, "width": 1280, "height": 720, "semantic": False},
    "Front": {"active": False, "light": 0.0, "width": 1280, "height": 720, "semantic": False},
    "Back": {"active": False, "light": 0.0, "width": 1280, "height": 720, "semantic": False},
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
MAX_STEER = 1.0  # [rad/s]
MAX_STEER_DELTA = 0.6  # [rad/s]
WAYPOINT_REACHED_DIST_THRESHOLD = 1.0  # distance threshold for moving to next waypoint [m]

# Maximum area of a rock segmentation mask in pixels, anything larger is ignored
ROCK_MASK_MAX_AREA = 50000

# Minimum area of a rock segmentation mask in pixels to be considered for obstacle avoidance
ROCK_MASK_AVOID_MIN_AREA = 1000

MAX_DEPTH = 56.5  # Maximum depth value in meters (40 * sqrt(2))
