"""Shared constants and parameters"""

import numpy as np
import json
import os
from dataclasses import dataclass

#BASE_PATH_LAC = "~/LunarAutonomyChallenge"
BASE_PATH_LAC = "~/Documents/sw_navlab/LunarAutonomyChallenge"

""" Constants """
LUNAR_GRAVITY = np.array([0.0, 0.0, 1.6220])  # m/s^2

FRAME_RATE = 20  # frames per second

STEREO_BASELINE = 0.162  # meters
IMG_WIDTH = 1280
IMG_HEIGHT = 720
IMG_FOV = 1.22173  # radians (70 degrees)
FL_X = IMG_WIDTH / (2 * np.tan(IMG_FOV / 2))  # Horizontal focal length
FL_Y = FL_X  # Vertical focal length  (square pixels)
CAMERA_INTRINSICS = np.array([[FL_X, 0, IMG_WIDTH / 2], [0, FL_Y, IMG_HEIGHT / 2], [0, 0, 1]])

GEOMETRY_DICT = json.load(open(os.path.expanduser(BASE_PATH_LAC + "/docs/geometry.json")))

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


""" Parameters """
# Maximum area of a rock segmentation mask in pixels, anything larger is ignored
ROCK_MASK_MAX_AREA = 50000

# Minimum area of a rock segmentation mask in pixels to be considered for obstacle avoidance
ROCK_MASK_AVOID_MIN_AREA = 1000

MAX_DEPTH = 56.5  # Maximum depth value in meters (40 * sqrt(2))
