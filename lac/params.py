"""Shared constants and parameters"""

import numpy as np
from dataclasses import dataclass


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

TAG_SIZE = 0.339  # meters

# Bottom of wheel points in robot frame
WHEEL_RIG_POINTS = np.array(
    [
        [0.222, 0.203, -0.134],
        [0.222, -0.203, -0.134],
        [-0.222, 0.203, -0.134],
        [-0.222, 0.203, -0.134],
    ]
)
WHEEL_RIG_COORDS = np.concatenate((WHEEL_RIG_POINTS.T, np.ones((1, 4))), axis=0)


""" Parameters """
# Maximum area of a rock segmentation mask in pixels, anything larger is ignored
ROCK_MASK_MAX_AREA = 50000

# Minimum area of a rock segmentation mask in pixels to be considered for obstacle avoidance
ROCK_MASK_AVOID_MIN_AREA = 1000

MAX_DEPTH = 56.5  # Maximum depth value in meters (40 * sqrt(2))
