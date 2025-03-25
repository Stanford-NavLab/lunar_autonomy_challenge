"""Utilities for 2D/3D geometry"""

import numpy as np


def in_bbox(point, bbox):
    """Check if a point is in a bounding box.

    Parameters
    ----------
    point : np.array (3,)
        3D point
    bbox : np.array (2, 3)
        Min and max points of the bounding box

    Returns
    -------
    bool
        True if the point is in the bounding box

    """
    return np.all(point >= bbox[0]) and np.all(point <= bbox[1])


def crop_points(points, bbox, buffer=0):
    """Crop points to a bounding box.

    Parameters
    ----------
    points : np.array (N, 3)
        3D points
    bbox : np.array (2, 3)
        Min and max points of the bounding box
    buffer : float
        Buffer to add to the bounding box in each dimension

    Returns
    -------
    np.array (M, 3)
        Cropped points

    """
    return points[
        (points[:, 0] >= bbox[0][0] - buffer)
        & (points[:, 0] <= bbox[1][0] + buffer)
        & (points[:, 1] >= bbox[0][1] - buffer)
        & (points[:, 1] <= bbox[1][1] + buffer)
        & (points[:, 2] >= bbox[0][2] - buffer)
        & (points[:, 2] <= bbox[1][2] + buffer)
    ]
