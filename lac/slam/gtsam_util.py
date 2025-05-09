"""

Notes on GTSAM:



"""

import numpy as np
import gtsam
import matplotlib.pyplot as plt
from collections import defaultdict

from lac.params import FL_X, FL_Y, IMG_HEIGHT, IMG_WIDTH


# Constants and parameters
K = gtsam.Cal3_S2(FL_X, FL_Y, 0.0, IMG_WIDTH / 2, IMG_HEIGHT / 2)

# rotation [rad], translation [m]
VO_ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.0004, 0.00027, 0.00033, 0.0012, 0.001, 0.0007])
)
# # NOTE: measured based on IMU integration
# IMU_ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
#     np.array([0.004, 0.0015, 0.0015, 0.004, 0.004, 0.001])
# )
IMU_ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(0.1 * np.ones(6))
LOOP_CLOSURE_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.00087, 0.00087, 0.00087, 0.005, 0.005, 0.005])
)

g = 1.622
IMU_PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
gyro_sigma = 1e-8
accel_sigma = 1e-8
integration_sigma = 1e-10
IMU_PARAMS.setGyroscopeCovariance(gyro_sigma**2 * np.eye(3))
IMU_PARAMS.setAccelerometerCovariance(accel_sigma**2 * np.eye(3))
IMU_PARAMS.setIntegrationCovariance(integration_sigma**2 * np.eye(3))


def remove_outliers(graph, result, threshold=5.0):
    """
    Remove outliers from the graph based on the pixel error.
    Args:
        graph (gtsam.NonlinearFactorGraph): The factor graph.
        result (gtsam.Values): The current estimate.
        threshold (float): The threshold for outlier removal.
    """
    cleaned_graph = gtsam.NonlinearFactorGraph()

    # Prune the factors
    for i in range(graph.size()):
        factor = graph.at(i)
        if isinstance(factor, gtsam.GenericProjectionFactorCal3_S2):
            residual = factor.unwhitenedError(result)
            pixel_error = np.linalg.norm(residual)

            if pixel_error < threshold:
                cleaned_graph.add(factor)  # Keep inlier
            # Else: skip (it's an outlier)
        else:
            # Keep non-vision factors (e.g., priors, odometry)
            cleaned_graph.add(factor)

    # Prune the keys
    used_keys = set()
    for i in range(cleaned_graph.size()):
        factor = cleaned_graph.at(i)
        for k in factor.keys():
            used_keys.add(k)

    cleaned_estimate = gtsam.Values()
    for key in result.keys():
        if key in used_keys:
            symbol = gtsam.Symbol(key)
            if symbol.chr() == ord("x"):
                value = result.atPose3(key)
            elif symbol.chr() == ord("l"):
                value = result.atPoint3(key)
            else:
                raise ValueError("Unknown symbol type")
            cleaned_estimate.insert(key, value)

    return cleaned_graph, cleaned_estimate


def find_factors_for_key(graph: gtsam.NonlinearFactorGraph, key: int):
    """
    Return a list of indices (and/or factors) that contain the given 'key'.

    Example usage:
        landmark_key = gtsam.Symbol('l', 19648).key()
        indices = find_factors_for_key(graph, landmark_key)

    """
    matching_factor_indices = []
    for i in range(graph.size()):
        factor = graph.at(i)
        factor_keys = factor.keys()
        if key in factor_keys:
            matching_factor_indices.append(i)

    return matching_factor_indices


def plot_reprojection_residuals(graph, result, ymax=None):
    """Plot reprojection residuals for each pose in the graph."""
    pose_errors = defaultdict(list)

    for i in range(graph.size()):
        factor = graph.at(i)
        if isinstance(factor, gtsam.GenericProjectionFactorCal3_S2):
            # Compute unwhitened pixel error (2D residual)
            residual = factor.unwhitenedError(result)
            error_pixels = np.linalg.norm(residual)

            # Get the pose key used in this factor
            pose_key = factor.keys()[0]  # index 0 is the pose key (camera pose)

            # Accumulate error
            pose_errors[pose_key].append(error_pixels)

    # Sort by key (e.g., X(1), X(2), ...)
    pose_keys = sorted(pose_errors.keys(), key=lambda k: gtsam.Symbol(k).index())
    mean_errors = [np.mean(pose_errors[k]) for k in pose_keys]
    max_errors = [np.max(pose_errors[k]) for k in pose_keys]
    pose_indices = [gtsam.Symbol(k).index() for k in pose_keys]  # numeric frame index

    plt.figure(figsize=(10, 4))
    plt.plot(pose_indices, mean_errors, label="Mean Reprojection Error (pixels)")
    plt.plot(pose_indices, max_errors, label="Max Reprojection Error (pixels)", linestyle="--")
    plt.xlabel("Pose Index")
    plt.ylabel("Reprojection Error (pixels)")
    plt.title("Reprojection Error per Pose")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if ymax is not None:
        plt.ylim(0, ymax)
    plt.show()
