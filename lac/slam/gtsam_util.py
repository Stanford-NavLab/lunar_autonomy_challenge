"""

Notes on GTSAM:



"""

import numpy as np
import gtsam
import gtsam_unstable
from gtsam.symbol_shorthand import B, V, X, L

from gtsam import (
    Cal3_S2,
    DoglegOptimizer,
    LevenbergMarquardtOptimizer,
    GenericProjectionFactorCal3_S2,
    NonlinearFactorGraph,
    PriorFactorPoint3,
    PriorFactorPose3,
    Values,
    Pose3,
)

from lac.params import FL_X, FL_Y, IMG_HEIGHT, IMG_WIDTH
from lac.utils.frames import get_cam_pose_rover, CAMERA_TO_OPENCV_PASSIVE


# Constants and parameters
K = Cal3_S2(FL_X, FL_Y, 0.0, IMG_WIDTH / 2, IMG_HEIGHT / 2)

g = 1.622
IMU_PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
gyro_sigma = 1e-8
accel_sigma = 1e-8
integration_sigma = 1e-10
IMU_PARAMS.setGyroscopeCovariance(gyro_sigma**2 * np.eye(3))
IMU_PARAMS.setAccelerometerCovariance(accel_sigma**2 * np.eye(3))
IMU_PARAMS.setIntegrationCovariance(integration_sigma**2 * np.eye(3))
