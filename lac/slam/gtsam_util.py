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
