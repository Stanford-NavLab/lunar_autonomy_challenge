import numpy as np
import gtsam


class GtsamFactorGraph:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()

        g = 1.622
        IMU_PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
        IMU_PARAMS.setAccelerometerCovariance(0.2 * np.eye(3))
        IMU_PARAMS.setGyroscopeCovariance(0.2 * np.eye(3))
        IMU_PARAMS.setIntegrationCovariance(0.2 * np.eye(3))

        BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 0.4)

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(1000)
        params.setlambdaUpperBound(1.0e6)
        params.setlambdaLowerBound(0.1)
        params.setDiagonalDamping(1000)
        params.setVerbosity("ERROR")
        params.setVerbosityLM("SUMMARY")
        params.setRelativeErrorTol(1.0e-9)
        params.setAbsoluteErrorTol(1.0e-9)

    def add_pose(self, key, pose):
        # TODO: Add a pose to the factor graph
        pass

    def add_imu_factor(self, key, imu_measurement, imu_noise):
        # TODO: Add an IMU factor to the factor graph
        pass

    def add_vision_factor(self, key, visual_measurement, visual_noise):
        # TODO: Add a vision factor to the factor graph
        pass
