import gtsam


class GtsamFactorGraph:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()

    def add_pose(self, key, pose):
        # TODO: Add a pose to the factor graph
        pass

    def add_imu_factor(self, key, imu_measurement, imu_noise):
        # TODO: Add an IMU factor to the factor graph
        pass

    def add_vision_factor(self, key, visual_measurement, visual_noise):
        # TODO: Add a vision factor to the factor graph
        pass
