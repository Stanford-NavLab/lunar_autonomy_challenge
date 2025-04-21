"""Frontend for SLAM"""

from lac.slam.feature_tracker import FeatureTracker
from lac.perception.segmentation import UnetSegmentation, SemanticClasses


class Frontend:
    """Frontend for SLAM"""

    def __init__(self, camera_config: dict):
        """Initialize the frontend"""
        self.feature_tracker = FeatureTracker(camera_config)
        self.segmentation = UnetSegmentation()

    def initialize(self, initial_pose: dict, left_image: dict, right_image: dict):
        """Initialize the frontend"""
        self.feature_tracker.initialize(initial_pose, left_image, right_image)
        self.segmentation.initialize()

    def process(self, data):
        """Process the data"""
        odometry = self.feature_tracker.track_pnp(data["left_image"], data["right_image"])

        # If VO failed, use IMU odometry instead
        if odometry is None:
            odometry = data["imu_odometry"]

        # Segmentation
        pred = self.segmentation.predict(data["left_image"])
