"""
TODO: move code from data_collection_agent into modular class which can be initialized
in any agent and logs data.

Logs data (IMU, wheel odometry, ground-truth pose, power, control inputs) to a json file.
Saves images from active cameras to folders.

"""

import os
import shutil
import time
import json
import cv2 as cv
import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.util import transform_to_numpy
from lac.params import DEFAULT_RUN_NAME


class DataLogger:
    def __init__(self, agent: AutonomousAgent, agent_name: str, camera_config: dict):
        self.agent_name = agent_name
        self.data = {}
        self.frames = []
        self.run_name = DEFAULT_RUN_NAME
        self.log_file = f"output/{self.agent_name}/{self.run_name}/data_log.json"

        self.agent = agent
        self.cameras = camera_config

        # Log initial data
        initial_rover_pose = transform_to_numpy(self.agent.get_initial_position())
        lander_pose_rover = transform_to_numpy(self.agent.get_initial_lander_position())
        lander_pose_world = initial_rover_pose @ lander_pose_rover
        self.data["initial_pose"] = initial_rover_pose.tolist()
        self.data["lander_pose_rover"] = lander_pose_rover.tolist()
        self.data["lander_pose_world"] = lander_pose_world.tolist()
        self.data["cameras"] = self.cameras
        self.data["use_fiducials"] = self.agent.use_fiducials()

        # Initialize output folders
        if os.path.exists(f"output/{self.agent_name}/{self.run_name}"):
            shutil.rmtree(f"output/{self.agent_name}/{self.run_name}")
        for cam_name, config in self.cameras.items():
            if config["active"]:
                os.makedirs(f"output/{self.agent_name}/{self.run_name}/{cam_name}")
                if config["semantic"]:
                    os.makedirs(f"output/{self.agent_name}/{self.run_name}/{cam_name}_semantic")

    def log_data(self, step: int, control: carla.VehicleVelocityControl):
        """
        step - current step in the simulation
        """
        log_entry = {
            "step": step,
            "timestamp": time.time(),
            "mission_time": self.agent.get_mission_time(),
            "current_power": self.agent.get_current_power(),
            "pose": transform_to_numpy(self.agent.get_transform()).tolist(),
            "imu": self.agent.get_imu_data().tolist(),
            "control": {"v": control.linear_target_velocity, "w": control.angular_target_velocity},
            "linear_speed": self.agent.get_linear_speed(),
            "angular_speed": self.agent.get_angular_speed(),
        }
        self.frames.append(log_entry)

    def log_images(self, step: int, input_data: dict):
        """
        input_data - input from run_step
        """
        for cam_name, config in self.cameras.items():
            if config["active"]:
                img = input_data["Grayscale"][getattr(carla.SensorPosition, cam_name)]
                cv.imwrite(
                    f"output/{self.agent_name}/{self.run_name}/{cam_name}/{step:06}.png",
                    img,
                )
                if config["semantic"]:
                    semantic_img = input_data["Semantic"][getattr(carla.SensorPosition, cam_name)]
                    cv.imwrite(
                        f"output/{self.agent_name}/{self.run_name}/{cam_name}_semantic/{step:06}.png",
                        semantic_img,
                    )

    def save_log(self):
        with open(self.log_file, "w") as f:
            self.data["frames"] = self.frames
            json.dump(self.data, f, indent=4)
