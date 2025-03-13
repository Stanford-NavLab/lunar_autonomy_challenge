"""Generate groundtruth.txt file for pyslam

Format is [timestamp, x, y, z, qx, qy, qz, qw, scale]

"""

import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation

from lac.util import load_data
from lac.params import LAC_BASE_PATH


data_path = Path(LAC_BASE_PATH) / "output/DataCollectionAgent/map1_preset0_nolight"
# initial_pose, lander_pose, poses, imu_data, cam_config = load_data(data_path)
json_data = json.load(open(f"{data_path}/data_log.json"))

# ENU_TO_RDF = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # Passive
# ENU_TO_RDF = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # Active?
ENU_TO_RDF = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])  # Hacked

output = []
t_prev = None
start_frame = 80

# for pose in poses:
for frame in json_data["frames"]:
    if frame["step"] % 2 == 0 and frame["step"] >= start_frame:
        pose = np.array(frame["pose"])
        R = ENU_TO_RDF @ pose[:3, :3] @ ENU_TO_RDF.T
        t = ENU_TO_RDF @ pose[:3, 3]
        q = Rotation.from_matrix(R).as_quat(scalar_first=False)  # scalar last
        timestamp = frame["mission_time"]
        if t_prev is not None:
            scale = np.linalg.norm(t - t_prev)
        else:
            scale = 1.0
        t_prev = t
        output.append([timestamp, *t, *q, scale])

np.savetxt(f"{data_path}/groundtruth.txt", output, fmt="%.6f")
print(f"Saved groundtruth.txt to {data_path}")
