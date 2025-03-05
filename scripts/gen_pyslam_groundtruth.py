"""Generate groundtruth.txt file for pyslam

Format is [timestamp, x, y, z, qx, qy, qz, qw, scale]

"""

import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation

from lac.util import load_data
from lac.params import LAC_BASE_PATH


# data_path = "../../output/LocalizationAgent/map1_preset0_4m_spiral"
data_path = Path(LAC_BASE_PATH) / "output/Old/LocalizationAgent_spiral_norocks"
# initial_pose, lander_pose, poses, imu_data, cam_config = load_data(data_path)
json_data = json.load(open(f"{data_path}/data_log.json"))


output = []
t_prev = None

# for pose in poses:
for frame in json_data["frames"]:
    pose = np.array(frame["pose"])
    R = pose[:3, :3]
    t = pose[:3, 3]
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
