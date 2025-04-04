"""Given a data collection folder, select a subset of images and place into a new folder"""

import os
from pathlib import Path

if __name__ == "__main__":
    data_path = Path("/home/shared/data_raw/LAC/runs/full_spiral_map1_preset0_recovery_agent")
    cameras = ["FrontLeft", "FrontRight"]
    START_FRAME = 240
    END_FRAME = 2000
    STEP = 40
    OUTPUT_FOLDER = os.path.expanduser("~/opt/vggt/examples/LAC/spiral_subset")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for camera in cameras:
        camera_path = data_path / camera
        for frame in range(START_FRAME, END_FRAME, STEP):
            img_name = f"{frame:06}.png"
            img_path = camera_path / img_name
            # Copy the image to output folder as {camera}_{frame}.png
            output_img_name = f"{camera}_{frame:06}.png"
            output_img_path = Path(OUTPUT_FOLDER) / output_img_name
            os.symlink(img_path, output_img_path)
