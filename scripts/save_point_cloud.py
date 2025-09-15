import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import viser

from lac.perception.segmentation import SemanticClasses
from lac.slam.backend import SemanticPointCloud
from lac.mapping.mapper import bin_points_to_grid, nanmedian_filter, process_map
from lac.mapping.map_utils import get_geometric_score, get_rocks_score
from lac.mapping.interpolation import interpolate_heights, interpolate_heights_rbf
from lac.utils.plotting import plot_heatmap, plot_surface, plot_rock_maps
from lac.util import load_data


if __name__ == "__main__":
    data_path = "/home/shared/data_raw/LAC/runs/2025-05-28_11-59-12"
    initial_pose, lander_pose, poses, imu_data, cam_config, json_data = load_data(data_path)
    semantic_points = SemanticPointCloud.from_file(Path(data_path) / "semantic_points.npz")

    # Downsample points by taking every 10th point
    downsampled_points = semantic_points.points[::10]
    downsampled_labels = semantic_points.labels[::10]

    # Filter out sky points (label 4)
    valid_mask = (downsampled_labels != 4) & (downsampled_labels != 0)
    downsampled_points = downsampled_points[valid_mask]
    downsampled_labels = downsampled_labels[valid_mask]

    # Create color mapping
    colors = {
        1: [1.0, 0.0, 0.0],  # red for rocks
        2: [1.0, 0.843, 0.0],  # gold for lander
        3: [0.5, 0.5, 0.5],  # gray for ground
    }

    # Convert labels to colors
    point_colors = np.array([colors[label] for label in downsampled_labels])

    # ---- normalize dtypes ----
    pts = downsampled_points.astype(np.float32)
    cols = point_colors
    if cols.dtype != np.uint8:
        cols = np.clip(cols * 255.0, 0, 255).astype(np.uint8)

    assert pts.shape[1] == 3 and cols.shape[1] == 3 and len(pts) == len(cols)

    # ---- write ASCII PLY (simple & portable) ----
    out_path = "cloud.ply"  # place this inside your repo
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts, cols):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    print("Wrote", out_path)
