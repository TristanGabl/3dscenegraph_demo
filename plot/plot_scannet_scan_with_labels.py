import open3d as o3d
import numpy as np
from plyfile import PlyData
import pandas as pd
import json
import random
import os

# Set the current working directory to the file's directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

scene = "scene0000_02"
# ground_truth
ply0 = PlyData.read("results_folder/" + scene + "/plot/" + scene + "_vh_clean_2.labels.ply")
# predicted
ply1 = PlyData.read("results_folder/" + scene + "/plot/" + scene + "_pointcloud_classes_with_scannet_ids.ply")


verts0 = np.stack([ply0['vertex'].data[name] for name in ('x', 'y', 'z')], axis=-1)
labels0 = ply0['vertex'].data['label']
labels0 = np.where(labels0 == 0, -1, labels0)

verts1 = np.stack([ply1['vertex'].data[name] for name in ('x', 'y', 'z')], axis=-1)
labels1 = ply1['vertex'].data['label']
labels1 = np.where(labels1 == 20, 2, labels1)
labels1 = np.where(labels1 == 37, 40, labels1) 

scannet_id_to_name = json.load(open("label_mapping/scannet_id_to_name.json", 'r'))
scannet_id_to_color = json.load(open("label_mapping/scannet_id_to_color.json", 'r'))
# Shuffle scannet_id_to_color
scannet_id_to_color_no_background = {k: v for k, v in scannet_id_to_color.items() if k != "-1"}
random.seed(45)
shuffled_colors = list(scannet_id_to_color_no_background.values())
random.shuffle(shuffled_colors)
scannet_id_to_color_no_background = {key: shuffled_colors[i] for i, key in enumerate(scannet_id_to_color_no_background.keys())}
scannet_id_to_color_no_background["-1"] = [0, 0, 0]  # Ensure background is black
scannet_id_to_color = scannet_id_to_color_no_background


# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
# Add both points with an offset for verts0
offset = np.array([0.0, 8.0, 0.0])  # Example offset along the x-axis
verts0_offset = verts0 + offset

# Combine both point clouds to be visualized together
combined_verts = np.vstack((verts0_offset, verts1))
pcd.points = o3d.utility.Vector3dVector(combined_verts)

# Combine labels for coloring for continuous color mapping
labels = np.hstack((labels0, labels1)).astype(np.int64)


# Print colors with their corresponding labels
unique_labels = np.unique(labels)
for label in unique_labels:
    if f"{label}" in scannet_id_to_color:
        color = scannet_id_to_color[f"{label}"]
        name = scannet_id_to_name[f"{label}"]
        print(f"\033[38;2;{color[0]};{color[1]};{color[2]}mLabel: {name}({label}), Color: {color}\033[0m")
    else:
        print(f"\033[38;2;0;0;0mLabel: {name}, Color: [0, 0, 0] (Default gray)\033[0m")

pcd.colors = o3d.utility.Vector3dVector(
    np.array([scannet_id_to_color[f"{label}"] for label in labels]) / 255.0
)

# Compute Intersection over Union (IoU)
def compute_iou(labels_gt, labels_pred):
    unique_labels = np.union1d(labels_gt, labels_pred)
    iou_scores = {}
    for label in unique_labels:
        intersection = np.sum((labels_gt == label) & (labels_pred == label))
        union = np.sum((labels_gt == label) | (labels_pred == label))
        if union > 0:
            iou_scores[label] = intersection / union
        else:
            iou_scores[label] = 0.0
    return iou_scores

# Compute IoU for ground truth and predicted labels
iou_scores = compute_iou(labels0, labels1)

# Print IoU scores
for label, iou in iou_scores.items():
    name = scannet_id_to_name.get(f"{label}", "background")
    print(f"Label: {name}, IoU: {iou:.4f}")

# Visualize
o3d.visualization.draw_geometries([pcd])
