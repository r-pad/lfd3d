import numpy as np
import open3d as o3d
from lfd3d.datasets.lerobot.lerobot_dataset import RpadLeRobotDataset
from omegaconf import OmegaConf
from tqdm import tqdm


robot_points = np.load("/home/haotian/lfd3d/robot_points.npy")
human_points = np.load("/home/haotian/lfd3d/human_points.npy")
scene_points = np.load("/home/haotian/lfd3d/scene_points.npy")


vis_points = []

for point in robot_points:
    action_pcd = o3d.geometry.PointCloud()
    action_pcd.points = o3d.utility.Vector3dVector(point.copy())
    action_pcd.paint_uniform_color([1, 0, 0])  # RGB: Red
    vis_points.append(action_pcd)

for point in human_points:
    action_pcd = o3d.geometry.PointCloud()
    action_pcd.points = o3d.utility.Vector3dVector(point.copy())
    action_pcd.paint_uniform_color([0, 0, 1])  # RGB: Blue
    vis_points.append(action_pcd)

anchor_pcd = o3d.geometry.PointCloud()
anchor_pcd.points = o3d.utility.Vector3dVector(scene_points)
anchor_pcd.paint_uniform_color([0.5, 0.5, 0.5])
vis_points.append(anchor_pcd)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
vis_points.append(frame)

o3d.visualization.draw_geometries(vis_points)