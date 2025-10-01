from datetime import datetime

import cv2
import numpy as np
import torch
import trimesh
from matplotlib import cm
from pytorch3d.ops import sample_farthest_points
from typing import List, Union
import imageio.v3 as iio

def project_pcd_on_image(pcd, mask, image, K, color, return_coords=False):
    """
    Project point cloud onto image, overwrite projected
    points with the provided colour.
    """
    height, width, ch = image.shape
    viz_image = image.copy()

    pcd = pcd[mask]

    projected_points = K @ pcd.T
    projected_points = projected_points[:2, :] / projected_points[2, :]
    projected_image_coords = projected_points.T.round().astype(int)

    coords = np.clip(projected_image_coords, 0, [width - 1, height - 1])
    for point in coords:
        cv2.circle(viz_image, point, color=color, thickness=-1, radius=1)

    if return_coords:
        return coords, viz_image
    else:
        return viz_image

def generate_heatmap_from_points(points_2d, img_shape):
    """
    Generate a 3-channel heatmap from projected 2D points.

    Creates a distance-based heatmap where each channel represents the distance
    from every pixel to one of the projected gripper points.

    Args:
        points_2d (np.ndarray): Nx2 array of 2D pixel coordinates
        img_shape (tuple): (height, width) of output image

    Returns:
        np.ndarray: HxWx3 heatmap image with uint8 values [0-255]
    """
    height, width = img_shape[:2]
    max_distance = np.sqrt(width**2 + height**2)

    # Clip points to image bounds
    clipped_points = np.clip(points_2d, [0, 0], [width - 1, height - 1]).astype(int)

    goal_image = np.zeros((height, width, 3))
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    pixel_coords = np.stack([x_coords, y_coords], axis=-1)

    # Use first 3 points for the 3 channels
    for i in range(3):
        target_point = clipped_points[i]  # (2,)
        distances = np.linalg.norm(
            pixel_coords - target_point, axis=-1
        )  # (height, width)
        goal_image[:, :, i] = distances

    # Apply square root transformation for steeper near-target gradients
    goal_image = np.sqrt(goal_image / max_distance) * 255
    goal_image = np.clip(goal_image, 0, 255).astype(np.uint8)
    return goal_image

def get_img_pcd(image, depth, K, max_depth, num_points):
    height, width = depth.shape
    # Create pixel coordinate grid
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Flatten grid coordinates and depth
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = depth.flatten()

    # Remove points with invalid depth
    valid_depth = np.logical_and(z_flat > 0, z_flat < max_depth)
    x_flat = x_flat[valid_depth]
    y_flat = y_flat[valid_depth]
    z_flat = z_flat[valid_depth]

    # Create homogeneous pixel coordinates
    pixels = np.stack([x_flat, y_flat, np.ones_like(x_flat)], axis=0)

    # Unproject points using K inverse
    K_inv = np.linalg.inv(K)
    points = K_inv @ pixels
    points = points * z_flat
    points = points.T  # Shape: (N, 3)

    # Get corresponding colors
    colors = image.reshape(-1, 3)[valid_depth]
    image_pcd = np.concatenate([points, colors], axis=-1)

    image_pcd_pt3d = torch.from_numpy(image_pcd[None])
    image_pcd_downsample, image_points_idx = sample_farthest_points(
        image_pcd_pt3d, K=num_points, random_start_point=False
    )
    image_pcd = image_pcd_downsample.squeeze().numpy()

    image_pcd_pts = image_pcd[:, :3]
    image_pcd_colors = image_pcd[:, 3:]
    return image_pcd_pts, image_pcd_colors


def get_img_and_track_pcd(
    image,
    depth,
    K,
    mask,
    init_pcd,
    gt_pcd,
    all_pred_pcd,
    init_pcd_color,
    gt_color,
    all_pred_color,
    max_depth,
    num_points,
):
    init_pcd_color, all_pred_color, gt_color = (
        np.array(init_pcd_color),
        np.array(all_pred_color),
        np.array(gt_color),
    )

    image_pcd_pts, image_pcd_colors = get_img_pcd(
        image, depth, K, max_depth, num_points
    )

    init_pcd_pts = init_pcd[mask]
    init_pcd_color = np.repeat(init_pcd_color[None, :], init_pcd_pts.shape[0], axis=0)

    gt_pcd_pts = gt_pcd[mask]
    gt_color = np.repeat(gt_color[None, :], gt_pcd_pts.shape[0], axis=0)

    all_pred_pcd_pts, all_pred_colors = [], []
    for i, pred_pcd in enumerate(all_pred_pcd):
        pred_pcd_pts = pred_pcd[mask]
        pred_color = np.repeat(
            all_pred_color[None, i], all_pred_pcd[0][mask].shape[0], axis=0
        )
        all_pred_pcd_pts.append(pred_pcd_pts)
        all_pred_colors.append(pred_color)

    num_pred_points = all_pred_pcd_pts[0].shape[0] * len(all_pred_pcd_pts)
    viz_pcd_pts = np.concatenate(
        [image_pcd_pts, init_pcd_pts, gt_pcd_pts, *all_pred_pcd_pts], axis=0
    )
    viz_pcd_colors = np.concatenate(
        [image_pcd_colors, init_pcd_color, gt_color, *all_pred_colors], axis=0
    )

    # Flip about x-axis to prevent mirroring of pcd in wandb
    flip_transform = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    homogeneous_pts = np.hstack([viz_pcd_pts, np.ones((viz_pcd_pts.shape[0], 1))])
    transformed_pts = homogeneous_pts @ flip_transform.T
    transformed_pts_3d = transformed_pts[:, :3] / transformed_pts[:, 3:]

    viz_pcd = np.concatenate([transformed_pts_3d, viz_pcd_colors], axis=-1)
    return viz_pcd, num_pred_points


def get_action_anchor_pcd(
    action_pcd,
    anchor_pcd,
    action_pcd_color,
    anchor_pcd_color,
):
    action_pcd_color, anchor_pcd_color = (
        np.array(action_pcd_color),
        np.array(anchor_pcd_color),
    )

    action_pcd_color = np.repeat(action_pcd_color[None, :], action_pcd.shape[0], axis=0)
    anchor_pcd_color = np.repeat(anchor_pcd_color[None, :], anchor_pcd.shape[0], axis=0)

    viz_pcd_pts = np.concatenate([action_pcd, anchor_pcd], axis=0)
    viz_pcd_colors = np.concatenate([action_pcd_color, anchor_pcd_color], axis=0)

    # Flip about x-axis to prevent mirroring of pcd in wandb
    flip_transform = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    homogeneous_pts = np.hstack([viz_pcd_pts, np.ones((viz_pcd_pts.shape[0], 1))])
    transformed_pts = homogeneous_pts @ flip_transform.T
    transformed_pts_3d = transformed_pts[:, :3] / transformed_pts[:, 3:]

    viz_pcd = np.concatenate([transformed_pts_3d, viz_pcd_colors], axis=-1)
    return viz_pcd


def save_weighted_displacement_pcd_viz(pcd, weighted_displacement):
    """
    Save a glTF binary (.glb) file visualizing a colored point cloud along with sampled 3D vector lines
    derived from a weighted displacement prediction.

    While wandb apparently supports .glb files, the UI shows a blank screen

    Parameters
    ----------
    pcd : numpy.ndarray
        Array of shape (N, 3) representing the 3D point cloud.
    weighted_displacement : numpy.ndarray
        Array of shape (N, 13), where the last column is contains weights
        and the preceding 12 values represent 4 vectors per point (reshaped as (-1, 4, 3)),

    Returns
    -------
    str
        Filename of the exported glTF binary file (.glb).
    """
    N = pcd.shape[0]
    sample_lines = 100

    def softmax(x, T=10.0):
        exp_x = np.exp(x / T)
        return exp_x / np.sum(exp_x)

    weights = weighted_displacement[:, -1]
    weights_ = softmax(weights)
    cmap = cm.get_cmap("viridis")
    colors = (cmap(weights_)[:, :3] * 255).astype(np.uint8)

    point_cloud = trimesh.points.PointCloud(pcd, colors=colors)

    # Only consider the first point to avoid clutter in viz
    vectors = weighted_displacement[:, :-1].reshape(-1, 4, 3)[:, 0]
    # Create line segments for vectors.
    # Each line segment goes from a point to the point plus its vector.

    sample_idx = np.random.choice(N, sample_lines, replace=False)
    sampled_pcd = pcd[sample_idx]
    sampled_vectors = vectors[sample_idx]
    sampled_endpoints = sampled_pcd - sampled_vectors
    # The vertices for the paths include all starting points followed by endpoints.
    line_vertices = np.vstack([sampled_pcd, sampled_endpoints])

    # Build one line (entity) per point.
    entities = [
        trimesh.path.entities.Line(np.array([i, i + sample_lines]))
        for i in range(sample_lines)
    ]
    line_path = trimesh.path.Path3D(entities=entities, vertices=line_vertices)

    scene = trimesh.Scene()
    scene.add_geometry(point_cloud, node_name="point_cloud")
    scene.add_geometry(line_path, node_name="vectors")

    fname = f"{datetime.now().timestamp()}.glb"
    scene.export(fname)
    return fname


def invert_augmentation_and_normalization(pcd, pcd_mean, pcd_std, R, t, C):
    pcd = (pcd * pcd_std) + pcd_mean
    pcd = pcd - t  # Subtract translation
    pcd = (
        np.dot(pcd - C, R.T) + C
    )  # Inverse rotation (subtract centroid, rotate, add back)
    return pcd

def save_video(
    save_path: str, frames: Union[np.ndarray, List], max_size: int = None, fps: int = 30
):
    if not isinstance(frames, np.ndarray):
        frames = np.stack(frames)
    n, h, w = frames.shape[:-1]
    if max_size is not None:
        scale = max_size / max(w, h)
        h = int(h * scale // 16) * 16
        w = int(w * scale // 16) * 16
        frames = (cv2.resize(frames, output_shape=(n, h, w)) * 255).astype(np.uint8)
    print(f"saving video of size {(n, h, w)} to {save_path}")
    iio.imwrite(save_path, frames, fps=fps, extension=".webm", codec="vp9")

def get_heatmap_viz(rgb_image, heatmap, alpha=0.4):
    """
    Overlay heatmap on RGB image with transparency.

    Args:
        rgb_image: (H, W, 3) RGB image
        heatmap: (1, H, W) predicted heatmap
        alpha: transparency factor for heatmap overlay

    Returns:
       RGB image with heatmap overlay
    """
    # Get single heatmap
    if torch.is_tensor(heatmap):
        heatmap = heatmap.squeeze().cpu().numpy() # (H, W)

    # Normalize heatmap to 0-1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Convert to colormap (jet colormap)
    colormap = cm.jet(heatmap)[:, :, :3]  # (H, W, 3)
    colormap = (colormap * 255).astype(np.uint8)

    # Blend with original image
    overlay = cv2.addWeighted(rgb_image, 1 - alpha, colormap, alpha, 0)

    return overlay