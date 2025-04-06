import cv2
import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points


def project_pcd_on_image(pcd, mask, image, K, color):
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

    return viz_image


def get_img_pcd(image, depth, K):
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
    valid_depth = np.logical_and(z_flat > 0, z_flat < 2)
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
        image_pcd_pt3d, K=4096, random_start_point=False
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
    pred_pcd,
    init_pcd_color,
    gt_color,
    pred_color,
):
    init_pcd_color, pred_color, gt_color = (
        np.array(init_pcd_color),
        np.array(pred_color),
        np.array(gt_color),
    )

    image_pcd_pts, image_pcd_colors = get_img_pcd(image, depth, K)

    init_pcd_pts = init_pcd[mask]
    init_pcd_color = np.repeat(init_pcd_color[None, :], init_pcd_pts.shape[0], axis=0)

    gt_pcd_pts = gt_pcd[mask]
    gt_color = np.repeat(gt_color[None, :], gt_pcd_pts.shape[0], axis=0)

    pred_pcd_pts = pred_pcd[mask]
    pred_color = np.repeat(pred_color[None, :], pred_pcd_pts.shape[0], axis=0)

    viz_pcd_pts = np.concatenate(
        [image_pcd_pts, init_pcd_pts, pred_pcd_pts, gt_pcd_pts], axis=0
    )
    viz_pcd_colors = np.concatenate(
        [image_pcd_colors, init_pcd_color, pred_color, gt_color], axis=0
    )

    # Flip about x-axis to prevent mirroring of pcd in wandb
    flip_transform = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    homogeneous_pts = np.hstack([viz_pcd_pts, np.ones((viz_pcd_pts.shape[0], 1))])
    transformed_pts = homogeneous_pts @ flip_transform.T
    transformed_pts_3d = transformed_pts[:, :3] / transformed_pts[:, 3:]

    viz_pcd = np.concatenate([transformed_pts_3d, viz_pcd_colors], axis=-1)
    return viz_pcd


def get_img_and_wta_tracks_pcd(
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
):
    init_pcd_color, all_pred_color, gt_color = (
        np.array(init_pcd_color),
        np.array(all_pred_color),
        np.array(gt_color),
    )

    image_pcd_pts, image_pcd_colors = get_img_pcd(image, depth, K)

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
    eps = 1e-7

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
