import cv2
import numpy as np


def project_pcd_on_image(pcd, image, K, color):
    """
    Project point cloud onto image, overwrite projected
    points with the provided colour.
    """
    height, width, ch = image.shape
    viz_image = image.copy()

    mask = ~np.all(pcd == 0, axis=1)  # Remove 0 points
    pcd = pcd[mask]

    projected_points = K @ pcd.T
    projected_points = projected_points[:2, :] / projected_points[2, :]
    projected_image_coords = projected_points.T.round().astype(int)

    coords = np.clip(projected_image_coords, 0, [width - 1, height - 1])
    for point in coords:
        cv2.circle(viz_image, point, color=color, thickness=-1, radius=5)

    return viz_image


def get_img_and_track_pcd(image, depth, K, pred_pcd, gt_pcd, pred_color, gt_color):
    height, width = depth.shape
    pred_color, gt_color = np.array(pred_color), np.array(gt_color)

    # Create pixel coordinate grid
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Flatten grid coordinates and depth
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = depth.flatten()

    # Remove points with invalid depth
    valid_depth = z_flat > 0
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
    # Keep one in 100 points
    image_pcd = image_pcd[::100]

    mask = ~np.all(gt_pcd == 0, axis=1)  # Remove 0 points
    gt_pcd = gt_pcd[mask]
    gt_color = np.repeat(gt_color[None, :], gt_pcd.shape[0], axis=0)
    gt_pcd = np.concatenate([gt_pcd, gt_color], axis=-1)

    pred_pcd = pred_pcd[mask]
    pred_color = np.repeat(pred_color[None, :], pred_pcd.shape[0], axis=0)
    pred_pcd = np.concatenate([pred_pcd, pred_color], axis=-1)

    viz_pcd = np.concatenate([image_pcd, pred_pcd, gt_pcd], axis=0)

    return viz_pcd
