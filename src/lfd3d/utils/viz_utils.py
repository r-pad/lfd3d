from datetime import datetime
from typing import List, Union

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from matplotlib import cm
from pytorch3d.ops import sample_farthest_points


def interpolate_colors(color_start, color_end, n_points):
    """
    Interpolate between two colors.

    Args:
        color_start: (R, G, B) starting color
        color_end: (R, G, B) ending color
        n_points: Number of interpolation points

    Returns:
        Array of shape (n_points, 3) with interpolated colors
    """
    if n_points == 1:
        return np.array([color_start])

    color_start = np.array(color_start)
    color_end = np.array(color_end)

    # Linear interpolation
    alphas = np.linspace(0, 1, n_points)[:, None]  # (n_points, 1)
    colors = (1 - alphas) * color_start + alphas * color_end

    return colors.astype(int)


def project_pcd_on_image(pcd, mask, image, K, color, return_coords=False, radius=1):
    """
    Project point cloud onto image, overwrite projected
    points with the provided colour.

    Args:
        pcd: Point cloud array (N, 3)
        mask: Boolean mask for points to project
        image: RGB image (H, W, 3)
        K: Camera intrinsics (3, 3)
        color: Either a single color tuple (R, G, B) or a sequence of colors [(R, G, B), ...]
               If sequence, length must match number of masked points
        return_coords: Whether to return projected coordinates
        radius: Circle radius for visualization

    Returns:
        viz_image or (coords, viz_image) if return_coords=True
    """
    height, width, ch = image.shape
    viz_image = image.copy()

    pcd = pcd[mask]

    projected_points = K @ pcd.T
    projected_points = projected_points[:2, :] / projected_points[2, :]
    projected_image_coords = projected_points.T.round().astype(int)

    coords = np.clip(projected_image_coords, 0, [width - 1, height - 1])

    # Check if color is a sequence of colors or a single color
    # Single color: tuple/list with 3 elements (R, G, B)
    # Sequence of colors: list/array where each element is a color
    is_single_color = (
        isinstance(color, (tuple, list))
        and len(color) == 3
        and all(isinstance(c, (int, np.integer)) for c in color)
    )

    if is_single_color:
        # Use same color for all points (backward compatible)
        for point in coords:
            cv2.circle(viz_image, point, color=color, thickness=-1, radius=radius)
    else:
        # Use different color for each point
        colors = np.array(color)
        if len(colors) != len(coords):
            raise ValueError(
                f"Number of colors ({len(colors)}) must match number of points ({len(coords)})"
            )
        for point, pt_color in zip(coords, colors):
            cv2.circle(
                viz_image,
                point,
                color=tuple(pt_color.tolist()),
                thickness=-1,
                radius=radius,
            )

    if return_coords:
        return coords, viz_image
    else:
        return viz_image


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
    """
    Create a combined point cloud visualization with image points and trajectory points.

    Args:
        image: RGB image
        depth: Depth map
        K: Camera intrinsics
        mask: Boolean mask for trajectory points
        init_pcd: Initial trajectory points
        gt_pcd: Ground truth trajectory points
        all_pred_pcd: List of predicted trajectories
        init_pcd_color: Single color (R,G,B) or sequence of colors (N_init, 3)
        gt_color: Single color (R,G,B) or sequence of colors (N_gt, 3)
        all_pred_color: Either:
            - Array (N_preds, 3): one color per prediction (backward compatible)
            - List of arrays [(N_points, 3), ...]: color sequence per prediction
        max_depth: Maximum depth for filtering
        num_points: Number of points to sample from image

    Returns:
        viz_pcd: Combined point cloud with colors
        num_pred_points: Number of prediction points
    """
    init_pcd_color = np.array(init_pcd_color)
    gt_color = np.array(gt_color)
    all_pred_color = (
        np.array(all_pred_color)
        if not isinstance(all_pred_color, list)
        else all_pred_color
    )

    image_pcd_pts, image_pcd_colors = get_img_pcd(
        image, depth, K, max_depth, num_points
    )

    # Process init_pcd colors
    init_pcd_pts = init_pcd[mask]
    if init_pcd_color.ndim == 1:
        # Single color: repeat for all points
        init_pcd_color = np.repeat(
            init_pcd_color[None, :], init_pcd_pts.shape[0], axis=0
        )
    else:
        # Color sequence: apply mask
        init_pcd_color = init_pcd_color[mask]

    # Process gt_pcd colors
    gt_pcd_pts = gt_pcd[mask]
    if gt_color.ndim == 1:
        # Single color: repeat for all points
        gt_color = np.repeat(gt_color[None, :], gt_pcd_pts.shape[0], axis=0)
    else:
        # Color sequence: apply mask
        gt_color = gt_color[mask]

    # Process all_pred_pcd colors
    all_pred_pcd_pts, all_pred_colors = [], []
    for i, pred_pcd in enumerate(all_pred_pcd):
        pred_pcd_pts = pred_pcd[mask]

        # Check if all_pred_color is list of color sequences or array of single colors
        if isinstance(all_pred_color, list):
            # List of color sequences
            pred_color = all_pred_color[i][mask]
        else:
            # Array of single colors (backward compatible)
            pred_color = np.repeat(
                all_pred_color[None, i], pred_pcd_pts.shape[0], axis=0
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


def get_heatmap_viz(rgb_image, heatmap, alpha=0.4):
    """
    Overlay heatmap on RGB image with transparency.

    Args:
        rgb_image: (H, W, 3) RGB image
        heatmap: (1, H, W) predicted heatmap
        alpha: transparency factor for heatmap overlay

    Returns:
        overlay: (np.ndarray) RGB image with heatmap overlay
    """
    # Get single heatmap
    if torch.is_tensor(heatmap):
        heatmap = heatmap.squeeze().cpu().numpy()  # (H, W)

    # Normalize heatmap to 0-1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Convert to colormap (jet colormap)
    colormap = cm.jet(heatmap)[:, :, :3]  # (H, W, 3)
    colormap = (colormap * 255).astype(np.uint8)

    # Blend with original image
    overlay = cv2.addWeighted(rgb_image, 1 - alpha, colormap, alpha, 0)

    return overlay


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


def plot_seq_data(data, title, xlabel, ylabel, path):
    """
    Plot a sequence of numeric data with its mean and variance statistics.

    The function creates a line plot of the input data sequence, adds a
    horizontal line representing the mean value, and displays the variance
    in a text box on the plot. The figure is then saved to the given file path.

    Args:
        data (list, np.ndarray): Input sequence of numeric values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        path (str): File path where the figure will be saved.
    """
    fig, ax = plt.subplots()
    data = np.array(data)
    ax.plot(data, label="Error", color="blue")

    avg = float(np.nanmean(data))
    std = float(np.nanstd(data))

    ax.axhline(
        avg, color="red", linestyle="--", linewidth=1.2, label=f"Mean = {avg:.4f}"
    )
    ax.text(
        0.02,
        0.95,
        f"Std = {std:.4f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_barchart_with_error(data, error, title, xlabel, ylabel, path):
    """
    Plot data as a bar with an error bar.
    """
    data = np.asarray(data)
    error = np.asarray(error)
    x = np.arange(len(data))

    fig, ax = plt.subplots()
    ax.bar(
        x,
        data,
        yerr=error,
        capsize=4,
        width=1.0,
        align="edge",
        color="skyblue",
        edgecolor="black",
    )

    ax.set_xticks(x + 0.5)
    ax.set_xticklabels([str(i) for i in x])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def annotate_video(frames, annotation=None, path=None, fps=30):
    """
    Args:
        frames: np.ndarray of shape (N, H, W, 3) dtype=uint8
        left_values: (N,) array of text/values to put on top-left
        right_values: (N,) array of text/values to put on top-right
        out_path: str, path to save the video
        fps: int, frames per second
    """
    H, W = frames.shape[1:3]
    font = cv2.FONT_HERSHEY_SIMPLEX

    annotated_frames = []
    pos = [
        (10, 30),
        (W - 300, 30),
        (10, H - 30),
        (W - 300, H - 30),
    ]
    for i, frame in enumerate(frames):
        img = frame.copy()

        if annotation is not None:
            for j, (key, value) in enumerate(annotation.items()):
                cv2.putText(
                    img,
                    f"{key}: {value[i]:.4f}",
                    pos[j],
                    font,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        annotated_frames.append(img)
    annotated_frames = np.stack(annotated_frames)
    if path:
        iio.imwrite(path, annotated_frames, fps=fps)
    else:
        return annotated_frames
