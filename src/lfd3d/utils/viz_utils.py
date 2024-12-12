import cv2
import numpy as np
import open3d as o3d


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
        cv2.circle(viz_image, point, color=color, thickness=-1, radius=5)

    return viz_image


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
    height, width = depth.shape
    init_pcd_color, pred_color, gt_color = (
        np.array(init_pcd_color),
        np.array(pred_color),
        np.array(gt_color),
    )

    # Create pixel coordinate grid
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Flatten grid coordinates and depth
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = depth.flatten()

    # Remove points with invalid depth
    valid_depth = np.logical_and(z_flat > 0, z_flat < 5)
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

    init_pcd = init_pcd[mask]
    init_pcd_color = np.repeat(init_pcd_color[None, :], init_pcd.shape[0], axis=0)
    init_pcd = np.concatenate([init_pcd, init_pcd_color], axis=-1)

    gt_pcd = gt_pcd[mask]
    gt_color = np.repeat(gt_color[None, :], gt_pcd.shape[0], axis=0)
    gt_pcd = np.concatenate([gt_pcd, gt_color], axis=-1)

    pred_pcd = pred_pcd[mask]
    pred_color = np.repeat(pred_color[None, :], pred_pcd.shape[0], axis=0)
    pred_pcd = np.concatenate([pred_pcd, pred_color], axis=-1)

    viz_pcd = np.concatenate([image_pcd, init_pcd, pred_pcd, gt_pcd], axis=0)

    return viz_pcd


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
    action_pcd = np.concatenate([action_pcd, action_pcd_color], axis=-1)

    anchor_pcd_color = np.repeat(anchor_pcd_color[None, :], anchor_pcd.shape[0], axis=0)
    anchor_pcd = np.concatenate([anchor_pcd, anchor_pcd_color], axis=-1)

    viz_pcd = np.concatenate([action_pcd, anchor_pcd], axis=0)
    return viz_pcd


def create_point_cloud_frames(points_colors, n_frames=30, width=640, height=480):
    """
    Create frames of a rotating point cloud using headless rendering.

    Args:
        points_colors: Nx6 numpy array where first 3 columns are XYZ coordinates
                      and last 3 columns are RGB values (0-255)
        n_frames: Number of frames to generate
        width: Width of output frames
        height: Height of output frames

    Returns:
        numpy array of shape (n_frames, height, width, 3) with uint8 RGB values
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_colors[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points_colors[:, 3:] / 255.0)

    # Pre-allocate frames array
    frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)

    # Create offscreen renderer
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Create material with larger point size
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    mat.shader = "defaultUnlit"
    mat.point_size = 5.0  # Increased point size

    # Set up scene
    render.scene.add_geometry("points", pcd, mat)
    render.scene.set_background([0, 0, 0, 1])  # Set to black for contrast

    # Set up camera positioning and orientation
    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = np.linalg.norm(bounds.get_max_bound() - bounds.get_min_bound())

    camera_distance = extent * 0.75
    up = [0, -1, 0]

    # Generate frames
    angles = np.linspace(0, 2 * np.pi, n_frames)
    for i, angle in enumerate(angles):
        eye = center + camera_distance * np.array([np.sin(angle), 0, np.cos(angle)])

        # Update camera to look at the center of the point cloud
        render.scene.camera.look_at(center, eye, up)

        # Render frame
        img = render.render_to_image()
        frames[i] = np.asarray(img)

    return frames
