import os
import random

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def compute_lookat_quaternion(camera_pos, up_vector=None):
    # Camera position
    x, y, z = camera_pos

    # Direction vector (camera -> origin)
    direction = np.array([-x, -y, -z])
    direction = direction / np.linalg.norm(direction)  # Normalize

    up_vector = np.array([0, 0, 1])

    # Compute right vector (cross product of direction and up)
    right = np.cross(direction, up_vector)
    right = right / np.linalg.norm(right)

    # Recompute consistent up vector
    consistent_up = np.cross(right, direction)
    consistent_up = consistent_up / np.linalg.norm(consistent_up)

    # Create rotation matrix
    rotation_matrix = np.column_stack((right, consistent_up, -direction))
    rotation = R.from_matrix(rotation_matrix)

    # Convert to quaternion
    quaternion = rotation.as_quat()  # [x, y, z, w]
    quaternion = quaternion[[3, 0, 1, 2]]

    return quaternion


def compute_intrinsic_matrix(fovy, width, height):
    # Convert vertical field of view from degrees to radians
    fovy_rad = np.deg2rad(fovy)

    # Compute focal length
    # Focal length is half the image height divided by tan(half fovy)
    focal_length_y = height / (2 * np.tan(fovy_rad / 2))

    # Assume symmetric perspective, so focal length x is same as y
    focal_length_x = focal_length_y

    # Optical center is typically the image center
    cx = width / 2
    cy = height / 2

    # Construct the K matrix
    K = np.array([[focal_length_x, 0, cx], [0, focal_length_y, cy], [0, 0, 1]])

    return K


def save_dataset_item(segmasks, depths, frames, init_pcd, final_pcd, itr):
    os.makedirs(f"synth_block_data/{itr}", exist_ok=True)
    init_block_mask = segmasks[0][:, :, 0] == 1
    final_block_mask = segmasks[-1][:, :, 0] == 1
    depth_init_block = np.where(init_block_mask, depths[0], 0)
    depth_final_block = np.where(final_block_mask, depths[-1], 0)

    depth_init_block_o3d = o3d.geometry.Image(depth_init_block)
    init_block_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_init_block_o3d, intrinsic, depth_scale=1, depth_trunc=20
    )
    init_block_pcd_pts = np.asarray(init_block_pcd.points)
    depth_final_block_o3d = o3d.geometry.Image(depth_final_block)
    final_block_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_final_block_o3d, intrinsic, depth_scale=1, depth_trunc=20
    )
    final_block_pcd_pts = np.asarray(final_block_pcd.points)

    cross_displacement = final_block_pcd_pts.mean(0) - init_block_pcd_pts.mean(0)
    cross_displacement = np.tile(
        cross_displacement[None], (init_block_pcd_pts.shape[0], 1)
    )

    init_pcd_points = np.asarray(init_pcd.points)
    final_pcd_points = np.asarray(final_pcd.points)

    plt.imsave(f"synth_block_data/{itr}/rgb_0.png", frames[0])
    plt.imsave(f"synth_block_data/{itr}/rgb_1.png", frames[-1])
    np.save(f"synth_block_data/{itr}/depth_0.npy", depths[0])
    np.save(f"synth_block_data/{itr}/depth_1.npy", depths[-1])
    plt.imsave(f"synth_block_data/{itr}/segmasks_0.png", segmasks[0][:, :, 0])
    plt.imsave(f"synth_block_data/{itr}/segmasks_1.png", segmasks[-1][:, :, 0])
    np.save(f"synth_block_data/{itr}/pcd_0.npy", init_pcd_points)
    np.save(f"synth_block_data/{itr}/pcd_1.npy", final_pcd_points)
    np.save(f"synth_block_data/{itr}/action_pcd.npy", init_block_pcd_pts)
    np.save(f"synth_block_data/{itr}/cross_displacement.npy", cross_displacement)


def visualize_dataset_item(segmasks, depths, frames, init_pcd, final_pcd, itr):
    os.makedirs(f"viz/{itr}/pcd", exist_ok=True)
    o3d.io.write_point_cloud(f"viz/{itr}/pcd/init.ply", init_pcd)
    o3d.io.write_point_cloud(f"viz/{itr}/pcd/final.ply", final_pcd)
    for i in range(len(frames)):
        os.makedirs(f"viz/{itr}/rgb", exist_ok=True)
        plt.imsave(f"viz/{itr}/rgb/{str(i).zfill(5)}.png", frames[i])

        os.makedirs(f"viz/{itr}/depth", exist_ok=True)
        plt.imsave(f"viz/{itr}/depth/{str(i).zfill(5)}.png", depths[i])

        os.makedirs(f"viz/{itr}/segmasks", exist_ok=True)
        plt.imsave(f"viz/{itr}/segmasks/{str(i).zfill(5)}.png", segmasks[i][:, :, 0])


# Load MJCF model
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

fps = 30.0
duration = 2
n_iters = 1000

width, height = 320, 240
fovy = model.cam("overview").fovy[0]
K = compute_intrinsic_matrix(fovy, width, height)

os.makedirs("synth_block_data", exist_ok=True)
np.save("synth_block_data/intrinsics.npy", K)
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(
    width=width, height=height, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
)

for itr in tqdm(range(n_iters)):
    frames = []
    depths = []
    segmasks = []

    # Set camera pos
    theta = random.uniform(0, 2 * np.pi)
    x, y, z = 10 * np.cos(theta), 10 * np.sin(theta), random.uniform(8, 10)
    cam_pos = np.array([x, y, z])
    cam_quat = compute_lookat_quaternion(cam_pos)

    # Set block size
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    random_size = np.random.uniform(0.2, 0.8, size=3)
    model.geom_size[geom_id] = random_size

    # Set block pos
    block_pos = np.array(
        [random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(1, 4)]
    )

    mujoco.mj_resetData(model, data)
    model.cam("overview").pos = cam_pos
    model.cam("overview").quat = cam_quat

    # Set block pos
    block_body_id = model.body("cube").id - 1
    data.qpos[block_body_id * 7 : block_body_id * 7 + 3] = block_pos

    while data.time <= duration:
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera="overview")

        if len(frames) < data.time * fps:
            frames.append(renderer.render())

            renderer.enable_depth_rendering()
            depth = renderer.render()
            depth[depth > 50] = 0
            depths.append(depth)
            renderer.disable_depth_rendering()

            renderer.enable_segmentation_rendering()
            segmask = renderer.render()
            segmasks.append(segmask)
            renderer.disable_segmentation_rendering()

    depth_o3d = o3d.geometry.Image(depths[0])
    rgb_o3d = o3d.geometry.Image(frames[0])
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1, depth_trunc=20
    )
    init_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    depth_o3d = o3d.geometry.Image(depths[-1])
    rgb_o3d = o3d.geometry.Image(frames[-1])
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1, depth_trunc=20
    )
    final_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    save_dataset_item(segmasks, depths, frames, init_pcd, final_pcd, itr)

    visualize = itr % 100 == 0
    if visualize:
        visualize_dataset_item(segmasks, depths, frames, init_pcd, final_pcd, itr)

renderer.close()
