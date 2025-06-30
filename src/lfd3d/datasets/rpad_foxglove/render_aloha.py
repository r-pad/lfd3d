import argparse
import os

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import open3d as o3d
import trimesh
import zarr
from lfd3d.utils.data_utils import combine_meshes
from robot_descriptions.loaders.mujoco import load_robot_description
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def align_timestamps(rgb_ts, other_ts):
    """
    Aligns timestamps between two arrays by finding closest matches.

    Args:
        rgb_ts: Array of RGB timestamps
        other_ts: Array of other timestamps

    Returns:
        Tuple of (aligned_rgb_indices, other_indices, time_differences)
    """
    aligned_pairs = []
    rgb_indices = []
    other_indices = []
    time_diffs = []

    # For each RGB timestamp, find the closest other timestamp
    for i, rgb_t in enumerate(rgb_ts):
        # Calculate absolute differences
        diffs = np.abs(other_ts - rgb_t)
        # Find index of minimum difference
        closest_idx = np.argmin(diffs)
        # Get the minimum time difference
        min_diff = diffs[closest_idx]

        rgb_indices.append(i)
        other_indices.append(closest_idx)
        time_diffs.append(min_diff)

    return np.array(rgb_indices), np.array(other_indices), np.array(time_diffs)


def setup_camera(model, cam_id, cam_to_world, width, height, K):
    world_to_cam = np.linalg.inv(cam_to_world)
    model.cam_pos[cam_id] = cam_to_world[:3, 3]
    R_flip = np.diag([1, -1, -1])
    R_cam = Rotation.from_matrix(cam_to_world[:3, :3] @ R_flip)
    cam_quat = R_cam.as_quat()  # [x, y, z, w]
    cam_quat = cam_quat[[3, 0, 1, 2]]  # Reorder to [w, x, y, z] for MuJoCo
    model.cam_quat[cam_id] = cam_quat
    fovy = np.degrees(2 * np.arctan((height / 2) / K[1, 1]))
    model.cam_fovy[cam_id] = fovy


def render_rightArm_images(renderer, data, camera="teleoperator_pov"):
    """
    Render RGB, depth, and segmentation images with right arm masking from MuJoCo simulation.
    Args:
        renderer (mujoco.Renderer): MuJoCo renderer instance configured for the scene
        data (mujoco.MjData): MuJoCo data object containing current simulation state
        camera (str, optional): Name of the camera to render from.

    Returns:
        tuple: A tuple containing:
            - rgb (np.ndarray): Masked RGB image of shape (H, W, 3), dtype uint8.
                               Right arm pixels retain original colors, background pixels are black.
            - depth (np.ndarray): Masked depth image of shape (H, W), dtype float32.
                                 Right arm pixels contain depth values, background pixels are zero.
            - seg (np.ndarray): Binary segmentation mask of shape (H, W), dtype bool.
                               True for right arm pixels, False for background.
    """
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render()

    # Depth rendering
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()

    # Segmentation rendering
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()

    seg = seg[:, :, 0]  # channel 1 is foreground/background
    # NOTE: Classes for the right arm excluding the camera mount. Handpicked
    target_classes = set(range(65, 91)) - {81, 82, 83}

    seg = np.isin(seg, list(target_classes)).astype(bool)
    rgb[~seg] = 0
    depth[~seg] = 0

    return rgb, depth, seg


def get_right_gripper_mesh(mj_model, mj_data):
    """
    Extract the visual meshes of the right gripper from the Aloha MJCF model.

    Args:
        mj_model: MuJoCo model object.
        mj_data: MuJoCo data object containing the current simulation state.

    Returns:
        List of open3d.geometry.TriangleMesh objects representing the right gripper's visual meshes
        in world coordinates.
    """
    meshes = []

    # Define the bodies that make up the right gripper
    right_gripper_body_names = [
        "right/gripper_base",
        "right/left_finger_link",
        "right/right_finger_link",
    ]
    exclude_mesh_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MESH, "d405_solid"),
        mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_MESH, "vx300s_7_gripper_wrist_mount"
        ),
    ]

    # Get body IDs for the right gripper components
    right_gripper_body_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in right_gripper_body_names
    ]

    # Iterate over all geoms in the model
    for geom_id in range(mj_model.ngeom):
        # Check if the geom belongs to the right gripper, is visual (group 2), and is a mesh, and is not camera
        if (
            mj_model.geom_bodyid[geom_id] in right_gripper_body_ids
            and mj_model.geom_group[geom_id] == 2
            and mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
            and mj_model.geom_dataid[geom_id] not in exclude_mesh_ids
        ):
            geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

            # Get the geom's world position and orientation from the simulation state
            geom_pos = mj_data.geom_xpos[geom_id]  # 3D position in world coordinates
            geom_mat = mj_data.geom_xmat[geom_id].reshape(3, 3)  # 3x3 rotation matrix

            # Get the mesh ID associated with this geom
            mesh_id = mj_model.geom_dataid[geom_id]
            if mesh_id >= 0:  # Ensure the geom has a valid mesh
                # Extract mesh vertex and face data
                mesh_vert_adr = mj_model.mesh_vertadr[
                    mesh_id
                ]  # Start index of vertices
                mesh_vert_num = mj_model.mesh_vertnum[mesh_id]  # Number of vertices
                mesh_face_adr = mj_model.mesh_faceadr[mesh_id]  # Start index of faces
                mesh_face_num = mj_model.mesh_facenum[mesh_id]  # Number of faces

                vertices_local = mj_model.mesh_vert[
                    mesh_vert_adr : mesh_vert_adr + mesh_vert_num
                ].copy()
                faces = mj_model.mesh_face[
                    mesh_face_adr : mesh_face_adr + mesh_face_num
                ].copy()

                # Transform local vertices to world coordinates
                vertices_world = vertices_local @ geom_mat.T + geom_pos

                # Create an Open3D mesh
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                meshes.append(mesh)

    return meshes


def process_demo(
    demo,
    model,
    data,
    renderer,
    width,
    height,
    K,
    world_to_cam,
    sample_n_points,
    visualize,
    in_mm=False,
):
    joint_positions = demo["raw"]["follower_right"]["joint_states"]["pos"][:]
    joint_positions_ts = demo["raw"]["follower_right"]["joint_states"]["ts"][:]
    rgb_imgs = demo["raw"]["rgb"]["image_rect"]["img"][:]
    rgb_ts = demo["raw"]["rgb"]["image_rect"]["ts"][:]
    depth_imgs = demo["raw"]["depth_registered"]["image_rect"]["img"][:]
    depth_ts = demo["raw"]["depth_registered"]["image_rect"]["ts"][:]

    rgb_idx, joint_idx, _ = align_timestamps(rgb_ts, joint_positions_ts)
    rgb_idx, depth_idx, _ = align_timestamps(rgb_ts, depth_ts)
    rgb_imgs = rgb_imgs[rgb_idx]
    rgb_imgs = np.array([cv2.resize(i, (0, 0), fx=0.25, fy=0.25) for i in rgb_imgs])
    depth_imgs = depth_imgs[depth_idx]
    depth_imgs = np.array([cv2.resize(i, (0, 0), fx=0.25, fy=0.25) for i in depth_imgs])
    joint_positions = joint_positions[joint_idx]

    POINTS = []
    masks = []
    for t in tqdm(range(joint_positions.shape[0])):
        data.qpos[0] = -np.pi  # move the left arm out of the way
        data.qpos[8 : 8 + 6] = joint_positions[t, :6]
        data.qpos[8 + 6 : 8 + 8] = joint_positions[t, [7, 7]]
        mujoco.mj_forward(model, data)

        meshes = get_right_gripper_mesh(model, data)
        mesh = combine_meshes(meshes)
        mesh_ = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
        )
        points_, _ = trimesh.sample.sample_surface(mesh_, sample_n_points, seed=42)

        gripper_urdf_3d_pos = np.concatenate(
            [points_, np.ones((sample_n_points, 1))], axis=-1
        )[:, :, None]
        urdf_cam3dcoords = (world_to_cam @ gripper_urdf_3d_pos)[:, :3].squeeze(2)

        POINTS.append(urdf_cam3dcoords)

        _, _, seg = render_rightArm_images(renderer, data)
        masks.append(seg)

        if visualize:
            os.makedirs(f"mujoco_renders/{demo.name}", exist_ok=True)

            urdf_proj_hom = (K @ urdf_cam3dcoords.T).T
            urdf_proj = (urdf_proj_hom / urdf_proj_hom[:, 2:])[:, :2]
            urdf_proj = np.clip(urdf_proj, [0, 0], [width - 1, height - 1]).astype(int)

            rgb = rgb_imgs[t]
            rgb[urdf_proj[:, 1], urdf_proj[:, 0]] = [0, 255, 0]  # Green
            plt.imsave(f"mujoco_renders/{demo.name}/{str(t).zfill(5)}.png", rgb)

    POINTS = np.array(POINTS)
    if in_mm:
        POINTS *= 1000  # Convert from meters to millimeters

    # Delete existing gripper_pos dataset if it exists
    if "gripper_pos" in demo:
        del demo["gripper_pos"]
    demo.create_dataset("gripper_pos", data=POINTS)

    if "masks" in demo:
        del demo["masks"]
    demo.create_dataset("masks", data=masks)


def main(args):
    dataset = zarr.group(args.root)
    model = load_robot_description("aloha_mj_description")
    data = mujoco.MjData(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "teleoperator_pov")

    rgb_cam_info = dataset[list(dataset.keys())[0]]["raw"]["rgb"]["camera_info"]
    width = rgb_cam_info["width"][0] // 4
    height = rgb_cam_info["height"][1] // 4
    K = rgb_cam_info["k"][0]
    K[0, 0] /= 4
    K[0, 2] /= 4
    K[1, 1] /= 4
    K[1, 2] /= 4

    cam_to_world = np.loadtxt(args.calib_file)
    setup_camera(model, cam_id, cam_to_world, width, height, K)
    renderer = mujoco.Renderer(model, width=width, height=height)

    for demo_name in tqdm(dataset):
        demo = dataset[demo_name]
        if "follower_right" not in demo["raw"]:
            print("Human demo. Skipping.")
            continue

        process_demo(
            demo,
            model,
            data,
            renderer,
            width,
            height,
            K,
            np.linalg.inv(cam_to_world),
            args.sample_n_points,
            args.visualize,
            args.in_mm,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Aloha robot data.")
    parser.add_argument(
        "--root", type=str, required=True, help="Root directory of the zarr dataset"
    )
    parser.add_argument(
        "--sample_n_points",
        type=int,
        default=500,
        help="Number of points to sample from the gripper mesh",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Flag to enable visualization",
        default=False,
    )
    parser.add_argument(
        "--in_mm",
        action="store_true",
        help="Save point coordinates in millimeters instead of meters",
        default=False,
    )
    parser.add_argument(
        "--calib_file",
        help="Path to T_world_from_camera_est.txt",
        required=True,
    )
    args = parser.parse_args()

    main(args)
