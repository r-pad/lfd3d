import argparse
import os

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import open3d as o3d
import trimesh
import zarr
from robot_descriptions.loaders.mujoco import load_robot_description
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from lfd3d.utils.data_utils import combine_meshes


def align_timestamps(rgb_ts, joint_positions_ts):
    """
    Aligns timestamps between two arrays by finding closest matches.

    Args:
        rgb_ts: Array of RGB timestamps
        joint_positions_ts: Array of joint position timestamps

    Returns:
        Tuple of (aligned_rgb_indices, aligned_joint_indices, time_differences)
    """
    aligned_pairs = []
    rgb_indices = []
    joint_indices = []
    time_diffs = []

    # For each RGB timestamp, find the closest joint position timestamp
    for i, rgb_t in enumerate(rgb_ts):
        # Calculate absolute differences
        diffs = np.abs(joint_positions_ts - rgb_t)
        # Find index of minimum difference
        closest_idx = np.argmin(diffs)
        # Get the minimum time difference
        min_diff = diffs[closest_idx]

        rgb_indices.append(i)
        joint_indices.append(closest_idx)
        time_diffs.append(min_diff)

    return np.array(rgb_indices), np.array(joint_indices), np.array(time_diffs)


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
):
    joint_positions = demo["_follower_right_joint_states"]["pos"][:]
    joint_positions_ts = demo["_follower_right_joint_states"]["ts"][:]
    rgb_imgs = demo["_rgb_image_rect"]["img"][:]
    rgb_ts = demo["_rgb_image_rect"]["ts"][:]

    rgb_idx, joint_idx, _ = align_timestamps(rgb_ts, joint_positions_ts)
    rgb_imgs = rgb_imgs[rgb_idx]
    rgb_imgs = np.array([cv2.resize(i, (0, 0), fx=0.25, fy=0.25) for i in rgb_imgs])
    joint_positions = joint_positions[joint_idx]

    POINTS = []
    for t in tqdm(range(joint_positions.shape[0])):
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

        if visualize:
            os.makedirs(f"mujoco_renders/{demo.name}", exist_ok=True)
            renderer.update_scene(data, camera="teleoperator_pov")
            rend = renderer.render()

            urdf_proj_hom = (K @ urdf_cam3dcoords.T).T
            urdf_proj = (urdf_proj_hom / urdf_proj_hom[:, 2:])[:, :2]
            urdf_proj = np.clip(urdf_proj, [0, 0], [width - 1, height - 1]).astype(int)
            rend[urdf_proj[:, 1], urdf_proj[:, 0]] = [0, 255, 0]  # Green

            rgb = rgb_imgs[t]
            rgb[urdf_proj[:, 1], urdf_proj[:, 0]] = [0, 255, 0]  # Green
            plt.imsave(f"mujoco_renders/{demo.name}/{str(t).zfill(5)}.png", rgb)

    POINTS = np.array(POINTS)
    demo.create_dataset("gripper_pos", data=POINTS)


def main(args):
    dataset = zarr.group(args.root)
    model = load_robot_description("aloha_mj_description")
    data = mujoco.MjData(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "teleoperator_pov")

    rgb_cam_info = dataset[list(dataset.keys())[0]]["_rgb_camera_info"]
    width = rgb_cam_info["width"][0] // 4
    height = rgb_cam_info["height"][1] // 4
    K = rgb_cam_info["k"][0]
    K[0, 0] /= 4
    K[0, 2] /= 4
    K[1, 1] /= 4
    K[1, 2] /= 4

    cam_to_world = np.loadtxt("T_world_from_camera_est.txt")
    setup_camera(model, cam_id, cam_to_world, width, height, K)
    renderer = mujoco.Renderer(model, width=width, height=height)

    for demo_name in tqdm(dataset):
        demo = dataset[demo_name]
        if "_follower_right_joint_states" not in demo:
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
    args = parser.parse_args()

    main(args)
