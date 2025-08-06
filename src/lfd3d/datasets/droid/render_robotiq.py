import argparse
import json
import os
import re
from glob import glob

import cv2
import mujoco
import numpy as np
import open3d as o3d
import tensorflow_datasets as tfds
import trimesh
from lfd3d.utils.data_utils import combine_meshes
from robot_descriptions.loaders.mujoco import load_robot_description
from scipy.spatial.transform import Rotation
from tqdm import tqdm

"""
NOTE: MJCF needs to be modified to add free joint
TODO: Dynamically add to MJCF to avoid this.
Location: ~/.cache/robot_descriptions/mujoco_menagerie/robotiq_2f85/2f85.xml

   <worldbody>
   <body name="base_mount" pos="0 0 0.007" childclass="2f85">
     <freejoint name="root_joint"/> !!!TO BE ADDED!!!
"""


def get_cam_to_world(cam_extrinsics):
    rotation = Rotation.from_euler("xyz", np.array(cam_extrinsics[3:])).as_matrix()
    translation = cam_extrinsics[:3]

    cam_to_world = np.zeros((4, 4))
    cam_to_world[:3, :3] = rotation
    cam_to_world[:3, 3] = translation
    cam_to_world[3, 3] = 1
    return cam_to_world


def get_gripper_mesh(mj_model, mj_data):
    """
    Get robotiq gripper mesh from MJCF.
    """
    meshes = []
    # Iterate over all geoms attached to this body
    for geom_id in range(mj_model.ngeom):
        # Get the geom's world position and orientation
        geom_pos = mj_data.geom_xpos[geom_id]  # World position of the body
        geom_mat = mj_data.geom_xmat[geom_id].reshape(3, 3)  # World rotation matrix

        if (
            mj_model.geom_group[geom_id] != 2
            and mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        ):
            # Get the mesh ID associated with this geom
            mesh_id = mj_model.geom_dataid[geom_id]
            if mesh_id >= 0:  # Ensure the geom has a valid mesh
                # Extract vertex data from the mesh
                mesh_vert_adr = mj_model.mesh_vertadr[
                    mesh_id
                ]  # Starting index of vertices
                mesh_vert_num = mj_model.mesh_vertnum[mesh_id]  # Number of vertices
                mesh_face_adr = mj_model.mesh_faceadr[
                    mesh_id
                ]  # Starting index of faces
                mesh_face_num = mj_model.mesh_facenum[mesh_id]  # Number of faces
                vertices_local = mj_model.mesh_vert[
                    mesh_vert_adr : mesh_vert_adr + mesh_vert_num
                ].copy()  # Local coordinates
                faces = mj_model.mesh_face[
                    mesh_face_adr : mesh_face_adr + mesh_face_num
                ].copy()

                # Transform vertices to world coordinates
                vertices_world = vertices_local @ geom_mat.T + geom_pos
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                meshes.append(mesh)
    return meshes


def search_annotations_json(
    file_path, droid_language_annotations, droid_language_annotations_keys
):
    query_p1 = file_path.split("/")[0]
    query_p2 = ".*"
    query_p3 = file_path.split("/")[2]
    query_p4 = file_path[-13:-5]
    try:
        hours, minutes, seconds = query_p4.split(":")
        query_p4 = f"-{hours}h-{minutes}m-{seconds}s"
    except:
        hours, minutes, seconds = query_p4.split("_")
        query_p4 = f"-{hours}h-{minutes}m-{seconds}s"
    query_key = query_p1 + query_p2 + query_p3 + query_p4

    regex = re.compile(query_key)

    match = None
    for index, key in enumerate(droid_language_annotations_keys):
        if regex.search(key):
            match = key
            droid_language_annotations_keys.pop(index)
            break

    if match is None:
        return None
    return droid_language_annotations[match]["language_instruction1"]


def process_item(
    item,
    idx,
    camera_intrinsics,
    height,
    width,
    sample_n_points,
    output_dir,
    viz_dir,
    metadata_dir,
    cam_extrinsics_key,
    cam_image_key,
    droid_language_annotations,
    droid_language_annotations_keys,
    idx_to_fname_mapping,
):
    fpath = item["episode_metadata"]["file_path"].numpy().decode("utf-8")
    if "failure" in fpath:
        return

    if os.path.exists(f"{output_dir}/{idx}.npz"):
        return

    steps = [i for i in item["steps"]]
    goal_text = steps[0]["language_instruction"].numpy().decode("utf-8")

    if not goal_text:
        # Try backup search in the json
        goal_text = search_annotations_json(
            idx_to_fname_mapping[idx],
            droid_language_annotations,
            droid_language_annotations_keys,
        )
        if not goal_text:
            return None

    subfolder_path = "/".join(fpath.split("/")[5:9])
    metadata_file = glob(f"{metadata_dir}/1.0.1/{subfolder_path}/metadata*json")
    if not metadata_file:
        print(f"Could not find metadata in {subfolder_path}")
        return

    with open(metadata_file[0]) as f:
        metadata = json.load(f)
    cam_serial = metadata["ext1_cam_serial"]
    cam_params = camera_intrinsics[cam_serial]
    K = np.array(
        [
            [cam_params["fx"], 0.0, cam_params["cx"]],
            [0.0, cam_params["fy"], cam_params["cy"]],
            [0.0, 0.0, 1.0],
        ]
    )
    K = K / 4  # downsampled images
    K[2, 2] = 1
    baseline = cam_params["baseline"] / 1000

    gripper_cartesian_pos = np.array(
        [i["observation"]["cartesian_position"] for i in steps]
    )
    gripper_action = np.array([i["observation"]["gripper_position"] for i in steps])
    extrinsics_left = metadata[cam_extrinsics_key]

    cam_to_world = get_cam_to_world(extrinsics_left)
    world_to_cam = np.linalg.inv(cam_to_world)

    gripper_action = np.clip(gripper_action, 0, 0.8)
    gripper_pcds = []

    model = load_robot_description("robotiq_2f85_mj_description")
    data = mujoco.MjData(model)
    root_joint_adr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_joint")
    assert (
        root_joint_adr != -1
    ), "Modify MJCF following the instructions at the top of the script."
    start_qpos = model.jnt_qposadr[root_joint_adr]

    for i in range(gripper_cartesian_pos.shape[0]):
        ee_pos = gripper_cartesian_pos[i, :3]
        ee_euler = gripper_cartesian_pos[i, 3:6]

        R_ee = Rotation.from_euler("xyz", ee_euler)
        R_correction = Rotation.from_euler("z", 90, degrees=True)
        R_ee = R_ee * R_correction

        ee_quat = R_ee.as_quat()
        ee_quat = ee_quat[[3, 0, 1, 2]]

        data.qpos[start_qpos : start_qpos + 3] = ee_pos
        data.qpos[start_qpos + 3 : start_qpos + 7] = ee_quat

        data.qpos[7] = gripper_action[i]
        data.qpos[11] = gripper_action[i]

        mujoco.mj_step(model, data)

        meshes = get_gripper_mesh(model, data)
        mesh = combine_meshes(meshes)
        mesh_ = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
        )
        # Extract the mesh from the URDF, simplify it
        # Sample point cloud from its surface for saving
        # NOTE: Keeping a seed is very important so that we maintain correspondences over samples.
        points_, _ = trimesh.sample.sample_surface(mesh_, sample_n_points, seed=42)

        homogenous_append = np.ones((sample_n_points, 1))
        gripper_urdf_3d_pos = np.concatenate([points_, homogenous_append], axis=-1)[
            :, :, None
        ]
        urdf_cam3dcoords = (world_to_cam @ gripper_urdf_3d_pos)[:, :3].squeeze(2)

        gripper_pcds.append(urdf_cam3dcoords)

        if idx % 100 == 0:
            images = np.array([i["observation"][cam_image_key] for i in steps])
            os.makedirs(f"{viz_dir}/{idx}", exist_ok=True)
            urdf_proj_hom = (K @ urdf_cam3dcoords.T).T
            urdf_proj = (urdf_proj_hom / urdf_proj_hom[:, 2:])[:, :2]
            urdf_proj = np.clip(urdf_proj, [0, 0], [width - 1, height - 1]).astype(int)
            images[i][urdf_proj[:, 1], urdf_proj[:, 0]] = [0, 255, 0]

            cv2.imwrite(f"{viz_dir}/{idx}/frame_{str(i).zfill(5)}.png", images[i])

    gripper_pcds_arr = np.array(gripper_pcds, dtype=np.float16)
    np.savez_compressed(f"{output_dir}/{idx}.npz", gripper_pcds_arr)


def main(args):
    height, width = 180, 320
    root = args.root
    builder = tfds.builder_from_directory(builder_dir=root)
    dataset = builder.as_dataset(split="train")
    metadata_dir = args.metadata_dir
    output_dir = args.output_dir
    viz_dir = args.viz_dir

    # Can use either 1 or 2, no particular reason
    cam_extrinsics_key = "ext1_cam_extrinsics"
    cam_image_key = "exterior_image_1_left"

    with open(f"{root}/../droid_language_annotations.json") as f:
        droid_language_annotations = json.load(f)
    droid_language_annotations_keys = list(droid_language_annotations.keys())
    with open("idx_to_fname_mapping.json") as f:
        idx_to_fname_mapping = json.load(f)
    with open("zed_intrinsics.json") as f:
        camera_intrinsics = json.load(f)

    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        process_item(
            item,
            idx,
            camera_intrinsics,
            height,
            width,
            args.sample_n_points,
            output_dir,
            viz_dir,
            metadata_dir,
            cam_extrinsics_key,
            cam_image_key,
            droid_language_annotations,
            droid_language_annotations_keys,
            idx_to_fname_mapping,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process DROID dataset for gripper point clouds"
    )

    parser.add_argument(
        "--root",
        type=str,
        default="/data/sriram/DROID/droid",
        help="Root directory of the DROID dataset",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="/data/sriram/autobot_mount/DROID/droid_raw",
        help="Directory containing metadata files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/sriram/DROID/droid_gripper_pcd",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--viz_dir",
        type=str,
        default="droid_gripper_pcd_viz",
        help="Directory for visualization outputs",
    )
    parser.add_argument(
        "--sample_n_points",
        type=int,
        default=500,
        help="Number of points to sample from gripper mesh",
    )

    args = parser.parse_args()

    main(args)
