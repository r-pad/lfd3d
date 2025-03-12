import json
import os
from glob import glob

import cv2
import mujoco
import numpy as np
import open3d as o3d
import tensorflow_datasets as tfds
import trimesh
from PIL import Image
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


def video_to_numpy(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = [frame for ret, frame in iter(lambda: cap.read(), (False, None))]
    cap.release()
    vid = np.array(frames)
    vid = np.array([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in vid])
    return vid


def get_cam_to_world(cam_extrinsics):
    rotation = Rotation.from_euler("xyz", np.array(cam_extrinsics[3:])).as_matrix()
    translation = cam_extrinsics[:3]

    cam_to_world = np.zeros((4, 4))
    cam_to_world[:3, :3] = rotation
    cam_to_world[:3, 3] = translation
    cam_to_world[3, 3] = 1
    return cam_to_world


def get_gripper_mesh(mj_model, mj_data):
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


def combine_meshes(meshes):
    """
    Combine multiple Open3D meshes into a single mesh with proper vertex and triangle indexing.

    Args:
        meshes: List of Open3D triangle meshes

    Returns:
        combined_mesh: A single Open3D triangle mesh
    """
    # Initialize vertices and triangles lists
    vertices = []
    triangles = []
    vertex_offset = 0

    # Combine meshes
    for mesh in meshes:
        # Convert mesh vertices and triangles to numpy arrays
        mesh_vertices = np.asarray(mesh.vertices)
        mesh_triangles = np.asarray(mesh.triangles)

        # Add vertices to the combined list
        vertices.append(mesh_vertices)

        # Adjust triangle indices and add to the combined list
        adjusted_triangles = mesh_triangles + vertex_offset
        triangles.append(adjusted_triangles)

        # Update vertex offset for the next mesh
        vertex_offset += len(mesh_vertices)

    # Concatenate all vertices and triangles
    all_vertices = np.vstack(vertices)
    all_triangles = np.vstack(triangles)

    # Create the combined mesh
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    combined_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)

    combined_mesh.compute_vertex_normals()

    combined_mesh.remove_duplicated_vertices()
    combined_mesh.remove_duplicated_triangles()
    combined_mesh.remove_degenerate_triangles()

    return combined_mesh


# "Average" intrinsics for Zed
K = np.array(
    [
        [522.42260742, 0.0, 653.9631958],
        [0.0, 522.42260742, 358.79196167],
        [0.0, 0.0, 1.0],
    ]
)
K = K / 4  # downsampled images
K[2, 2] = 1
height, width = 180, 320
RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)

sample_n_points = 500
model = load_robot_description("robotiq_2f85_mj_description")
data = mujoco.MjData(model)
root_joint_adr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_joint")
assert (
    root_joint_adr != -1
), "Modify MJCF following the instructions at the top of the script."
start_qpos = model.jnt_qposadr[root_joint_adr]

root = "/data/sriram/DROID/droid"
builder = tfds.builder_from_directory(builder_dir=f"{root}")
dataset = builder.as_dataset(split="train")
metadata_dir = "/data/sriram/DROID/droid_raw"
output_dir = "/data/sriram/DROID/droid_gripper_pcd"
viz_dir = "droid_gripper_pcd_viz"

# Can use either 1 or 2, no particular reason
cam_extrinsics_key = "ext1_cam_extrinsics"
cam_image_key = "exterior_image_1_left"


for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
    fpath = item["episode_metadata"]["file_path"].numpy().decode("utf-8")
    if "failure" in fpath:
        continue

    if os.path.exists(f"{output_dir}/{idx}.npz"):
        continue

    steps = [i for i in item["steps"]]
    images = np.array([i["observation"][cam_image_key] for i in steps])
    goal_text = steps[0]["language_instruction"].numpy().decode("utf-8")

    if goal_text == "":
        continue

    subfolder_path = "/".join(fpath.split("/")[5:9])  # Extract path for raw dataset
    metadata_file = glob(f"{metadata_dir}/1.0.1/{subfolder_path}/metadata*json")
    if metadata_file == []:
        print(f"Could not find find metadata in {subfolder_path}")
        continue

    metadata_file = metadata_file[0]
    with open(metadata_file) as f:
        metadata = json.load(f)

    gripper_cartesian_pos = np.array(
        [i["observation"]["cartesian_position"] for i in steps]
    )
    gripper_action = np.array([i["observation"]["gripper_position"] for i in steps])
    extrinsics_left = metadata[cam_extrinsics_key]

    cam_to_world = get_cam_to_world(extrinsics_left)
    # Get the world to camera transformation
    world_to_cam = np.linalg.inv(cam_to_world)

    num_images = images.shape[0]
    gripper_action = np.clip(
        gripper_action, 0, 0.8
    )  # set max limit to prevent the gripper *bending inwards*
    gripper_pcds = []

    for i in range(gripper_cartesian_pos.shape[0]):
        # Extract gripper pose
        ee_pos = gripper_cartesian_pos[i, :3]  # Position [x, y, z]
        ee_euler = gripper_cartesian_pos[i, 3:6]  # Orientation [rx, ry, rz]

        # Convert Euler angles to quaternion
        R_ee = Rotation.from_euler("xyz", ee_euler)
        # Getting it to align in mujoco ...
        R_correction = Rotation.from_euler("z", 90, degrees=True)
        R_ee = R_ee * R_correction

        ee_quat = R_ee.as_quat()  # [x, y, z, w]
        ee_quat = ee_quat[[3, 0, 1, 2]]  # Reorder to [w, x, y, z] for MuJoCo

        # Set gripper base pose in qpos (free joint: 7D)
        data.qpos[start_qpos : start_qpos + 3] = ee_pos
        data.qpos[start_qpos + 3 : start_qpos + 7] = ee_quat

        data.qpos[7] = gripper_action[i]
        data.qpos[11] = gripper_action[i]

        # Update kinematics
        mujoco.mj_step(model, data)

        # Extract the mesh from the URDF, simplify it
        # Sample point cloud from its surface for saving
        # NOTE: Keeping a seed is very important so that we maintain correspondences over samples.
        meshes = get_gripper_mesh(model, data)
        mesh = combine_meshes(meshes)
        mesh_ = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
        )
        # In world frame
        points_, faces_ = trimesh.sample.sample_surface(mesh_, sample_n_points, seed=42)

        homogenous_append = np.ones((sample_n_points, 1))
        gripper_urdf_3d_pos = np.concatenate([points_, homogenous_append], axis=-1)[
            :, :, None
        ]
        urdf_cam3dcoords = (world_to_cam @ gripper_urdf_3d_pos)[:, :3].squeeze(2)

        # Save gripper pcd in camera frame
        gripper_pcds.append(urdf_cam3dcoords)

        # Save some visualizations
        if idx % 100 == 0:
            os.makedirs(f"{viz_dir}/{idx}", exist_ok=True)
            # Project gripper pcd to image
            urdf_proj_hom = (K @ urdf_cam3dcoords.T).T
            urdf_proj = (urdf_proj_hom / urdf_proj_hom[:, 2:])[:, :2]
            urdf_proj = np.clip(urdf_proj, [0, 0], [width - 1, height - 1]).astype(int)
            images[i][urdf_proj[:, 1], urdf_proj[:, 0]] = GREEN

            # Save the image
            Image.fromarray(images[i]).save(
                f"{viz_dir}/{idx}/frame_{str(i).zfill(5)}.png"
            )

    gripper_pcds_arr = np.array(gripper_pcds, dtype=np.float16)
    np.savez_compressed(f"{output_dir}/{idx}.npz", gripper_pcds_arr)
