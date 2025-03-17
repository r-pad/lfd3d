import matplotlib.pyplot as plt
import mujoco
import numpy as np
import open3d as o3d
import trimesh
import zarr
from robot_descriptions.loaders.mujoco import load_robot_description
from tqdm import tqdm


def get_camera_matrices(model, data, cam_id, width, height):
    mujoco.mj_forward(model, data)
    # Get camera position and orientation
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    # MuJoCo gives us camera-to-world transform
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = cam_mat
    cam_to_world[:3, 3] = cam_pos

    # Invert to get world-to-camera transform
    world_to_cam = np.linalg.inv(cam_to_world)

    # Fix the horizontal flip issue
    # This accounts for MuJoCo's different coordinate conventions
    flip_x = np.eye(4)
    flip_x[0, 0] = -1  # Flip X axis

    # Apply the correction
    world_to_cam = world_to_cam @ flip_x

    # Compute intrinsics from field of view
    fovy = model.cam_fovy[cam_id]
    f = height / (2 * np.tan(np.deg2rad(fovy) / 2))

    # Camera intrinsics matrix
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])

    return K, world_to_cam


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


def get_right_gripper_mesh(mj_model, mj_data):
    """
    Extract the visual meshes of the right gripper from an MJCF model.

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

            # # Get the mesh ID associated with this geom
            # mesh_id = mj_model.geom_dataid[geom_id]
            # # Get the mesh name using mj_id2name with mjOBJ_MESH
            # mesh_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            # print(f"Geom {geom_id} uses mesh '{mesh_name}'")

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


# Path and data loading
root = "/data/sriram/rpad_foxglove/pick_mug_all.zarr"
dataset = zarr.group(root)
model = load_robot_description("aloha_mj_description")
data = mujoco.MjData(model)

# Hijack existing camera
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "teleoperator_pov")
height, width = 480, 640
K, world_to_cam = get_camera_matrices(model, data, cam_id, width, height)
sample_n_points = 500
RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)

visualize = False
renderer = mujoco.Renderer(model, width=width, height=height)
# viewer = MujocoViewer(model, data)

for demo_name in tqdm(dataset):
    demo = dataset[demo_name]

    if "_puppet_right_joint_states" not in demo:
        print("Human demo. Skipping.")
        continue

    # Extract joint positions from the demo
    joint_positions = demo["_puppet_right_joint_states"]["pos"][:]
    num_timesteps, num_joints = joint_positions.shape
    POINTS = []
    for t in range(num_timesteps):
        data.qpos[8 : 8 + 6] = joint_positions[t, :6]
        data.qpos[8 + 6 : 8 + 8] = joint_positions[t, [7, 7]]
        mujoco.mj_forward(model, data)
        # viewer.render()

        meshes = get_right_gripper_mesh(model, data)
        mesh = combine_meshes(meshes)
        mesh_ = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
        )
        points_, faces_ = trimesh.sample.sample_surface(mesh_, sample_n_points, seed=42)

        # Project gripper urdf to image
        homogenous_append = np.ones((sample_n_points, 1))
        gripper_urdf_3d_pos = np.concatenate([points_, homogenous_append], axis=-1)[
            :, :, None
        ]
        urdf_cam3dcoords = (world_to_cam @ gripper_urdf_3d_pos)[:, :3].squeeze(2)

        POINTS.append(urdf_cam3dcoords)

        if visualize:
            os.makedirs("mujoco_renders", exist_ok=True)
            renderer.update_scene(data, camera="teleoperator_pov")
            image = renderer.render()

            urdf_proj_hom = (K @ urdf_cam3dcoords.T).T
            urdf_proj = (urdf_proj_hom / urdf_proj_hom[:, 2:])[:, :2]
            urdf_proj = np.clip(urdf_proj, [0, 0], [width - 1, height - 1]).astype(int)
            image[urdf_proj[:, 1], urdf_proj[:, 0]] = GREEN

            plt.imsave(f"mujoco_renders/{str(t).zfill(5)}.png", image)

    POINTS = np.array(POINTS)
    demo.create_dataset("gripper_pos", data=POINTS)
