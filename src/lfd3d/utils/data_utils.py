import os

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from manopth.manolayer import ManoLayer
from pytorch3d.structures import Pointclouds

MANO_ROOT = "mano/models"


class MANOInterface:
    def __init__(self):
        self.add_back_legacy_types_numpy()
        if HydraConfig.initialized():
            original_cwd = HydraConfig.get().runtime.cwd
        else:
            original_cwd = os.getcwd()
        self.manolayer = ManoLayer(
            mano_root=f"{original_cwd}/{MANO_ROOT}",
            use_pca=False,
            ncomps=45,
            flat_hand_mean=True,
            side="right",
        )

    def add_back_legacy_types_numpy(self):
        """
        A hacky way to add back deprecated imports into numpy.
        """
        np.bool = np.bool_
        np.int = np.int_
        np.float = np.float64
        np.complex = np.complex128
        np.object = np.object_
        np.str = np.str_

    def get_hand_params(self, theta, beta):
        theta = torch.from_numpy(theta).unsqueeze(0)
        beta = torch.from_numpy(beta).unsqueeze(0)
        hand_verts, hand_joints = self.manolayer(theta, beta)
        return hand_verts, hand_joints


def collate_pcd_fn(batch):
    """
    Generic collate function that handles different data types:
    - Point clouds: detected by checking shape and creating Pointclouds objects
    - Strings: stored as lists without modification
    - Tensors: stacked into batches

    Args:
        batch: List of dictionaries containing the items from dataset
    Returns:
        Collated dictionary with properly batched items
    """
    keys = batch[0].keys()
    collated_batch = {}

    # Process each key
    for key in keys:
        values = [item[key] for item in batch]

        # Detect type of first element to determine collation strategy
        sample = values[0]

        if isinstance(sample, str):
            collated_batch[key] = values

        # handle anchor pcd separately ...
        elif "anchor" in key:
            if "anchor_pcd" in collated_batch:
                continue
            anchor_pcds = [
                torch.as_tensor(item["anchor_pcd"]).float() for item in batch
            ]
            anchor_feat_pcds = [
                torch.as_tensor(item["anchor_feat_pcd"]).float() for item in batch
            ]
            anchor_pointclouds = Pointclouds(
                points=anchor_pcds, features=anchor_feat_pcds
            )
            collated_batch["anchor_pcd"] = anchor_pointclouds

        # Process point clouds and exclude intrinsics
        elif (
            isinstance(sample, np.ndarray)
            and len(sample.shape) == 2
            and sample.shape[1] == 3
            and key not in ["intrinsics", "augment_R"]
        ):
            tensor_values = [torch.as_tensor(v).float() for v in values]
            collated_batch[key] = Pointclouds(points=tensor_values)
            # If this is the first point cloud, calculate padding mask
            if "padding_mask" not in collated_batch:
                batch_size, max_points, *_ = collated_batch[key].points_padded().shape
                num_points = collated_batch[key].num_points_per_cloud()
                padding_mask = torch.arange(max_points)[None, :] < num_points[:, None]
                collated_batch["padding_mask"] = padding_mask

        elif isinstance(sample, (np.ndarray, list, int)) or torch.is_tensor(sample):
            # Convert to tensors if they aren't already
            tensor_values = [torch.as_tensor(v) for v in values]
            collated_batch[key] = torch.stack(tensor_values)

        else:
            raise ValueError("Unexpected type for key:", key)
    assert collated_batch["augment_t"].mean().item() == 0.0
    return collated_batch


def combine_meshes(meshes):
    """
    Combine multiple Open3D meshes into a single mesh with proper vertex and triangle indexing.

    Args:
        meshes: List of Open3D triangle meshes

    Returns:
        combined_mesh: A single Open3D triangle mesh
    """
    import open3d as o3d

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


def video_to_numpy(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = [frame for ret, frame in iter(lambda: cap.read(), (False, None))]
    cap.release()
    vid = np.array(frames)
    vid = np.array([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in vid])
    return vid
