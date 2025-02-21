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
        np.float = np.float_
        np.complex = np.complex_
        np.object = np.object_
        np.str = np.str_

    def get_hand_params(self, theta, beta):
        theta = torch.from_numpy(theta).unsqueeze(0)
        beta = torch.from_numpy(beta).unsqueeze(0)
        hand_verts, hand_joints = self.manolayer(theta, beta)
        return hand_verts, hand_joints


def collate_pcd_fn(batch):
    """
    Custom collate function that handles:
    - Point clouds (action_pcd, anchor_pcd and cross_displacement)
    - Strings (caption and vid_name)
    - Regular tensors (intrinsics, rgbs, depths, start2end)

    Args:
        batch: List of dictionaries containing the items from dataset

    Returns:
        Collated dictionary with properly batched items
    """
    # Initialize lists to store items
    action_pcds = []
    anchor_pcds = []
    anchor_feat_pcds = []
    cross_displacements = []
    captions = []
    text_embeds = []
    vid_names = []
    intrinsics = []
    rgbs = []
    depths = []
    start2ends = []
    pcd_means = []
    pcd_stds = []

    # Separate items from batch
    for item in batch:
        # Convert point clouds to tensors if they aren't already
        action_pcd = torch.as_tensor(item["action_pcd"]).float()
        anchor_pcd = torch.as_tensor(item["anchor_pcd"]).float()
        anchor_feat_pcd = torch.as_tensor(item["anchor_feat_pcd"]).float()
        cross_displacement = torch.as_tensor(item["cross_displacement"]).float()

        action_pcds.append(action_pcd)
        anchor_feat_pcds.append(anchor_feat_pcd)
        anchor_pcds.append(anchor_pcd)
        cross_displacements.append(cross_displacement)
        captions.append(item["caption"])
        vid_names.append(item["vid_name"])

        # Convert other items to tensors if they aren't already
        intrinsics.append(torch.as_tensor(item["intrinsics"]))
        text_embeds.append(torch.as_tensor(item["text_embed"]))
        rgbs.append(torch.as_tensor(item["rgbs"]))
        depths.append(torch.as_tensor(item["depths"]))
        start2ends.append(torch.as_tensor(item["start2end"]))
        pcd_means.append(torch.as_tensor(item["pcd_mean"]))
        pcd_stds.append(torch.as_tensor(item["pcd_std"]))

    # Create Pointclouds objects
    action_pointclouds = Pointclouds(points=action_pcds)
    anchor_pointclouds = Pointclouds(points=anchor_pcds, features=anchor_feat_pcds)
    cross_displacement_pointclouds = Pointclouds(points=cross_displacements)

    batch_size, max_points, _ = action_pointclouds.points_padded().shape
    num_points = action_pointclouds.num_points_per_cloud()
    padding_mask = torch.arange(max_points)[None, :] < num_points[:, None]

    # Stack regular tensors
    intrinsics_batch = torch.stack(intrinsics)
    text_embeds_batch = torch.stack(text_embeds)
    rgbs_batch = torch.stack(rgbs)
    depths_batch = torch.stack(depths)
    start2ends_batch = torch.stack(start2ends)
    pcd_means_batch = torch.stack(pcd_means)
    pcd_stds_batch = torch.stack(pcd_stds)

    # Create the output dictionary
    collated_batch = {
        # Point clouds
        "action_pcd": action_pointclouds,
        "anchor_pcd": anchor_pointclouds,
        "cross_displacement": cross_displacement_pointclouds,
        "padding_mask": padding_mask,
        # Strings
        "caption": captions,
        "vid_name": vid_names,
        # Regular tensors
        "intrinsics": intrinsics_batch,
        "text_embed": text_embeds_batch,
        "rgbs": rgbs_batch,
        "depths": depths_batch,
        "start2end": start2ends_batch,
        "pcd_mean": pcd_means_batch,
        "pcd_std": pcd_stds_batch,
    }

    return collated_batch
