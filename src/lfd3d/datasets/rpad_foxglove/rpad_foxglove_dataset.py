import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torchdatasets as td
import zarr
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from transformers import AutoModel, AutoProcessor

from lfd3d.datasets.base_data import BaseDataModule
from lfd3d.datasets.rgb_text_feature_gen import (
    get_dinov2_image_embedding,
    get_siglip_text_embedding,
)


class RpadFoxgloveDataset(td.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split

        self.cache_dir = dataset_cfg.cache_dir
        self.dataset_cfg = dataset_cfg

        self.data_sources = dataset_cfg.data_sources
        self.current_dir = os.path.dirname(__file__)
        with open(f"{self.current_dir}/{split}.json") as f:
            self.split_names = json.load(f)

        # Voxel size for downsampling
        self.voxel_size = 0.03

        # Target shape of images (same as DINOv2)
        self.target_shape = 224
        self.rgb_preprocess = transforms.Compose(
            [
                transforms.Resize(
                    self.target_shape,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(self.target_shape),
            ]
        )
        self.depth_preprocess = transforms.Compose(
            [
                transforms.Resize(
                    self.target_shape,
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.CenterCrop(self.target_shape),
            ]
        )

        self.captions = {}
        self.dataset = zarr.group(root)
        self.dataset_index = self.expand_all_events()
        self.size = len(self.dataset_index)

        self.siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(
            "cpu"
        )
        self.siglip_processor = AutoProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(
            "cpu"
        )

    def __len__(self):
        return self.size

    def is_in_data_source(self, demo):
        """
        Filters out demos which aren't specified as a data source.
        i.e. if we specify "aloha" as a data source, filter out all human demos
        """
        # Currently identifying demo type by checking number of vertices
        # For human data, we have 778 vertices from the MANO mesh
        # For robot data, we're sampling 500 points from the mesh.
        # HACK: This is pretty hacky, should probably find a better way to do this.
        n_points = demo["gripper_pos"].shape[1]
        if n_points == 778:
            demo_type = "human"
        elif n_points == 500:
            demo_type = "aloha"
        else:
            raise NotImplementedError

        return demo_type in self.data_sources

    def expand_all_events(self):
        """This function *expands* each event to have an associated event_idx.
        Updates `self.captions` with each event and its various subgoals

        Returns:
            expanded_index (list of tuples (int, int)): A list of tuples
                  where each tuple contains the event index and the corresponding
                  chunk index.
        """
        expanded_index = []

        for demo_name in self.dataset:
            if demo_name not in self.split_names:
                continue

            demo = self.dataset[demo_name]

            if "gripper_pos" not in demo.keys():
                print(f"No GT found for {demo_name}")
                continue

            if not self.is_in_data_source(demo):
                continue

            events = demo["events"]
            num_events = len(events["event"])
            expanded_event_idx = [(demo_name, i) for i in range(num_events)]
            expanded_event_caption = {
                (demo_name, i): (
                    events["end"][i],
                    events["event"][i].replace("_", " "),
                )
                for i in range(num_events)
            }

            expanded_index.extend(expanded_event_idx)
            self.captions.update(expanded_event_caption)
        return expanded_index

    def load_camera_params(self, demo_name):
        demo = self.dataset[demo_name]
        K = demo["_rgb_camera_info"]["k"][0]
        height = demo["_rgb_camera_info"]["height"][0]
        width = demo["_rgb_camera_info"]["width"][0]
        return K, (height, width)

    def get_scaled_intrinsics(self, K, orig_shape):
        # Getting scale factor from torchvision.transforms.Resize behaviour
        K_ = K.copy()

        scale_factor = self.target_shape / min(orig_shape)

        # Apply the scale factor to the intrinsics
        K_[0, 0] *= scale_factor  # fx
        K_[1, 1] *= scale_factor  # fy
        K_[0, 2] *= scale_factor  # cx
        K_[1, 2] *= scale_factor  # cy

        # Adjust the principal point (cx, cy) for the center crop
        crop_offset_x = (orig_shape[1] * scale_factor - self.target_shape) / 2
        crop_offset_y = (orig_shape[0] * scale_factor - self.target_shape) / 2

        # Adjust the principal point (cx, cy) for the center crop
        K_[0, 2] -= crop_offset_x  # Adjust cx for crop
        K_[1, 2] -= crop_offset_y  # Adjust cy for crop
        return K_

    def get_event_start_end_ts(self, demo_name, subgoal_idx):
        demo = self.dataset[demo_name]

        event_end_ts = datetime.fromisoformat(
            demo["events"]["end"][subgoal_idx]
        ).timestamp()

        if subgoal_idx == 0:
            # First frame where we have non-zero gripper_pos
            # For estimating hand pose, we add zeros if hand is not detected.
            event_start_idx = np.argmax(
                np.any(np.asarray(demo["gripper_pos"]) != 0, axis=(1, 2))
            )
            event_start_ts = demo["_rgb_image_rect"]["ts"][event_start_idx]
        else:
            # Start timestamp is end timestamp of previous subgoal
            event_start_ts = datetime.fromisoformat(
                demo["events"]["end"][subgoal_idx - 1]
            ).timestamp()
        return event_start_ts, event_end_ts

    def load_rgbd(self, demo_name, subgoal_idx, K):
        demo = self.dataset[demo_name]

        event_start_ts, event_end_ts = self.get_event_start_end_ts(
            demo_name, subgoal_idx
        )

        rgb_ts = demo["_rgb_image_rect"]["publish_ts"]
        depth_ts = demo["_depth_registered_image_rect"]["publish_ts"]

        event_start_idx_rgb = np.searchsorted(rgb_ts, event_start_ts)
        event_end_idx_rgb = np.searchsorted(rgb_ts, event_end_ts)

        event_start_idx_depth = np.searchsorted(depth_ts, event_start_ts)
        event_end_idx_depth = np.searchsorted(depth_ts, event_end_ts)

        # Return rgb/depth at beginning and end of event
        rgb_init = Image.fromarray(demo["_rgb_image_rect"]["img"][event_start_idx_rgb])
        rgb_init = np.asarray(self.rgb_preprocess(rgb_init))
        rgb_end = Image.fromarray(demo["_rgb_image_rect"]["img"][event_end_idx_rgb])
        rgb_end = np.asarray(self.rgb_preprocess(rgb_end))
        rgbs = np.array([rgb_init, rgb_end])

        depth_init = (
            (
                demo["_depth_registered_image_rect"]["img"][event_start_idx_depth]
                / 1000.0
            )
            .squeeze()
            .astype(np.float32)
        )
        depth_init = Image.fromarray(depth_init)
        depth_init = np.asarray(self.depth_preprocess(depth_init))
        depth_end = (
            (demo["_depth_registered_image_rect"]["img"][event_end_idx_depth] / 1000.0)
            .squeeze()
            .astype(np.float32)
        )
        depth_end = Image.fromarray(depth_end)
        depth_end = np.asarray(self.depth_preprocess(depth_end))
        depths = np.array([depth_init, depth_end])

        return rgbs, depths, event_start_idx_rgb, event_end_idx_rgb

    def load_gripper_pcd(self, demo_name, event_start_idx, event_end_idx):
        demo = self.dataset[demo_name]
        start_tracks = demo["gripper_pos"][event_start_idx]
        end_tracks = demo["gripper_pos"][event_end_idx]
        return start_tracks, end_tracks

    def compute_rgb_text_feat(self, rgb, text):
        """
        Compute RGB/text features generated with DINOv2 and SIGLIP
        """
        # Compute features on CPU to avoid CUDA multiprocessing issues
        # We're only computing the features once and caching so its okay.
        text_embed = get_siglip_text_embedding(
            text,
            siglip=self.siglip,
            siglip_processor=self.siglip_processor,
            device="cpu",
        )
        rgb_embed = get_dinov2_image_embedding(
            Image.fromarray(rgb), dinov2=self.dinov2, device="cpu"
        )

        # Compress RGB features
        pca_n_components = 256
        pca_model = PCA(n_components=pca_n_components)
        rgb_embed = pca_model.fit_transform(rgb_embed.reshape(-1, rgb_embed.shape[2]))
        rgb_embed = rgb_embed.reshape(
            self.target_shape, self.target_shape, pca_n_components
        )

        return rgb_embed, text_embed

    def get_scene_pcd(self, rgb_embed, depth, K):
        height, width = depth.shape
        # Create pixel coordinate grid
        x = np.arange(width)
        y = np.arange(height)
        x_grid, y_grid = np.meshgrid(x, y)

        # Flatten grid coordinates and depth
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = depth.flatten()
        feat_flat = rgb_embed.reshape(-1, rgb_embed.shape[-1])

        # Remove points with invalid depth
        valid_depth = np.logical_and(z_flat > 0, z_flat < 2)
        x_flat = x_flat[valid_depth]
        y_flat = y_flat[valid_depth]
        z_flat = z_flat[valid_depth]
        feat_flat = feat_flat[valid_depth]

        # Create homogeneous pixel coordinates
        pixels = np.stack([x_flat, y_flat, np.ones_like(x_flat)], axis=0)

        # Unproject points using K inverse
        K_inv = np.linalg.inv(K)
        points = K_inv @ pixels
        points = points * z_flat
        points = points.T  # Shape: (N, 3)

        scene_pcd_o3d = o3d.geometry.PointCloud()
        scene_pcd_o3d.points = o3d.utility.Vector3dVector(points)
        scene_pcd_o3d_downsample = scene_pcd_o3d.voxel_down_sample(
            voxel_size=self.voxel_size
        )

        scene_pcd = np.asarray(scene_pcd_o3d_downsample.points)

        # Find closest indices in the original point cloud so we can index the features
        downsampled_indices = [
            np.argmin(np.linalg.norm(points - scene_pcd[i], axis=1))
            for i in range(scene_pcd.shape[0])
        ]
        scene_feat_pcd = feat_flat[downsampled_indices]
        return scene_pcd, scene_feat_pcd

    def __getitem__(self, idx):
        demo_name, subgoal_idx = self.dataset_index[idx]

        K_ = self.get_scaled_intrinsics(*self.load_camera_params(demo_name))

        _, caption = self.captions[(demo_name, subgoal_idx)]
        start2end = torch.eye(4)  # Static camera

        rgbs, depths, event_start_idx, event_end_idx = self.load_rgbd(
            demo_name, subgoal_idx, K_
        )

        start_tracks, end_tracks = self.load_gripper_pcd(
            demo_name, event_start_idx, event_end_idx
        )

        rgb_embed, text_embed = self.compute_rgb_text_feat(rgbs[0], caption)
        start_scene_pcd, start_scene_feat_pcd = self.get_scene_pcd(
            rgb_embed, depths[0], K_
        )

        # Center on action_pcd
        action_pcd_mean = start_tracks.mean(axis=0)
        start_tracks = start_tracks - action_pcd_mean
        end_tracks = end_tracks - action_pcd_mean
        start_scene_pcd = start_scene_pcd - action_pcd_mean
        # Standardize on scene_pcd
        scene_pcd_std = start_scene_pcd.std(axis=0)
        start_tracks = start_tracks / scene_pcd_std
        end_tracks = end_tracks / scene_pcd_std
        start_scene_pcd = start_scene_pcd / scene_pcd_std

        # collate_pcd_fn handles batching of the point clouds
        item = {
            "action_pcd": start_tracks,
            "anchor_pcd": start_scene_pcd,
            "anchor_feat_pcd": start_scene_feat_pcd,
            "caption": caption,
            "text_embed": text_embed,
            "cross_displacement": end_tracks - start_tracks,
            "intrinsics": K_,
            "rgbs": rgbs,
            "depths": depths,
            "start2end": start2end,
            "vid_name": demo_name,
            "pcd_mean": action_pcd_mean,
            "pcd_std": scene_pcd_std,
        }
        return item


class RpadFoxgloveDataModule(BaseDataModule):
    def setup(self, stage: str = "fit"):
        self.stage = stage

        self.train_dataset = RpadFoxgloveDataset(self.root, self.dataset_cfg, "train")
        self.val_dataset = RpadFoxgloveDataset(self.root, self.dataset_cfg, "val")
        if self.train_dataset.cache_dir:
            self.train_dataset.cache(
                td.cachers.Pickle(Path(self.train_dataset.cache_dir))
            )
            self.val_dataset.cache(
                td.cachers.Pickle(Path(self.train_dataset.cache_dir) / "val")
            )
        self.test_dataset = RpadFoxgloveDataset(self.root, self.dataset_cfg, "test")
