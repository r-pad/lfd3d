import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lfd3d.datasets.base_data import BaseDataModule, BaseDataset


class GenGoalGenDataset(BaseDataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        assert split == "test"
        self.root = root
        self.dataset_dir = self.root

        with open(f"{root}/fileList.json") as f:
            self.data_files = json.load(f)
        self.dataset_cfg = dataset_cfg
        self.size = len(self.data_files)

        self.scale_factor = 0.25
        # Approximate values for Azure Kinect
        self.K = np.array(
            [
                [
                    700.0,
                    0.0,
                    640.0,
                ],
                [0.0, 700.0, 360.0],
                [0.0, 0.0, 1.0],
            ]
        )

        # scale down intrinsics
        self.K = self.K * self.scale_factor
        self.K[2, 2] = 1

        self.num_points = dataset_cfg.num_points
        self.max_depth = dataset_cfg.max_depth

    def __len__(self):
        return self.size

    def load_image_data(self, image_path):
        # Return rgb/depth at beginning and end of event
        rgb = cv2.cvtColor(
            cv2.imread(f"{self.root}/{image_path}"),
            cv2.COLOR_BGR2RGB,
        )
        rgb = cv2.resize(rgb, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        rgbs = np.array(
            [rgb, rgb]
        )  # just repeat the image twice, we don't have a goal image

        depth_path = image_path.replace("-color.png", "-depth.npy")
        # Assumes depth in mm
        depth = (np.load(f"{self.root}/{depth_path}") / 1000.0).astype(np.float32)
        depth = cv2.resize(
            depth,
            (0, 0),
            fx=self.scale_factor,
            fy=self.scale_factor,
            interpolation=cv2.INTER_NEAREST,
        )
        depths = np.array([depth, depth])

        seg_path = image_path.replace("color.png", "color_mask.npy")
        segmask = np.load(f"{self.root}/{seg_path}")
        segmask = cv2.resize(
            segmask,
            (0, 0),
            fx=self.scale_factor,
            fy=self.scale_factor,
            interpolation=cv2.INTER_NEAREST,
        )

        return rgbs, depths, segmask

    def get_action_pcd(self, depth, segmask, K):
        segmask = segmask.astype(bool)
        height, width = depth.shape
        # Create pixel coordinate grid
        x = np.arange(width)
        y = np.arange(height)
        x_grid, y_grid = np.meshgrid(x, y)

        # Only keep action pcd
        x_flat = x_grid[segmask]
        y_flat = y_grid[segmask]
        z_flat = depth[segmask]

        # Remove points with invalid depth
        valid_depth = np.logical_and(z_flat > 0, z_flat < self.max_depth)
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

        action_pcd_o3d = o3d.geometry.PointCloud()
        action_pcd_o3d.points = o3d.utility.Vector3dVector(points)
        action_pcd_o3d_downsample = action_pcd_o3d.voxel_down_sample(
            voxel_size=self.voxel_size
        )

        action_pcd = np.asarray(action_pcd_o3d_downsample.points)

        return action_pcd

    def load_rgb_text_feat(self, image_path, height, width):
        """
        Load RGB/text features generated with SIGLIP using ConceptFusion.
        """
        feat_path = image_path.replace("-color.png", "-color_feat.npz")
        features = np.load(f"{self.root}/{feat_path}")
        rgb_embed, text_embed = features["rgb_embed"], features["text_embed"].squeeze()

        upscale_by = 2
        _, pca_feat_dim = rgb_embed.shape
        rgb_embed = (
            rgb_embed.transpose(1, 0)
            .reshape(1, pca_feat_dim, height // upscale_by, width // upscale_by)
            .astype(np.float32)
        )
        rgb_embed = (
            F.interpolate(
                torch.from_numpy(rgb_embed),
                scale_factor=upscale_by,
                mode="bilinear",
                align_corners=False,
            )
            .numpy()
            .squeeze()
            .transpose(1, 2, 0)
        )
        return rgb_embed, text_embed

    def __getitem__(self, index):
        raise NotImplementedError(
            "needs to be updated."
        )
        item = self.data_files[index]
        image_path, caption = item["image"], item["action"]

        rgbs, depths, segmask = self.load_image_data(image_path)
        rgb_embed, text_embed = self.load_rgb_text_feat(
            image_path, rgbs[0].shape[0], rgbs[0].shape[1]
        )
        start2end = np.eye(4)

        anchor_pcd, anchor_feat_pcd, augment_tf = self.get_scene_pcd(
            rgb_embed, depths[0], self.K, self.num_points, self.max_depth
        )
        action_pcd = self.get_action_pcd(depths[0], segmask, self.K)

        action_pcd_mean, scene_pcd_std = self.get_normalize_mean_std(
            action_pcd, anchor_pcd, self.dataset_cfg
        )

        end_pcd = np.zeros_like(action_pcd)  # Dummy value, no GT available
        action_pcd, end_pcd, anchor_pcd = self.transform_pcds(
            action_pcd, end_pcd, anchor_pcd, action_pcd_mean, scene_pcd_std, augment_tf
        )

        # collate_pcd_fn handles batching of the point clouds
        item = {
            "action_pcd": action_pcd,
            "anchor_pcd": anchor_pcd,
            "anchor_feat_pcd": anchor_feat_pcd,
            "caption": caption,
            "text_embed": text_embed,
            "cross_displacement": end_pcd - action_pcd,
            "intrinsics": self.K,
            "rgbs": rgbs,
            "depths": depths,
            "start2end": start2end,
            "vid_name": image_path,
            "pcd_mean": action_pcd_mean,
            "pcd_std": scene_pcd_std,
            "augment_R": augment_tf["R"],
            "augment_t": augment_tf["t"],
            "augment_C": augment_tf["C"],
        }
        return item


class GenGoalGenDataModule(BaseDataModule):
    def setup(self, stage: str = "fit"):
        self.stage = stage
        self.test_datasets = {}
        self.test_datasets["test_gen"] = GenGoalGenDataset(
            self.root, self.dataset_cfg, "test"
        )
