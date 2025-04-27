import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchdatasets as td
from pytorch3d.ops import sample_farthest_points
from torch.utils import data

from lfd3d.utils.data_utils import collate_pcd_fn


class BaseDataset(td.Dataset):
    def get_scene_pcd(self, rgb_embed, depth, K, num_points, max_depth):
        """
        Generate a downsampled point cloud (PCD) from RGB embeddings and depth map.

        Args:
            rgb_embed (np.ndarray): RGB feature embeddings of shape (H, W, feat_dim).
            depth (np.ndarray): Depth map of shape (H, W).
            K (np.ndarray): Camera intrinsics matrix (3x3).
            num_points (int): Number of points to sample from the PCD.
            max_depth (float): Maximum depth value for valid points.

        Returns:
            tuple: (scene_pcd, scene_feat_pcd) where:
                - scene_pcd (np.ndarray): Downsampled 3D points of shape (num_points, 3).
                - scene_feat_pcd (np.ndarray): Features for downsampled points of shape (num_points, feat_dim).
        """
        height, width = depth.shape
        # Create pixel coordinate grid
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
        x_flat, y_flat, z_flat = x_grid.flatten(), y_grid.flatten(), depth.flatten()
        feat_flat = rgb_embed.reshape(-1, rgb_embed.shape[-1])

        # Remove points with invalid depth
        valid_depth = np.logical_and(z_flat > 0, z_flat < max_depth)
        x_flat, y_flat, z_flat, feat_flat = (
            arr[valid_depth] for arr in (x_flat, y_flat, z_flat, feat_flat)
        )

        # Unproject points using K inverse
        pixels = np.stack([x_flat, y_flat, np.ones_like(x_flat)], axis=0)
        K_inv = np.linalg.inv(K)
        points = (K_inv @ pixels) * z_flat  # Shape: (3, N)
        points = points.T  # Shape: (N, 3)

        scene_pcd_pt3d = torch.from_numpy(points[None])  # (1, N, 3)
        scene_pcd_downsample, scene_points_idx = sample_farthest_points(
            scene_pcd_pt3d, K=num_points, random_start_point=False
        )
        scene_pcd = scene_pcd_downsample.squeeze().numpy()  # (num_points, 3)
        scene_feat_pcd = feat_flat[
            scene_points_idx.squeeze().numpy()
        ]  # (num_points, feat_dim)
        return scene_pcd, scene_feat_pcd

    def get_normalize_mean_std(self, action_pcd, scene_pcd, dataset_cfg):
        """
        Compute mean and standard deviation for normalization based on action and scene PCDs.

        Args:
            action_pcd (np.ndarray): Action point cloud of shape (N, 3).
            scene_pcd (np.ndarray): Scene point cloud of shape (M, 3).
            dataset_cfg (dict): Dataset configuration; checks for 'normalize' flag.

        Returns:
            tuple: (action_pcd_mean, scene_pcd_std) where:
                - action_pcd_mean (np.ndarray): Mean vector of shape (3,).
                - scene_pcd_std (np.ndarray): Standard deviation vector of shape (3,).
        """
        if not dataset_cfg.get("normalize", True):
            return np.zeros(3), np.ones(3)
        return action_pcd.mean(axis=0), scene_pcd.std(axis=0)

    def get_scaled_intrinsics(self, K, orig_shape, target_shape=224):
        """
        Scale camera intrinsics matrix based on image resizing and cropping.

        Args:
            K (np.ndarray): Original camera intrinsics matrix (3x3).
            orig_shape (tuple): Original image shape (height, width).
            target_shape (int): Target size for resized images (default: 224).

        Returns:
            np.ndarray: Scaled intrinsics matrix (3x3).
        """
        # Getting scale factor from torchvision.transforms.Resize behaviour
        K_ = K.copy()
        scale_factor = target_shape / min(orig_shape)

        # Apply the scale factor to the intrinsics
        K_[[0, 1], [0, 1]] *= scale_factor  # fx, fy
        K_[[0, 1], 2] *= scale_factor  # cx, cy

        # Adjust the principal point (cx, cy) for the center crop
        crop_offset_x = (orig_shape[1] * scale_factor - target_shape) / 2
        crop_offset_y = (orig_shape[0] * scale_factor - target_shape) / 2
        K_[0, 2] -= crop_offset_x
        K_[1, 2] -= crop_offset_y
        return K_


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg, seed):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.stage = None
        self.dataset_cfg = dataset_cfg
        self.seed = seed

        # setting root directory based on dataset type
        data_dir = os.path.expanduser(dataset_cfg.data_dir)
        self.root = data_dir

        # Subset of train to use for eval
        self.TRAIN_SUBSET_SIZE = 500

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "fit"):
        raise NotImplementedError("Not implemented for baseclass")

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            raise AttributeError(
                "train_dataset has not been set. Make sure to call setup() first."
            )
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if self.stage == "fit" else False,
            num_workers=self.num_workers,
            collate_fn=collate_pcd_fn,
        )

    def train_subset_dataloader(self):
        """A subset of train used for eval."""
        if not hasattr(self, "train_dataset"):
            raise AttributeError(
                "train_dataset has not been set. Make sure to call setup() first."
            )
        indices = torch.randint(
            0, len(self.train_dataset), (self.TRAIN_SUBSET_SIZE,)
        ).tolist()
        return data.DataLoader(
            data.Subset(self.train_dataset, indices),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pcd_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_datasets"):
            raise AttributeError(
                "val_datasets has not been set. Make sure to call setup() first."
            )
        return {
            tag: data.DataLoader(
                dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_pcd_fn,
            )
            for tag, dataset in self.val_datasets.items()
        }

    def test_dataloader(self):
        if not hasattr(self, "test_datasets"):
            raise AttributeError(
                "test_datasets has not been set. Make sure to call setup() first."
            )
        return {
            tag: data.DataLoader(
                dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_pcd_fn,
            )
            for tag, dataset in self.test_datasets.items()
        }
