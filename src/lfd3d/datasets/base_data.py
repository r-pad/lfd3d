import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torchdatasets as td
from pytorch3d.ops import sample_farthest_points
from torch.utils import data
from torch_cluster import grid_cluster
from torch_scatter import scatter_mean
from torchvision import transforms

from lfd3d.utils.data_utils import collate_pcd_fn


class BaseDataset(td.Dataset):
    def __init__(self):
        super().__init__()
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

    def add_gaussian_noise(self, points, noise_magnitude=0.01):
        # points: (N, 3) array
        noise = np.random.normal(0, noise_magnitude, points.shape)
        return points + noise

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

        scene_pcd = torch.from_numpy(points[None])  # (1, N, 3)

        if (
            self.split == "train"
            and self.dataset_cfg.augment_train
            and random.random() < self.dataset_cfg.augment_cfg.augment_prob
        ):
            scene_pcd, scene_feat_pcd, augment_tf = self.augment_scene_pcd(
                scene_pcd, feat_flat, self.dataset_cfg.augment_cfg
            )
        else:  # Just FPS
            scene_pcd, scene_feat_pcd = self.get_fps_pcd(
                scene_pcd, feat_flat, num_points
            )
            augment_tf = {"R": np.eye(3), "t": np.zeros(3), "C": scene_pcd.mean(0)}

        return scene_pcd, scene_feat_pcd, augment_tf

    def augment_scene_pcd(self, scene_pcd, feat_flat, augment_cfg):
        """
        Augment the scene point cloud by applying a random sampling method (e.g., FPS or voxel)
        Also return a random SO(2) rotation around Y-axis and translation for downstream transform

        Args:
            scene_pcd (torch.Tensor): Input point cloud of shape (1, N, 3).
            feat_flat (np.ndarray): Flattened features of shape (M, feat_dim).
            augment_cfg (dict): Configuration for augmentation, including 'pcd_sample', 'fps_num_points', and 'voxel_size'.

        Returns:
            tuple: (scene_pcd, scene_feat_pcd) where:
                - scene_pcd (np.ndarray): Augmented point cloud of shape (num_points, 3).
                - scene_feat_pcd (np.ndarray): Corresponding features of shape (num_points, feat_dim).
        """

        pcd_methods = augment_cfg["pcd_sample"]
        selected_method = random.choice(pcd_methods)
        if selected_method == "fps":
            fps_num_points_options = augment_cfg["fps_num_points"]
            num_points = random.randint(
                fps_num_points_options[0], fps_num_points_options[1]
            )
            scene_pcd, scene_feat_pcd = self.get_fps_pcd(
                scene_pcd, feat_flat, num_points
            )
        elif selected_method == "voxel":
            voxel_sizes = augment_cfg["voxel_size"]
            voxel_size = random.uniform(voxel_sizes[0], voxel_sizes[1])
            size_tensor = torch.tensor([voxel_size] * 3)
            pos = scene_pcd.squeeze(0)  # (N, 3)
            indices = grid_cluster(pos, size=size_tensor)  # Subsampled indices
            _, cluster_ids = torch.unique(indices, return_inverse=True)
            scene_pcd = scatter_mean(pos, cluster_ids, dim=0).numpy()
            scene_feat_pcd = scatter_mean(
                torch.from_numpy(feat_flat), cluster_ids, dim=0
            ).numpy()
        else:
            raise NotImplementedError

        if augment_cfg["augment_transform"]:
            # Apply random SO(2) rotation (around Y-axis as pcd is in camera frame) and translation
            theta = np.random.uniform(0, 2 * np.pi)  # Random angle
            R = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
            )  # Rotation matrix around Y-axis
            translation = np.random.uniform(
                -0.2, 0.2, size=3
            )  # Random translation vector
        else:
            R, translation = np.eye(3), np.zeros(3)

        centroid = scene_pcd.mean(0)

        # Add some gaussian noise
        scene_pcd = self.add_gaussian_noise(scene_pcd)

        return scene_pcd, scene_feat_pcd, {"R": R, "t": translation, "C": centroid}

    def get_fps_pcd(self, scene_pcd, feat_flat, num_points):
        scene_pcd_downsample, scene_points_idx = sample_farthest_points(
            scene_pcd, K=num_points, random_start_point=False
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

    def transform_pcds(
        self,
        start_tracks,
        end_tracks,
        start_scene_pcd,
        action_pcd_mean,
        scene_pcd_std,
        augment_tf,
    ):
        """
        Apply augmentation and normalization to tracks and to scene PCD.

        Args:
            start_tracks (np.ndarray): Initial action PCD (N, 3).
            end_tracks (np.ndarray): Goal action PCD (N, 3).
            start_scene_pcd (np.ndarray): Augmented scene PCD (M, 3).
            action_pcd_mean (np.ndarray): Mean of action PCD (3,).
            scene_pcd_std (np.ndarray): Standard deviation of scene PCD (3,).
            augment_tf (dict): Augmentation parameters {'R': rotation matrix (3x3), 't': translation vector (3,)}.

        Returns:
            tuple: (normalized_start_tracks, normalized_end_tracks, normalized_start_scene_pcd, cross_displacement)
                - normalized_start_tracks (np.ndarray): Normalized initial action PCD.
                - normalized_end_tracks (np.ndarray): Normalized goal action PCD (for reference, though cross_displacement is used).
                - normalized_start_scene_pcd (np.ndarray): Normalized scene PCD.
        """
        R = augment_tf["R"]  # Rotation matrix (3x3)
        t = augment_tf["t"]  # Translation vector (3,)
        scene_centroid = augment_tf["C"]

        # Apply augmentation
        # If augmentation is disabled or for val/test set. augment_tf will have R=I and t=0
        start_tracks_aug = np.dot(start_tracks - scene_centroid, R) + scene_centroid + t
        end_tracks_aug = np.dot(end_tracks - scene_centroid, R) + scene_centroid + t
        start_scene_pcd = (
            np.dot(start_scene_pcd - scene_centroid, R) + scene_centroid + t
        )

        # Normalize: Center on action_pcd_mean and scale by scene_pcd_std
        normalized_start_tracks = (start_tracks_aug - action_pcd_mean) / scene_pcd_std
        normalized_end_tracks = (end_tracks_aug - action_pcd_mean) / scene_pcd_std
        normalized_start_scene_pcd = (start_scene_pcd - action_pcd_mean) / scene_pcd_std

        return (
            normalized_start_tracks,
            normalized_end_tracks,
            normalized_start_scene_pcd,
        )

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
        try:
            data_dir = os.path.expanduser(dataset_cfg.data_dir)
            self.root = data_dir
        except:
            print("data_dir not set.")
            self.root = None

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
