import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchdatasets as td

from lfd3d.utils.data_utils import collate_pcd_fn


class RT1Dataset(td.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split

        self.cache_dir = dataset_cfg.cache_dir

        self.depth_dir = f"{root}/rt1_depth"  # Depth estimated with RollingDepth is expected to be here.
        self.track_dir = f"{root}/rt1_tracks"  # Tracks estimated with Co-Tracker are expected to be here.
        self.rgb_dir = (
            f"{root}/rt1_rgb_chunk"  # RGB images of events are expected to be here.
        )
        self.dataset_cfg = dataset_cfg
        with open(f"{root}/chunked_captions.json") as f:
            self.captions = json.load(f)

        # No camera intrinsics available, an arbitrary choice based on RT-1 resolution (256, 320)
        self.K = np.array([[257.0, 0, 160], [0, 257.0, 128], [0, 0, 1]])
        # Voxel size for downsampling
        self.voxel_size = 0.03

        self.rt1_index = self.load_split(split)
        self.rt1_index = self.expand_all_events(self.rt1_index)

        self.size = len(self.rt1_index)

    def __len__(self):
        return self.size

    def load_split(self, split):
        """
        Load the filenames corresponding to each split - [train, val, test]
        Test splits are from 3D-VLA. Train/val were manually generated.
        """
        current_dir = os.path.dirname(__file__)
        with open(f"{current_dir}/fractal20220817_data_{split}.txt", "r") as f:
            split_idxs = f.readlines()
        split_idxs = [int(i) for i in split_idxs]
        return split_idxs

    def expand_all_events(self, rt1_index):
        """This function *expands* each event to have an associated chunk_idx.
        RT-1 events have either 1/2 chunks depending on the event.
        If no chunks were generated due to bad data/other issues,
        the event is removed from the index.

        Args:
            rt1_index (list of int): List of indexes in RT-1 to be processed.

        Returns:
            expanded_rt_index (list of tuples (int, int)): A list of tuples
                  where each tuple contains the event index and the corresponding
                  chunk index.
        """
        expanded_rt_index = []
        for idx in rt1_index:
            if not os.path.exists(f"{self.rgb_dir}/{idx}"):
                continue

            caption = self.captions[idx]["original"]
            if caption == "" or caption.split()[0] in ["open", "close"]:
                expanded_event = [(idx, 0)]
            else:
                expanded_event = [(idx, 0), (idx, 1)]

            expanded_rt_index.extend(expanded_event)
        return expanded_rt_index

    def load_event_indexes(self, index, chunk_idx):
        """Load the indexes of the images corresponding to the chunk in the specified event."""
        image_indexes = sorted(os.listdir(f"{self.rgb_dir}/{index}"))
        image_indexes = [int(i.split(".")[0]) for i in image_indexes]
        if chunk_idx == 0:
            return image_indexes[:2]
        else:
            return image_indexes[1:]

    def load_rgbd(self, index, event_start_idx, event_end_idx):
        """Load RGB-D frames from the video at specified idxs.
        Depth is estimated using RollingDepth.

        data_steps: - Data of one item in the TFDS
        index: int - Index of event in dataset
        event_start_idx: int - Start index of event
        event_end_idx: int - End index of event
        """
        rgb_start = cv2.cvtColor(
            cv2.imread(f"{self.rgb_dir}/{index}/{str(event_start_idx).zfill(5)}.png"),
            cv2.COLOR_BGR2RGB,
        )
        rgb_end = cv2.cvtColor(
            cv2.imread(f"{self.rgb_dir}/{index}/{str(event_end_idx).zfill(5)}.png"),
            cv2.COLOR_BGR2RGB,
        )
        rgbs = np.array([rgb_start, rgb_end])

        depth_vid = np.load(f"{self.depth_dir}/{index}_pred.npz")["arr_0"]
        depths = depth_vid[[event_start_idx, event_end_idx]]

        # RollingDepth estimates *inverse* relative depth, thus -ive sign in scale
        # Shifting by 0.5 for viz
        # Doesn't matter for network input since we standardize the pcd anyway
        depths = (depths * -1) + 0.5
        return rgbs, depths

    def load_tracks(self, index, event_start_idx, event_end_idx, depths):
        """Load tracks from the video at specified idxs.
        Tracks are computed using Co-Tracker.

        index: int - Index of event in dataset
        event_start_idx: int - Start index of event
        event_end_idx: int - End index of event
        """
        tracks = np.load(f"{self.track_dir}/cotracker_{index}.npz")["tracks"]
        start_tracks, end_tracks = tracks[event_start_idx], tracks[event_end_idx]

        # Unproject tracks to 3D

        ty = np.clip(start_tracks[:, 1].round().astype(int), 0, depths[0].shape[0] - 1)
        tx = np.clip(start_tracks[:, 0].round().astype(int), 0, depths[0].shape[1] - 1)
        start_track_depth = depths[0][ty, tx]
        start_tracks[:, 0] = (
            (start_tracks[:, 0] - self.K[0, 2]) * start_track_depth
        ) / self.K[0, 0]
        start_tracks[:, 1] = (
            (start_tracks[:, 1] - self.K[1, 2]) * start_track_depth
        ) / self.K[1, 1]
        start_tracks = np.concatenate(
            [start_tracks, start_track_depth[:, None]], axis=-1
        )

        ty = np.clip(end_tracks[:, 1].round().astype(int), 0, depths[0].shape[0] - 1)
        tx = np.clip(end_tracks[:, 0].round().astype(int), 0, depths[0].shape[1] - 1)
        end_track_depth = depths[1][ty, tx]
        end_tracks[:, 0] = (
            (end_tracks[:, 0] - self.K[0, 2]) * end_track_depth
        ) / self.K[0, 0]
        end_tracks[:, 1] = (
            (end_tracks[:, 1] - self.K[1, 2]) * end_track_depth
        ) / self.K[1, 1]
        end_tracks = np.concatenate([end_tracks, end_track_depth[:, None]], axis=-1)

        return start_tracks, end_tracks

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

    def load_rgb_text_feat(self, event_idx, chunk_idx, height, width):
        """
        Load RGB/text features generated with SIGLIP using ConceptFusion.
        """
        features = np.load(
            f"{self.root}/rt1_rgb_feat/{event_idx}_{chunk_idx}_compressed.npz"
        )
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

    def __getitem__(self, idx):
        raise NotImplementedError(
            "switch to gripper-only prediction + dino features. not yet implemented for this dataset."
        )
        index, chunk_idx = self.rt1_index[idx]

        caption = self.captions[index]["chunked"][chunk_idx]
        start2end = torch.eye(4)  # Static camera in RT-1

        event_start_idx, event_end_idx = self.load_event_indexes(index, chunk_idx)
        rgbs, depths = self.load_rgbd(index, event_start_idx, event_end_idx)

        start_tracks, end_tracks = self.load_tracks(
            index, event_start_idx, event_end_idx, depths
        )

        rgb_embed, text_embed = self.load_rgb_text_feat(
            index, chunk_idx, rgbs[0].shape[0], rgbs[0].shape[1]
        )
        start_scene_pcd, start_scene_feat_pcd = self.get_scene_pcd(
            rgb_embed, depths[0], self.K
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
            "intrinsics": self.K,
            "rgbs": rgbs,
            "depths": depths,
            "start2end": start2end,
            "vid_name": index,  # no name in RT-1, just return idx in dataset
            "pcd_mean": action_pcd_mean,
            "pcd_std": scene_pcd_std,
        }
        return item


class RT1DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.stage = None
        self.dataset_cfg = dataset_cfg

        # setting root directory based on dataset type
        data_dir = os.path.expanduser(dataset_cfg.data_dir)
        self.root = data_dir

        # Subset of train to use for eval
        self.TRAIN_SUBSET_SIZE = 500

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "fit"):
        self.stage = stage

        self.train_dataset = RT1Dataset(self.root, self.dataset_cfg, "train")
        if self.train_dataset.cache_dir:
            self.train_dataset.cache(
                td.cachers.Pickle(Path(self.train_dataset.cache_dir))
            )
            self.val_dataset.cache(
                td.cachers.Pickle(Path(self.train_dataset.cache_dir) / "val")
            )
        self.val_dataset = RT1Dataset(self.root, self.dataset_cfg, "val")
        self.test_dataset = RT1Dataset(self.root, self.dataset_cfg, "test")

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if self.stage == "fit" else False,
            num_workers=self.num_workers,
            collate_fn=collate_pcd_fn,
        )

    def train_subset_dataloader(self):
        """A subset of train used for eval."""
        indices = torch.arange(0, self.TRAIN_SUBSET_SIZE).tolist()
        return data.DataLoader(
            data.Subset(self.train_dataset, indices),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pcd_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pcd_fn,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = data.DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pcd_fn,
        )
        return test_dataloader
