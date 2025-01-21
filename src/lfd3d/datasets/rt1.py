import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchdatasets as td

from lfd3d.utils.data_utils import collate_pcd_fn


class RT1Dataset(td.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split

        self.cache_dir = dataset_cfg.cache_dir

        # glob dir and then expand if 2 events
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
        self.voxel_size = 0.05

        # TODO: handle expansion for each sub event
        self.rt1_index = os.listdir(self.rgb_dir)
        self.size = len(self.rt1_index)

    def __len__(self):
        return self.size

    def select_event_chunk(self, index):
        """Many RT-1 events are decomposed into two sub-events.
        For those events, sample one of the two randomly.

        index: int - Index in RT-1 TFDS
        """
        return random.randint(0, len(self.captions[index]["chunked"]) - 1)

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
        # Scaling/shifting by 2 for viz
        # Doesn't matter for network input since we standardize the pcd anyway
        depths = (depths * -2) + 2
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

    def get_scene_pcd(self, rgb, depth, K):
        height, width = depth.shape
        # Create pixel coordinate grid
        x = np.arange(width)
        y = np.arange(height)
        x_grid, y_grid = np.meshgrid(x, y)

        # Flatten grid coordinates and depth
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = depth.flatten()

        # Remove points with invalid depth
        valid_depth = np.logical_and(z_flat > 0, z_flat < 5)
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

        scene_pcd_o3d = o3d.geometry.PointCloud()
        scene_pcd_o3d.points = o3d.utility.Vector3dVector(points)
        scene_pcd_o3d_downsample = scene_pcd_o3d.voxel_down_sample(
            voxel_size=self.voxel_size
        )

        scene_pcd = np.asarray(scene_pcd_o3d_downsample.points)
        return scene_pcd

    def __getitem__(self, idx):
        index = int(self.rt1_index[idx])

        chunk_idx = self.select_event_chunk(index)
        caption = self.captions[index]["chunked"][chunk_idx]

        start2end = torch.eye(4)  # Static camera in RT-1

        event_start_idx, event_end_idx = self.load_event_indexes(index, chunk_idx)
        rgbs, depths = self.load_rgbd(index, event_start_idx, event_end_idx)

        start_tracks, end_tracks = self.load_tracks(
            index, event_start_idx, event_end_idx, depths
        )
        start_scene_pcd = self.get_scene_pcd(rgbs[0], depths[0], self.K)

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
            "caption": caption,
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
