import os
from glob import glob

import cv2
import numpy as np
from torch.utils import data

from lfd3d.datasets.base_data import BaseDataModule


class SynthBlockDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_dir = self.root

        self.data_files = sorted(
            glob(f"{self.dataset_dir}/**/action_pcd.npy", recursive=True)
        )

        if self.split == "train":
            self.data_files = self.data_files[: int(0.8 * len(self.data_files))]
        if self.split == "val":
            self.data_files = self.data_files[
                int(0.8 * len(self.data_files)) : int(0.9 * len(self.data_files))
            ]
        if self.split == "test":
            self.data_files = self.data_files[int(0.9 * len(self.data_files)) :]

        self.dataset_cfg = dataset_cfg
        self.size = len(self.data_files)

        self.K = np.load(f"{self.dataset_dir}/intrinsics.npy")

    def __len__(self):
        return self.size

    def load_rgbd(self, dir_name):
        # Return rgb/depth at beginning and end of event
        rgb_init = cv2.cvtColor(
            cv2.imread(f"{dir_name}/rgb_0.png"),
            cv2.COLOR_BGR2RGB,
        )
        rgb_end = cv2.cvtColor(
            cv2.imread(f"{dir_name}/rgb_1.png"),
            cv2.COLOR_BGR2RGB,
        )
        rgbs = np.array([rgb_init, rgb_end])

        depth_init = np.load(f"{dir_name}/depth_0.npy")
        depth_end = np.load(f"{dir_name}/depth_1.npy")
        depths = np.array([depth_init, depth_end])
        return rgbs, depths

    def get_normalize_mean_std(self, action_pcd, scene_pcd):
        if self.dataset_cfg.normalize is False:
            mean, std = 0, np.array([1.0, 1.0, 1.0])
        else:
            mean, std = action_pcd.mean(axis=0), scene_pcd.std(axis=0)
        return mean, std

    def __getitem__(self, index):
        raise NotImplementedError(
            "switch to gripper-only prediction + dino features. not yet implemented for this dataset."
        )
        pcd_name = self.data_files[index]
        dir_name = os.path.dirname(pcd_name)

        action_pcd = np.load(f"{dir_name}/action_pcd.npy")
        anchor_pcd = np.load(f"{dir_name}/pcd_0.npy")
        caption = "putdown block"

        cross_displacement = np.load(f"{dir_name}/cross_displacement.npy")
        rgbs, depths = self.load_rgbd(dir_name)

        # Scale down values to 1/10m (else breaks visualization)
        action_pcd = action_pcd / 10
        cross_displacement = cross_displacement / 10
        anchor_pcd = anchor_pcd / 10
        depths = depths / 10

        start2end = np.eye(4)

        action_pcd_mean, scene_pcd_std = self.get_normalize_mean_std(
            action_pcd, anchor_pcd
        )
        # Center on action_pcd
        action_pcd = action_pcd - action_pcd_mean
        anchor_pcd = anchor_pcd - action_pcd_mean
        # Standardize on scene_pcd
        action_pcd = action_pcd / scene_pcd_std
        anchor_pcd = anchor_pcd / scene_pcd_std
        cross_displacement = cross_displacement / scene_pcd_std

        action_pcd = action_pcd[::10]
        cross_displacement = cross_displacement[::10]
        anchor_pcd = anchor_pcd[::100]

        # collate_pcd_fn handles batching of the point clouds
        item = {
            "action_pcd": action_pcd,
            "anchor_pcd": anchor_pcd,
            "caption": caption,
            "cross_displacement": cross_displacement,
            "intrinsics": self.K,
            "rgbs": rgbs,
            "depths": depths,
            "start2end": start2end,
            "vid_name": dir_name,
            "pcd_mean": action_pcd_mean,
            "pcd_std": scene_pcd_std,
        }
        return item


class SynthBlockDataModule(BaseDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):
        super().__init__(batch_size, val_batch_size, num_workers, dataset_cfg)
        self.val_tags = ["block"]

    def setup(self, stage: str = "fit"):
        self.stage = stage
        self.val_datasets = {}
        self.test_datasets = {}

        self.train_dataset = SynthBlockDataset(self.root, self.dataset_cfg, "train")
        for tag in self.val_tags:
            self.val_datasets[tag] = SynthBlockDataset(
                self.root, self.dataset_cfg, "val"
            )
            self.test_datasets[tag] = SynthBlockDataset(
                self.root, self.dataset_cfg, "test"
            )
