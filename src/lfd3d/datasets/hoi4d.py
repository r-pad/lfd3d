import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import torch.utils.data as data
import torchvision as tv
from torchvision import transforms as T

import rpad.visualize_3d.plots as vpl

import numpy as np
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
import torch_geometric.transforms as tgt

from pathlib import Path
import os
from pytorch3d.transforms import Transform3d, Translate

from non_rigid.utils.transform_utils import random_se3
from non_rigid.utils.pointcloud_utils import downsample_pcd
from non_rigid.utils.augmentation_utils import ball_occlusion, plane_occlusion, maybe_apply_augmentations
from glob import glob
import cv2
import json
import random
import torch.nn.functional as F

class HOI4DDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_dir = self.root
        self.data_files = sorted(glob(f"{self.dataset_dir}/**/image.mp4", recursive=True))
        self.data_files = self.data_files[:16]
        self.num_demos = len(self.data_files)
        print(self.num_demos)
        self.dataset_cfg = dataset_cfg

        self.size = self.num_demos

        # setting sample sizes
        self.scene = self.dataset_cfg.scene
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        self.world_frame = self.dataset_cfg.world_frame
        self.PAD_SIZE = 300

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        vid_name = self.data_files[index]
        dir_name = os.path.dirname(os.path.dirname(vid_name))

        # rgb = np.array([cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in sorted(glob(f"{dir_name}/align_rgb/*jpg"))])
        # depth = np.array([cv2.imread(i, -1) for i in sorted(glob(f"{dir_name}/align_depth/*png"))])
        # depth = depth / 1000. # Conver to metres

        tracks = np.load(f"{dir_name}/spatracker_3d_tracks.npy")
        tracks[:,:,0] /= tracks[:,:,0].max()
        tracks[:,:,1] /= tracks[:,:,1].max()

        obj_name = json.load(open(f"{dir_name}/objpose/00000.json"))["dataList"][0]["label"]

        action_annotation = json.load(open(f"{dir_name}/action/color.json"))
        event = random.choice(action_annotation["events"])
        event_start_idx = int(event["startTime"]*30)
        event_end_idx = int(event["endTime"]*30) - 1
        event_name = event["event"]

        caption = f"{event_name} {obj_name}"
        item = {}
        item["start_pcd"] = tracks[event_start_idx]
        # Pad points on the `left` to have common size for batching
        item["start_pcd"] = np.pad(item["start_pcd"], ((self.PAD_SIZE - item["start_pcd"].shape[0], 0),
                                                       (0,0)))
        item["caption"] = caption
        # item["rgb"] = rgb[event_start_idx]
        # item["depth"] = depth[event_start_idx]
        item["cross_displacement"] = tracks[event_end_idx] - tracks[event_start_idx]
        item["cross_displacement"] = np.pad(item["cross_displacement"], ((self.PAD_SIZE - item["cross_displacement"].shape[0], 0),
                                                                         (0,0)))
        return item


class HOI4DDataModule(L.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):
        super().__init__()
        # self.root = root
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.stage = None
        self.dataset_cfg = dataset_cfg

        # setting root directory based on dataset type
        data_dir = os.path.expanduser(dataset_cfg.data_dir)
        self.root = data_dir

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "fit"):
        self.stage = stage

        self.train_dataset = HOI4DDataset(
            self.root, self.dataset_cfg, "traintax3d"
        )
        self.val_dataset = HOI4DDataset(
            self.root, self.dataset_cfg, "val_tax3d"
        )
        self.val_ood_dataset = HOI4DDataset(
            self.root, self.dataset_cfg, "val_ood_tax3d"
        )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if self.stage == "train" else False,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        val_ood_dataloader = data.DataLoader(
            self.val_ood_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return val_dataloader, val_ood_dataloader
