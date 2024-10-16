import json
import os
import random
from glob import glob

import lightning as L
import numpy as np
import open3d as o3d
import torch.utils.data as data


class HOI4DDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_dir = self.root
        self.data_files = sorted(
            glob(f"{self.dataset_dir}/**/image.mp4", recursive=True)
        )
        self.num_demos = len(self.data_files)
        self.dataset_cfg = dataset_cfg

        self.size = self.num_demos
        self.PAD_SIZE = 1000
        self.fps = 15
        # Events where there is meaningfully described object motion
        self.valid_event_types = [
            "Pickup",
            "close",
            "dump",
            "open",
            "pull",
            "push",
            "putdown",
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        vid_name = self.data_files[index]
        dir_name = os.path.dirname(os.path.dirname(vid_name))

        cam_trajectory = o3d.io.read_pinhole_camera_trajectory(
            f"{dir_name}/3Dseg/output.log"
        )
        K = cam_trajectory.parameters[0].intrinsic.intrinsic_matrix

        # rgb = np.array([cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in sorted(glob(f"{dir_name}/align_rgb/*jpg"))])
        # depth = np.array([cv2.imread(i, -1) for i in sorted(glob(f"{dir_name}/align_depth/*png"))])
        # depth = depth / 1000. # Conver to metres

        tracks = np.load(f"{dir_name}/spatracker_3d_tracks.npy")
        # Pad points on the "left" to have common size for batching
        tracks = np.pad(tracks, ((0, 0), (self.PAD_SIZE - tracks.shape[1], 0), (0, 0)))
        # A few videos don't have any valid tracks, a check to avoid div by 0.
        if tracks.max() != 0:
            # SpatialTracker tracks u,v in image plane and z in 3d.
            # Unproject u,v to x,y
            tracks[:, :, 0] = ((tracks[:, :, 0] - K[0, 2]) * tracks[:, :, 2]) / K[0, 0]
            tracks[:, :, 1] = ((tracks[:, :, 1] - K[1, 2]) * tracks[:, :, 2]) / K[1, 1]

        # Get the object name from the pose file.
        # The dataset does not have a consistent naming scheme .......
        objpose_fname = f"{dir_name}/objpose/0.json"
        if not os.path.exists(objpose_fname):
            objpose_fname = f"{dir_name}/objpose/00000.json"
        obj_name = json.load(open(objpose_fname))["dataList"][0]["label"]

        action_annotation = json.load(open(f"{dir_name}/action/color.json"))
        # The dataset does not have a consistent naming scheme .......
        try:
            # Filter out useful events
            all_events = action_annotation["events"]
            valid_events = [
                i for i in all_events if i["event"] in self.valid_event_types
            ]
            event = random.choice(valid_events)
            # Convert timestamp in seconds to frame_idx
            event_start_idx = int(event["startTime"] * self.fps)
            event_end_idx = int(event["endTime"] * self.fps) - 1
        except KeyError:
            all_events = action_annotation["markResult"]["marks"]
            valid_events = [
                i for i in all_events if i["event"] in self.valid_event_types
            ]
            event = random.choice(valid_events)
            event_start_idx = int(event["hdTimeStart"] * self.fps)
            event_end_idx = int(event["hdTimeEnd"] * self.fps) - 1
        event_name = event["event"]

        caption = f"{event_name} {obj_name}"
        item = {}
        item["start_pcd"] = tracks[event_start_idx]
        item["caption"] = caption
        # item["rgb"] = rgb[event_start_idx]
        # item["depth"] = depth[event_start_idx]
        item["cross_displacement"] = tracks[event_end_idx] - tracks[event_start_idx]
        return item


class HOI4DDataModule(L.LightningDataModule):
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

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "fit"):
        self.stage = stage

        self.train_dataset = HOI4DDataset(self.root, self.dataset_cfg, "train")
        self.val_dataset = HOI4DDataset(self.root, self.dataset_cfg, "val")
        self.test_dataset = HOI4DDataset(self.root, self.dataset_cfg, "test")

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
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = data.DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_dataloader
