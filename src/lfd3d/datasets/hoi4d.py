import json
import os
from glob import glob

import cv2
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
import torch.utils.data as data

from lfd3d.utils.data_utils import collate_pcd_fn


class HOI4DDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_dir = self.root

        # SpatialTracker could not track the points in these files.
        # or no gt depth available at some tracks
        self.blacklisted_files = set(
            [
                f"{self.dataset_dir}/ZY20210800002/H2/C18/N26/S183/s01/T2/align_rgb/image.mp4",
                f"{self.dataset_dir}/ZY20210800004/H4/C18/N19/S160/s05/T1/align_rgb/image.mp4",
                f"{self.dataset_dir}/ZY20210800004/H4/C18/N26/S162/s02/T1/align_rgb/image.mp4",
                f"{self.dataset_dir}/ZY20210800003/H3/C13/N50/S203/s02/T4/align_rgb/image.mp4",
                f"{self.dataset_dir}/ZY20210800003/H3/C2/N35/S217/s04/T2/align_rgb/image.mp4",
                f"{self.dataset_dir}/ZY20210800003/H3/C2/N36/S217/s04/T2/align_rgb/image.mp4",
                f"{self.dataset_dir}/ZY20210800004/H4/C18/N22/S162/s01/T1/align_rgb/image.mp4",
                f"{self.dataset_dir}/ZY20210800004/H4/C18/N40/S162/s02/T1/align_rgb/image.mp4",
            ]
        )

        self.data_files = sorted(
            glob(f"{self.dataset_dir}/**/image.mp4", recursive=True)
        )
        # Events where there is meaningful object motion
        self.valid_event_types = [
            "Pickup",
            "close",
            "dump",
            "open",
            "pull",
            "push",
            "putdown",
        ]

        split_files = self.load_split(split)
        # Keep only the files that are in the requested split
        self.data_files = (
            set(self.data_files).intersection(set(split_files)) - self.blacklisted_files
        )
        self.data_files = sorted(list(self.data_files))
        self.data_files = self.expand_all_events(self.data_files)

        self.num_demos = len(self.data_files)
        self.dataset_cfg = dataset_cfg

        self.size = self.num_demos
        # Downsample factor for scene point cloud
        self.DOWNSAMPLE_FACTOR = 1000

    def __len__(self):
        return self.size

    def expand_all_events(self, data_files):
        """This function *expands* each file to have an associated event_idx.

        Args:
            data_files (list of str): List of file paths to the data files
                  to be processed.

        Returns:
            expanded_data_files (list of tuples (str, int)): A list of tuples
                  where each tuple contains the file path and the corresponding
                  event index (int) for valid events.
        """
        expanded_data_files = []
        for f in data_files:
            dir_name = os.path.dirname(os.path.dirname(f))
            action_annotation = json.load(open(f"{dir_name}/action/color.json"))
            if "events" in action_annotation:
                all_events = action_annotation["events"]
            else:
                all_events = action_annotation["markResult"]["marks"]
            valid_events = [
                i for i in all_events if i["event"] in self.valid_event_types
            ]
            expanded_data_files.extend([(f, idx) for idx in range(len(valid_events))])
        return expanded_data_files

    def load_split(self, split):
        """
        Load the filenames corresponding to each split - [train, val, test]
        The file containing the splits `metadata.json` is expected to be
        placed *outside* the directory. The splits were generated using the code
        in the General Flow codebase
        """
        with open(f"{self.dataset_dir}/../metadata.json", "r") as f:
            all_splits = json.load(f)
        split_data = all_splits[split]

        split_data_fnames = list(set([i["index"] for i in split_data]))
        split_data_fnames = [
            f"{self.dataset_dir}/{i}/align_rgb/image.mp4" for i in split_data_fnames
        ]
        return split_data_fnames

    def load_camera_params(self, dir_name):
        cam_trajectory = o3d.io.read_pinhole_camera_trajectory(
            f"{dir_name}/3Dseg/output.log"
        )
        # A bit hacky.
        # Camera intrinsics are expected to be placed one level above hoi4d dir
        # with the directory name "camera_params"
        cam_name = dir_name.split("/")[-7]
        cam_param_pathname = f"{self.dataset_dir}/../camera_params/{cam_name}/"
        K = np.load(f"{cam_param_pathname}/intrin.npy")
        cam2world = np.array([i.extrinsic for i in cam_trajectory.parameters])
        return K, cam2world

    def load_event(self, dir_name, event_idx):
        action_annotation = json.load(open(f"{dir_name}/action/color.json"))
        # The dataset does not have a consistent naming scheme .......
        try:
            # 300 frames per video, 30 or 15 fps depending on length of video
            fps = 300 / action_annotation["info"]["duration"]
            # Filter out useful events
            all_events = action_annotation["events"]
        except KeyError:
            fps = 300 / action_annotation["info"]["Duration"]
            all_events = action_annotation["markResult"]["marks"]

        valid_events = [i for i in all_events if i["event"] in self.valid_event_types]
        event = valid_events[event_idx]

        try:
            # Convert timestamp in seconds to frame_idx
            event_start_idx = int(event["startTime"] * fps)
            event_end_idx = int(event["endTime"] * fps) - 1
        except KeyError:
            event_start_idx = int(event["hdTimeStart"] * fps)
            event_end_idx = int(event["hdTimeEnd"] * fps) - 1
        return event, event_start_idx, event_end_idx

    def load_rgbd(self, dir_name, event_start_idx, event_end_idx):
        # Return rgb/depth at beginning and end of event
        rgb_init = cv2.cvtColor(
            cv2.imread(f"{dir_name}/align_rgb/{str(event_start_idx).zfill(5)}.jpg"),
            cv2.COLOR_BGR2RGB,
        )
        rgb_end = cv2.cvtColor(
            cv2.imread(f"{dir_name}/align_rgb/{str(event_end_idx).zfill(5)}.jpg"),
            cv2.COLOR_BGR2RGB,
        )
        rgbs = np.array([rgb_init, rgb_end])

        depth_init = cv2.imread(
            f"{dir_name}/align_depth/{str(event_start_idx).zfill(5)}.png", -1
        )
        depth_init = depth_init / 1000.0  # Convert to metres
        depth_end = cv2.imread(
            f"{dir_name}/align_depth/{str(event_end_idx).zfill(5)}.png", -1
        )
        depth_end = depth_end / 1000.0  # Convert to metres
        depths = np.array([depth_init, depth_end])
        return rgbs, depths

    def process_and_register_tracks(
        self, dir_name, event_start_idx, event_end_idx, depths, K, cam2world
    ):
        """
        Load the tracks corresponding to the start and end of the event.
        Unproject the tracks to 3D point clouds, register the point cloud
        `end_tracks` to the `start_tracks` coordinate frame.
        """
        tracks = np.load(f"{dir_name}/spatracker_3d_tracks.npy")
        start_tracks = tracks[event_start_idx]

        # Clip to image boundaries, unproject
        ty = np.clip(start_tracks[:, 1].round().astype(int), 0, depths[0].shape[0] - 1)
        tx = np.clip(start_tracks[:, 0].round().astype(int), 0, depths[0].shape[1] - 1)
        start_tracks[:, 2] = depths[0][ty, tx]
        start_tracks[:, 0] = ((start_tracks[:, 0] - K[0, 2]) * start_tracks[:, 2]) / K[
            0, 0
        ]
        start_tracks[:, 1] = ((start_tracks[:, 1] - K[1, 2]) * start_tracks[:, 2]) / K[
            1, 1
        ]

        end_tracks = tracks[event_end_idx]

        # Clip to image boundaries, unproject
        ty = np.clip(end_tracks[:, 1].round().astype(int), 0, depths[0].shape[0] - 1)
        tx = np.clip(end_tracks[:, 0].round().astype(int), 0, depths[0].shape[1] - 1)
        end_tracks[:, 2] = depths[1][ty, tx]
        end_tracks[:, 0] = ((end_tracks[:, 0] - K[0, 2]) * end_tracks[:, 2]) / K[0, 0]
        end_tracks[:, 1] = ((end_tracks[:, 1] - K[1, 2]) * end_tracks[:, 2]) / K[1, 1]

        # Register the tracks at the end of the chunk with respect
        # to the coordinate frame at the beginning of the chunk
        start2world = cam2world[event_start_idx]
        end2world = cam2world[event_end_idx]
        end_tracks = np.hstack((end_tracks, np.ones((end_tracks.shape[0], 1))))
        start2end = start2world @ np.linalg.inv(end2world)
        end_tracks = (start2end @ end_tracks.T).T[:, :3].astype(np.float32)

        return start_tracks, end_tracks, start2end

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
        valid_depth = z_flat > 0
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

        scene_pcd = points[:: self.DOWNSAMPLE_FACTOR]
        return scene_pcd

    def __getitem__(self, index):
        vid_name, event_idx = self.data_files[index]
        dir_name = os.path.dirname(os.path.dirname(vid_name))

        K, cam2world = self.load_camera_params(dir_name)

        # Get the object name from the pose file.
        # The dataset does not have a consistent naming scheme .......
        objpose_fname = f"{dir_name}/objpose/0.json"
        if not os.path.exists(objpose_fname):
            objpose_fname = f"{dir_name}/objpose/00000.json"
        obj_name = json.load(open(objpose_fname))["dataList"][0]["label"]

        event, event_start_idx, event_end_idx = self.load_event(dir_name, event_idx)
        event_name = event["event"]

        rgbs, depths = self.load_rgbd(dir_name, event_start_idx, event_end_idx)
        start_tracks, end_tracks, start2end = self.process_and_register_tracks(
            dir_name, event_start_idx, event_end_idx, depths, K, cam2world
        )
        start_scene_pcd = self.get_scene_pcd(rgbs[0], depths[0], K)
        caption = f"{event_name} {obj_name}"

        # Center on action_pcd
        action_pcd_mean = start_tracks.mean(axis=0)
        start_tracks -= action_pcd_mean
        end_tracks -= action_pcd_mean
        start_scene_pcd -= action_pcd_mean

        # collate_pcd_fn handles batching of the point clouds
        item = {
            "action_pcd": start_tracks,
            "anchor_pcd": start_scene_pcd,
            "caption": caption,
            "cross_displacement": end_tracks - start_tracks,
            "intrinsics": K,
            "rgbs": rgbs,
            "depths": depths,
            "start2end": start2end,
            "vid_name": dir_name,
            "action_pcd_mean": action_pcd_mean,
        }
        return item


class HOI4DDataModule(pl.LightningDataModule):
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

        self.train_dataset = HOI4DDataset(self.root, self.dataset_cfg, "train")
        self.val_dataset = HOI4DDataset(self.root, self.dataset_cfg, "val")
        self.test_dataset = HOI4DDataset(self.root, self.dataset_cfg, "test")

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
        indices = torch.randint(
            0, len(self.train_dataset), (self.TRAIN_SUBSET_SIZE,)
        ).tolist()
        return data.DataLoader(
            data.Subset(self.train_dataset, indices),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pcd_fn,
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
