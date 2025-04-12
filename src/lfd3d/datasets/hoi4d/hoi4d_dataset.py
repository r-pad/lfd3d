import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchdatasets as td
from lfd3d.datasets.base_data import BaseDataModule
from lfd3d.utils.data_utils import MANOInterface
from PIL import Image
from pytorch3d.ops import sample_farthest_points
from torchvision import transforms
from tqdm import tqdm


class HOI4DDataset(td.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_dir = self.root

        self.cache_dir = dataset_cfg.cache_dir
        self.gt_source = dataset_cfg.gt_source
        assert self.gt_source in ["spatrack_tracks", "gflow_tracks", "mano_handpose"]

        self.intrinsic_dict = self.load_intrinsics()
        current_dir = os.path.dirname(__file__)
        with open(f"{current_dir}/hoi4d_videos.json") as f:
            self.data_files = json.load(f)
            self.data_files = [f"{self.dataset_dir}/{i}" for i in self.data_files]
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

        self.mano_interface = MANOInterface()

        split_files = self.load_split(split)

        if self.gt_source == "gflow_tracks":
            # Use tracks generated from General Flow's label_gen_event.py in sriramsk1999/general-flow
            with open(
                f"{self.root}/../hoi4d_general_flow_event_traj/metadata.json"
            ) as f:
                self.event_metadata = json.load(f)["test"]
            self.event_metadata_dict = {
                (i["index"], i["img"]): (idx, i)
                for idx, i in enumerate(self.event_metadata)
            }
            with open(
                f"{self.root}/../hoi4d_general_flow_event_traj/kpst_hoi4d_test_traj.pkl",
                "rb",
            ) as f:
                self.dtraj_list = pickle.load(f)["dtraj"]

        # Keep only the files that are in the requested split
        self.data_files = set(self.data_files).intersection(set(split_files))
        self.data_files = sorted(list(self.data_files))
        self.data_files = self.expand_all_events(self.data_files)
        self.data_files, self.filtered_tracks = self.filter_all_tracks(self.data_files)

        self.dataset_cfg = dataset_cfg
        self.num_points = dataset_cfg.num_points
        self.max_depth = dataset_cfg.max_depth

        # Target shape of images (same as DINOv2)
        self.orig_shape = (1080, 1920)
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

    def __len__(self):
        assert len(self.data_files) == len(self.filtered_tracks)
        return len(self.data_files)

    def load_intrinsics(self):
        """
        Load camera intrinsics for HOI4D. A bit hacky.
        Camera intrinsics are expected to be placed one level above hoi4d dir
        with the directory name "camera_params"
        """
        intrinsic_dict = {}
        for cam_name in [
            "ZY20210800001",
            "ZY20210800002",
            "ZY20210800003",
            "ZY20210800004",
        ]:
            cam_param_pathname = f"{self.dataset_dir}/../camera_params/{cam_name}/"
            intrinsic_dict[cam_name] = np.load(f"{cam_param_pathname}/intrin.npy")
        return intrinsic_dict

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
            valid_event_idxs = [
                idx
                for idx, i in enumerate(all_events)
                if i["event"] in self.valid_event_types
            ]
            expanded_data_files.extend([(f, idx) for idx in valid_event_idxs])
        return expanded_data_files

    def load_split(self, split):
        """
        Load the filenames corresponding to each split - [train, val, test]
        The file containing the splits `metadata.json` is expected to be
        placed *outside* the directory. The splits were generated using the code
        in the General Flow codebase
        """
        with open(f"{self.dataset_dir}/../metadata.json") as f:
            all_splits = json.load(f)
        split_data = all_splits[split]

        split_data_fnames = list(set([i["index"] for i in split_data]))
        split_data_fnames = [
            f"{self.dataset_dir}/{i}/align_rgb/image.mp4" for i in split_data_fnames
        ]
        return split_data_fnames

    def load_camera_params(self, dir_name):
        with open(f"{dir_name}/3Dseg/output.log") as f:
            lines = f.readlines()

        matrices = []
        i = 0
        while i < len(lines):
            # Skip the first line (e.g., "0 0 1")
            i += 1

            # Read the next 4 lines for the matrix
            if i + 4 <= len(lines):
                matrix_data = []
                for j in range(4):
                    # Convert the line to float values
                    row = list(map(float, lines[i + j].split()))
                    matrix_data.append(row)

                matrices.append(np.array(matrix_data))

                # Move to the next block (skip 4 matrix lines + 1 header line)
                i += 4
            else:
                break
        cam2world = np.stack(matrices)
        cam_name = dir_name.split("/")[-7]
        K = self.intrinsic_dict[cam_name]
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

        event = all_events[event_idx]

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
        rgb_init = Image.open(
            f"{dir_name}/align_rgb/{str(event_start_idx).zfill(5)}.jpg"
        ).convert("RGB")
        rgb_init = np.asarray(self.rgb_preprocess(rgb_init))
        rgb_end = Image.open(
            f"{dir_name}/align_rgb/{str(event_end_idx).zfill(5)}.jpg"
        ).convert("RGB")
        rgb_end = np.asarray(self.rgb_preprocess(rgb_end))
        rgbs = np.array([rgb_init, rgb_end])

        depth_init = np.asarray(
            Image.open(f"{dir_name}/align_depth/{str(event_start_idx).zfill(5)}.png")
        )
        depth_init = Image.fromarray(depth_init / 1000.0)  # Convert to metres
        depth_init = np.asarray(self.depth_preprocess(depth_init))

        depth_end = np.asarray(
            Image.open(f"{dir_name}/align_depth/{str(event_end_idx).zfill(5)}.png")
        )
        depth_end = Image.fromarray(depth_end / 1000.0)  # Convert to metres
        depth_end = np.asarray(self.depth_preprocess(depth_end))

        depths = np.array([depth_init, depth_end])
        return rgbs, depths

    def filter_all_tracks(self, data_files):
        """
        We have some noisy data points in the dataset.
        When there is a segmentation/depth mismatch, points in the background are tracked.
        Some objects irrelevant to the current event are tracked.
        And some events just have very minimal motion.
        All these tracks are filtered out in this function.
        """
        print("Number of events before filtering:", len(data_files))

        # Check if cached
        cache_name = f"{self.cache_dir}/filter_{self.split}.pkl"
        if os.path.exists(cache_name):
            with open(cache_name, "rb") as f:
                cache_data = pickle.load(f)
            filtered_data_files = cache_data["filtered_data_files"]
            filtered_tracks = cache_data["filtered_tracks"]
            print("Number of events after filtering:", len(filtered_data_files))
            return filtered_data_files, filtered_tracks

        # We want points which move atleast `norm_threshold` cm,
        # and atleast `num_points_threshold` such points in an event
        norm_threshold = 0.075
        num_points_threshold = 25

        print("Beginning filtering of tracks:")
        filtered_data_files = []
        filtered_tracks = []
        for index in tqdm(range(len(data_files))):
            vid_name, event_idx = data_files[index]
            dir_name = os.path.dirname(os.path.dirname(vid_name))

            K, cam2world = self.load_camera_params(dir_name)
            _, event_start_idx, event_end_idx = self.load_event(dir_name, event_idx)
            start2end = self.get_start2end_transform(
                cam2world, event_start_idx, event_end_idx
            )

            if self.gt_source == "gflow_tracks":
                # Some events are missing because they've been filtered out during General Flow preprocessing.
                try:
                    idx, data = self.event_metadata_dict[
                        (vid_name[vid_name.find("ZY2021") : -20], event_start_idx)
                    ]
                except KeyError:
                    print(
                        "Missing",
                        (vid_name[vid_name.find("ZY2021") : -20], event_start_idx),
                    )
                    continue
                traj_data = self.dtraj_list[idx]
                start_tracks, end_tracks = traj_data[:, 0, :], traj_data[:, -1, :]
                caption = f"{data['action']} {data['object']}"
            elif self.gt_source == "spatrack_tracks":
                _, depths = self.load_rgbd(dir_name, event_start_idx, event_end_idx)
                start_tracks, end_tracks = self.process_and_register_tracks(
                    dir_name,
                    event_idx,
                    event_start_idx,
                    event_end_idx,
                    depths,
                    K,
                    start2end,
                )

                cross_displacement = end_tracks - start_tracks
                cd_mask = np.linalg.norm(cross_displacement, axis=1) > norm_threshold

                start_tracks = start_tracks[cd_mask]
                end_tracks = end_tracks[cd_mask]

            elif self.gt_source == "mano_handpose":
                hand_tracks = []
                try:
                    for hand_idx in [event_start_idx, event_end_idx]:
                        vid_path = vid_name[vid_name.find("ZY2021") : -20]
                        with open(
                            f"{self.root}/../handpose/refinehandpose_right/{vid_path}/{hand_idx}.pickle",
                            "rb",
                        ) as f:
                            hand_info = pickle.load(f, encoding="latin1")

                        theta = hand_info["poseCoeff"]
                        beta = hand_info["beta"]
                        hand_verts, _ = self.mano_interface.get_hand_params(theta, beta)

                        trans = hand_info["trans"]
                        tracks = (hand_verts.squeeze(0).numpy() / 1000.0) + trans
                        hand_tracks.append(tracks)
                except FileNotFoundError:
                    print(f"Could not find handpose for {vid_name}")
                    continue
                start_tracks, end_tracks = hand_tracks
            else:
                raise NotImplementedError

            if start_tracks.shape[0] > num_points_threshold:
                filtered_data_files.append(data_files[index])
                filtered_tracks.append((start_tracks, end_tracks))

        # Cache dataset
        if self.cache_dir and not os.path.exists(cache_name):
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_name, "wb") as f:
                cache_data = {
                    "filtered_data_files": filtered_data_files,
                    "filtered_tracks": filtered_tracks,
                }
                pickle.dump(cache_data, f)

        print("Number of events after filtering:", len(filtered_data_files))
        return filtered_data_files, filtered_tracks

    def get_scaled_intrinsics(self, K):
        # Getting scale factor from torchvision.transforms.Resize behaviour
        K_ = K.copy()

        scale_factor = self.target_shape / min(self.orig_shape)

        # Apply the scale factor to the intrinsics
        K_[0, 0] *= scale_factor  # fx
        K_[1, 1] *= scale_factor  # fy
        K_[0, 2] *= scale_factor  # cx
        K_[1, 2] *= scale_factor  # cy

        # Adjust the principal point (cx, cy) for the center crop
        crop_offset_x = (self.orig_shape[1] * scale_factor - self.target_shape) / 2
        crop_offset_y = (self.orig_shape[0] * scale_factor - self.target_shape) / 2

        # Adjust the principal point (cx, cy) for the center crop
        K_[0, 2] -= crop_offset_x  # Adjust cx for crop
        K_[1, 2] -= crop_offset_y  # Adjust cy for crop
        return K_

    def get_start2end_transform(self, cam2world, event_start_idx, event_end_idx):
        start2world = cam2world[event_start_idx]
        end2world = cam2world[event_end_idx]
        start2end = start2world @ np.linalg.inv(end2world)
        return start2end

    def process_and_register_tracks(
        self, dir_name, event_idx, event_start_idx, event_end_idx, depths, K, start2end
    ):
        """
        Load the tracks corresponding to the start and end of the event.
        Unproject the tracks to 3D point clouds, register the point cloud
        `end_tracks` to the `start_tracks` coordinate frame.
        """
        K_ = self.get_scaled_intrinsics(K)
        tracks = np.load(f"{dir_name}/spatracker_3d_tracks.npz")
        event_tracks = tracks[f"tracks_{event_idx}"]

        start_tracks = event_tracks[0]
        start_tracks = start_tracks * self.scale_factor

        # Clip to image boundaries, unproject
        ty = np.clip(start_tracks[:, 1].round().astype(int), 0, depths[0].shape[0] - 1)
        tx = np.clip(start_tracks[:, 0].round().astype(int), 0, depths[0].shape[1] - 1)
        # Overwrite SpatialTracker depth with GT depth
        start_tracks[:, 2] = depths[0][ty, tx]
        start_tracks[:, 0] = (
            (start_tracks[:, 0] - K_[0, 2]) * start_tracks[:, 2]
        ) / K_[0, 0]
        start_tracks[:, 1] = (
            (start_tracks[:, 1] - K_[1, 2]) * start_tracks[:, 2]
        ) / K_[1, 1]
        start_mask = np.linalg.norm(start_tracks, axis=1) != 0

        end_tracks = event_tracks[-1]
        end_tracks = end_tracks * self.scale_factor

        # Clip to image boundaries, unproject
        ty = np.clip(end_tracks[:, 1].round().astype(int), 0, depths[0].shape[0] - 1)
        tx = np.clip(end_tracks[:, 0].round().astype(int), 0, depths[0].shape[1] - 1)
        # Overwrite SpatialTracker depth with GT depth
        end_tracks[:, 2] = depths[1][ty, tx]
        end_tracks[:, 0] = ((end_tracks[:, 0] - K_[0, 2]) * end_tracks[:, 2]) / K_[0, 0]
        end_tracks[:, 1] = ((end_tracks[:, 1] - K_[1, 2]) * end_tracks[:, 2]) / K_[1, 1]
        end_mask = np.linalg.norm(end_tracks, axis=1) != 0

        # Remove zero points
        mask = np.logical_and(start_mask, end_mask)
        start_tracks = start_tracks[mask]
        end_tracks = end_tracks[mask]

        mask = np.logical_and(start_mask, end_mask)
        start_tracks = start_tracks[mask]
        end_tracks = end_tracks[mask]

        # Register the tracks at the end of the chunk with respect
        # to the coordinate frame at the beginning of the chunk
        end_tracks = np.hstack((end_tracks, np.ones((end_tracks.shape[0], 1))))
        end_tracks = (start2end @ end_tracks.T).T[:, :3].astype(np.float32)

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
        valid_depth = np.logical_and(z_flat > 0, z_flat < self.max_depth)
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

        scene_pcd_pt3d = torch.from_numpy(points[None])
        scene_pcd_downsample, scene_points_idx = sample_farthest_points(
            scene_pcd_pt3d, K=self.num_points, random_start_point=False
        )
        scene_pcd = scene_pcd_downsample.squeeze().numpy()

        # Get corresponding features at the indices
        scene_feat_pcd = feat_flat[scene_points_idx.squeeze().numpy()]
        return scene_pcd, scene_feat_pcd

    def compose_caption(self, dir_name, event):
        """Compose the caption from the event name and the object."""
        event_name = event["event"]
        # Get the object name from the pose file.
        # The dataset does not have a consistent naming scheme .......
        objpose_fname = f"{dir_name}/objpose/0.json"
        if not os.path.exists(objpose_fname):
            objpose_fname = f"{dir_name}/objpose/00000.json"
        obj_name = json.load(open(objpose_fname))["dataList"][0]["label"]
        caption = f"{event_name} {obj_name}"
        return caption

    def load_rgb_text_feat(self, dir_name, event_idx):
        """
        Load RGB/text features generated with DINOv2 and SIGLIP
        """
        features = np.load(f"{dir_name}/rgb_text_features/{event_idx}_compressed.npz")
        rgb_embed, text_embed = features["rgb_embed"], features["text_embed"]

        upscale_by = 4
        rgb_embed = rgb_embed.transpose(2, 0, 1)[None].astype(np.float32)
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
        vid_name, event_idx = self.data_files[index]
        dir_name = os.path.dirname(os.path.dirname(vid_name))

        K, cam2world = self.load_camera_params(dir_name)
        K_ = self.get_scaled_intrinsics(K)
        event, event_start_idx, event_end_idx = self.load_event(dir_name, event_idx)
        start2end = self.get_start2end_transform(
            cam2world, event_start_idx, event_end_idx
        )

        rgbs, depths = self.load_rgbd(dir_name, event_start_idx, event_end_idx)
        start_tracks, end_tracks = self.filtered_tracks[index]

        caption = self.compose_caption(dir_name, event)
        rgb_embed, text_embed = self.load_rgb_text_feat(dir_name, event_idx)
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
            "vid_name": dir_name,
            "pcd_mean": action_pcd_mean,
            "pcd_std": scene_pcd_std,
        }
        return item


class HOI4DDataModule(BaseDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):
        super().__init__(batch_size, val_batch_size, num_workers, dataset_cfg)
        self.val_tags = ["human"]

    def setup(self, stage: str = "fit"):
        self.stage = stage
        self.val_datasets = {}
        self.test_datasets = {}

        self.train_dataset = HOI4DDataset(self.root, self.dataset_cfg, "train")
        for tag in self.val_tags:
            self.val_datasets[tag] = HOI4DDataset(self.root, self.dataset_cfg, "val")
            self.test_datasets[tag] = HOI4DDataset(self.root, self.dataset_cfg, "test")
        if self.train_dataset.cache_dir:
            self.train_dataset.cache(
                td.cachers.Pickle(Path(self.train_dataset.cache_dir))
            )
            for tag in self.val_tags:
                self.val_datasets[tag].cache(
                    td.cachers.Pickle(Path(self.train_dataset.cache_dir) / f"val_{tag}")
                )
