import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchdatasets as td
import zarr
from lfd3d.datasets.base_data import BaseDataModule, BaseDataset
from lfd3d.datasets.rgb_text_feature_gen import (
    get_dinov2_image_embedding,
    get_siglip_text_embedding,
)
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from transformers import AutoModel, AutoProcessor


class RpadFoxgloveDataset(BaseDataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.additional_img_dir = dataset_cfg.additional_img_dir
        self.split = split

        self.cache_dir = dataset_cfg.cache_dir
        self.dataset_cfg = dataset_cfg

        self.data_sources = dataset_cfg.data_sources
        self.current_dir = os.path.dirname(__file__)
        with open(f"{self.current_dir}/{split}.json") as f:
            self.split_names = json.load(f)

        self.num_points = dataset_cfg.num_points
        self.max_depth = dataset_cfg.max_depth

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

        self.captions = {}
        self.text_embeddings = {}
        self.dataset = zarr.group(root)
        self.dataset_index = self.expand_all_events()
        self.size = len(self.dataset_index)

        self.siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(
            "cpu"
        )
        self.siglip_processor = AutoProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(
            "cpu"
        )
        # indexes of selected gripper points -> handpicked
        self.GRIPPER_IDX = {
            "aloha": np.array([6, 197, 174]),
            "human": np.array([343, 763, 60]),
        }

    def __len__(self):
        return self.size

    def source_of_data(self, demo_name):
        """
        Return the source of the current demo i.e. method of collection
        """
        if demo_name not in self.dataset:
            return "human"

        demo = self.dataset[demo_name]
        # Currently identifying demo type by checking number of vertices
        # For human data, we have 778 vertices from the MANO mesh
        # For robot data, we're sampling 500 points from the mesh.
        # HACK: This is pretty hacky, should probably find a better way to do this.
        n_points = demo["gripper_pos"].shape[1]
        if n_points == 778:
            demo_type = "human"
        elif n_points == 500:
            demo_type = "aloha"
        else:
            raise NotImplementedError

        return demo_type

    def expand_all_events(self):
        """This function *expands* each event to have an associated event_idx.
        Updates `self.captions` with each event and its various subgoals

        Returns:
            expanded_index (list of tuples (int/fname, int, [indexes])):
        A list of tuples where each tuple contains the event index, subgoal index
        and start/end frame indexes
        """
        expanded_index = []

        for demo_name in self.dataset:
            if demo_name not in self.split_names:
                continue

            if self.source_of_data(demo_name) not in self.data_sources:
                continue

            demo = self.dataset[demo_name]
            events = demo["events"]
            num_events = len(events["event"])

            idx = np.argsort(events["event"])
            sorted_event = np.asarray(events["event"])[idx]

            # Overwrite with a concatenation of all the subgoals
            if self.dataset_cfg.use_full_text:
                sorted_event = np.array(
                    [" and ".join(sorted_event)] * len(sorted_event)
                )

            for i in range(num_events):
                expanded_event_idx = self.get_event_start_end_indexes(demo_name, i)
                expanded_event_caption = {
                    (demo_name, i): sorted_event[i].replace("_", " ")
                }
                expanded_index.extend(expanded_event_idx)
                self.captions.update(expanded_event_caption)

        if self.additional_img_dir is not None and self.split == "train":
            # A different format for the additional image only data
            dirs = os.listdir(self.additional_img_dir)
            dirs = [d for d in dirs if os.path.isdir(f"{self.additional_img_dir}/{d}")]
            for dir in dirs:
                hand_pose = np.load(f"{self.additional_img_dir}/{dir}/hand_pose.npy")
                hand_pose_mean = hand_pose.mean(axis=(1, 2))
                idxs = np.array(hand_pose_mean.nonzero()).flatten().tolist()
                events = json.load(open(f"{self.additional_img_dir}/{dir}/events.json"))
                for i in range(len(events) - 1):
                    if i in idxs and (i + 1) in idxs:
                        expanded_index.append(
                            (
                                dir,
                                i,
                                {
                                    "rgb_start": i,
                                    "rgb_end": i + 1,
                                    "depth_start": i,
                                    "depth_end": i + 1,
                                },
                            )
                        )
                        self.captions[(dir, i)] = events[i]
        return expanded_index

    def load_camera_params(self, demo_name):
        if demo_name not in self.dataset:
            metadata = np.load(f"{self.additional_img_dir}/metadata.npz")
            K = metadata["K"]
            height, width = metadata["height"], metadata["width"]
            return K, (height, width)
        demo = self.dataset[demo_name]
        cam_info = demo["raw"]["rgb"]["camera_info"]
        K = cam_info["k"][0]
        height = cam_info["height"][0]
        width = cam_info["width"][0]
        return K, (height, width)

    def get_event_start_end_indexes(self, demo_name, subgoal_idx):
        demo = self.dataset[demo_name]

        events = demo["events"]
        idx = np.argsort(events["event"])
        sorted_end = np.asarray(events["end"])[idx]
        event_end_ts = datetime.fromisoformat(sorted_end[subgoal_idx]).timestamp()

        if subgoal_idx == 0:
            # First frame where we have non-zero gripper_pos
            # For estimating hand pose, we add zeros if hand is not detected.
            event_start_idx = np.argmax(
                np.any(np.asarray(demo["gripper_pos"]) != 0, axis=(1, 2))
            )
            event_start_ts = demo["raw"]["rgb"]["image_rect"]["ts"][event_start_idx]
        else:
            # Start timestamp is end timestamp of previous subgoal
            event_start_ts = datetime.fromisoformat(
                sorted_end[subgoal_idx - 1]
            ).timestamp()

        rgb_ts = demo["raw"]["rgb"]["image_rect"]["publish_ts"]
        depth_ts = demo["raw"]["depth_registered"]["image_rect"]["publish_ts"]

        event_start_idx_rgb = np.searchsorted(rgb_ts, event_start_ts)
        event_end_idx_rgb = np.searchsorted(rgb_ts, event_end_ts)

        event_start_idx_depth = np.searchsorted(depth_ts, event_start_ts)
        event_end_idx_depth = np.searchsorted(depth_ts, event_end_ts)

        if self.dataset_cfg.use_intermediate_frames:
            rgb_ts_event = rgb_ts[event_start_idx_rgb : event_end_idx_rgb - 1]

            event_data = []
            for rgb_idx_offset in range(rgb_ts_event.shape[0]):
                rgb_idx = event_start_idx_rgb + rgb_idx_offset
                rgb_timestamp = rgb_ts_event[rgb_idx_offset]

                # Find the index of the closest depth timestamp
                depth_idx = np.searchsorted(depth_ts, rgb_timestamp)
                event_data.append(
                    (
                        demo_name,
                        subgoal_idx,
                        {
                            "rgb_start": rgb_idx,
                            "rgb_end": event_end_idx_rgb,
                            "depth_start": depth_idx,
                            "depth_end": event_end_idx_depth,
                        },
                    )
                )
        else:
            event_data = [
                (
                    demo_name,
                    subgoal_idx,
                    {
                        "rgb_start": event_start_idx_rgb,
                        "rgb_end": event_end_idx_rgb,
                        "depth_start": event_start_idx_depth,
                        "depth_end": event_end_idx_depth,
                    },
                )
            ]
        return event_data

    def load_rgbd_for_img(self, demo_name, subgoal_idx, event_indexes, K):
        """An alternate function for loading the RGBD data for the additional images.
        Kind of ugly tbh, alternate way is to record the image data in zarr as well instead of
        this bespoke format."""
        start_idx = subgoal_idx  # Generalize start index
        end_idx = start_idx + 1

        # Load RGB images
        rgb_init_path = f"{self.additional_img_dir}/{demo_name}/rgb_{start_idx:03d}.png"
        rgb_end_path = f"{self.additional_img_dir}/{demo_name}/rgb_{end_idx:03d}.png"
        rgb_init = Image.open(rgb_init_path).convert("RGB")
        rgb_end = Image.open(rgb_end_path).convert("RGB")

        # Load depth images
        depth_init_path = (
            f"{self.additional_img_dir}/{demo_name}/depth_{start_idx:03d}.png"
        )
        depth_end_path = (
            f"{self.additional_img_dir}/{demo_name}/depth_{end_idx:03d}.png"
        )
        depth_init = np.asarray(Image.open(depth_init_path)).astype(np.float32) / 1000.0
        depth_end = np.asarray(Image.open(depth_end_path)).astype(np.float32) / 1000.0

        # Preprocess and stack
        rgb_init = np.asarray(self.rgb_preprocess(rgb_init))
        rgb_end = np.asarray(self.rgb_preprocess(rgb_end))
        rgbs = np.array([rgb_init, rgb_end])

        depth_init = Image.fromarray(depth_init)
        depth_init = np.asarray(self.depth_preprocess(depth_init))
        depth_end = Image.fromarray(depth_end)
        depth_end = np.asarray(self.depth_preprocess(depth_end))
        depths = np.array([depth_init, depth_end])

        return rgbs, depths

    def load_rgbd(self, demo_name, subgoal_idx, event_indexes, K):
        if demo_name not in self.dataset:
            return self.load_rgbd_for_img(demo_name, subgoal_idx, event_indexes, K)
        demo = self.dataset[demo_name]

        # Return rgb/depth at beginning and end of event
        rgb_init = Image.fromarray(
            demo["raw"]["rgb"]["image_rect"]["img"][event_indexes["rgb_start"]]
        )
        rgb_init = np.asarray(self.rgb_preprocess(rgb_init))
        rgb_end = Image.fromarray(
            demo["raw"]["rgb"]["image_rect"]["img"][event_indexes["rgb_end"]]
        )
        rgb_end = np.asarray(self.rgb_preprocess(rgb_end))
        rgbs = np.array([rgb_init, rgb_end])

        depth_init = (
            (
                demo["raw"]["depth_registered"]["image_rect"]["img"][
                    event_indexes["depth_start"]
                ]
                / 1000.0
            )
            .squeeze()
            .astype(np.float32)
        )
        depth_init = Image.fromarray(depth_init)
        depth_init = np.asarray(self.depth_preprocess(depth_init))
        depth_end = (
            (
                demo["raw"]["depth_registered"]["image_rect"]["img"][
                    event_indexes["depth_end"]
                ]
                / 1000.0
            )
            .squeeze()
            .astype(np.float32)
        )
        depth_end = Image.fromarray(depth_end)
        depth_end = np.asarray(self.depth_preprocess(depth_end))
        depths = np.array([depth_init, depth_end])

        return rgbs, depths

    def load_gripper_pcd(self, demo_name, event_start_idx, event_end_idx):
        if demo_name not in self.dataset:
            handpose = np.load(f"{self.additional_img_dir}/{demo_name}/hand_pose.npy")
            start_tracks = handpose[event_start_idx]
            end_tracks = handpose[event_end_idx]
        else:
            demo = self.dataset[demo_name]
            start_tracks = demo["gripper_pos"][event_start_idx]
            end_tracks = demo["gripper_pos"][event_end_idx]
        return start_tracks, end_tracks

    def compute_rgb_text_feat(self, rgb, text):
        """
        Compute RGB/text features generated with DINOv2 and SIGLIP
        """
        siglip_dim = 1152
        if text not in self.text_embeddings:
            # Compute features on CPU to avoid CUDA multiprocessing issues
            # We're only computing the features once and caching so its okay.
            self.text_embeddings[text] = get_siglip_text_embedding(
                text,
                siglip=self.siglip,
                siglip_processor=self.siglip_processor,
                device="cpu",
            )
        text_embed = self.text_embeddings[text]

        if self.dataset_cfg.rgb_feat:
            # Compress RGB features
            pca_n_components = 256
            rgb_embed = get_dinov2_image_embedding(
                Image.fromarray(rgb), dinov2=self.dinov2, device="cpu"
            )

            pca_model = PCA(n_components=pca_n_components)
            rgb_embed = pca_model.fit_transform(
                rgb_embed.reshape(-1, rgb_embed.shape[2])
            )
            rgb_embed = rgb_embed.reshape(
                self.target_shape, self.target_shape, pca_n_components
            )
        else:
            # Just return the (normalized) RGB values if features are not required.
            rgb_embed = (((rgb / 255.0) * 2) - 1).astype(np.float32)

        return rgb_embed, text_embed

    def __getitem__(self, idx):
        demo_name, subgoal_idx, event_indexes = self.dataset_index[idx]

        K, orig_shape = self.load_camera_params(demo_name)
        K_ = self.get_scaled_intrinsics(K, orig_shape, self.target_shape)

        caption = self.captions[(demo_name, subgoal_idx)]
        start2end = torch.eye(4)  # Static camera

        rgbs, depths = self.load_rgbd(demo_name, subgoal_idx, event_indexes, K_)

        start_tracks, end_tracks = self.load_gripper_pcd(
            demo_name, event_indexes["rgb_start"], event_indexes["rgb_end"]
        )

        rgb_embed, text_embed = self.compute_rgb_text_feat(rgbs[0], caption)
        start_scene_pcd, start_scene_feat_pcd, augment_tf = self.get_scene_pcd(
            rgb_embed, depths[0], K_, self.num_points, self.max_depth
        )

        gripper_idx = self.GRIPPER_IDX[self.source_of_data(demo_name)]

        action_pcd_mean, scene_pcd_std = self.get_normalize_mean_std(
            start_tracks, start_scene_pcd, self.dataset_cfg
        )
        start_tracks, end_tracks, start_scene_pcd = self.transform_pcds(
            start_tracks,
            end_tracks,
            start_scene_pcd,
            action_pcd_mean,
            scene_pcd_std,
            augment_tf,
        )

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
            "vid_name": demo_name,
            "pcd_mean": action_pcd_mean,
            "pcd_std": scene_pcd_std,
            "gripper_idx": gripper_idx,
            "augment_R": augment_tf["R"],
            "augment_t": augment_tf["t"],
            "augment_C": augment_tf["C"],
        }
        return item


class RpadFoxgloveDataModule(BaseDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg, seed):
        super().__init__(batch_size, val_batch_size, num_workers, dataset_cfg, seed)
        self.val_tags = ["human", "aloha"]
        # Subset of train to use for eval
        self.TRAIN_SUBSET_SIZE = 20

    def setup(self, stage: str = "fit"):
        self.stage = stage
        self.val_datasets = {}
        self.test_datasets = {}

        self.train_dataset = RpadFoxgloveDataset(self.root, self.dataset_cfg, "train")
        for tag in self.val_tags:
            dataset_cfg = self.dataset_cfg.copy()
            dataset_cfg.data_sources = [tag]
            self.val_datasets[tag] = RpadFoxgloveDataset(self.root, dataset_cfg, "val")
            self.test_datasets[tag] = RpadFoxgloveDataset(
                self.root, dataset_cfg, "test"
            )

        if self.train_dataset.cache_dir:
            invalidating_cacher = td.cachers.ProbabilisticCacherWrapper(
                td.cachers.HDF5(Path(self.train_dataset.cache_dir)),
                invalidation_rate=self.dataset_cfg.cache_invalidation_rate,
                seed=self.seed,
            )
            self.train_dataset.cache(invalidating_cacher)
            for tag in self.val_tags:
                self.val_datasets[tag].cache(
                    td.cachers.HDF5(Path(self.train_dataset.cache_dir) / f"val_{tag}")
                )
                self.test_datasets[tag].cache(
                    td.cachers.HDF5(Path(self.train_dataset.cache_dir) / f"test_{tag}")
                )
