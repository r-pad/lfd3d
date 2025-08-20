"""This file adapts a LeRobot dataset to the LFD3D format."""

from pathlib import Path

import numpy as np
import torch
import torchdatasets as td
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lfd3d.datasets.base_data import BaseDataModule, BaseDataset
from lfd3d.datasets.rpad_foxglove.rpad_foxglove_dataset import (
    RGBTextFeaturizer,
    RpadFoxgloveDataset,
)
from PIL import Image
from tqdm import tqdm


class RpadLeRobotDataset(BaseDataset):
    def __init__(self, dataset_cfg, root: str | None = None, split: str = "train"):
        super().__init__()
        repo_id = dataset_cfg.repo_id

        self.lerobot_dataset = LeRobotDataset(
            repo_id=repo_id, root=root, tolerance_s=0.0004, video_backend="pyav"
        )
        self.lerobot_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)

        # Store the same dataset configuration...
        self.cache_dir = dataset_cfg.cache_dir
        self.dataset_cfg = dataset_cfg
        self.num_points = dataset_cfg.num_points
        self.max_depth = dataset_cfg.max_depth
        self.split = split

        self.rgb_text_featurizer = RGBTextFeaturizer(
            target_shape=self.target_shape, rgb_feat=self.dataset_cfg.rgb_feat
        )

        # indexes of selected gripper points -> handpicked
        self.GRIPPER_IDX = {
            "aloha": np.array([6, 197, 174]),
            "human": np.array([343, 763, 60]),
        }

    def load_transition(
        self, idx
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
        start_item = self.lerobot_dataset[idx]
        # The next_event_idx is relative to the episode, so we calculate the absolute index
        end_idx = (
            start_item["next_event_idx"] - start_item["frame_index"] + idx
        ).item()
        # HACK! I don't understand why we have an off-by-one in the next_event_idx.
        # So far it only seems to cause problems in the final transition. But this is
        # a major code smell, that makes me wonder if it's off-by-one everywhere. But
        # I haven't checked thouroghly. Feels more like a subtle bug than an off-by-one.
        end_idx = min(end_idx, len(self.lerobot_dataset) - 1)
        end_item = self.lerobot_dataset[end_idx]
        task = start_item["task"]
        episode_index = start_item["episode_index"]

        COLOR_KEY = "observation.images.cam_azure_kinect.color"
        rgb_init = (start_item[COLOR_KEY].permute(1, 2, 0).numpy() * 255).astype(
            np.uint8
        )
        rgb_init = Image.fromarray(rgb_init)
        rgb_init = np.asarray(self.rgb_preprocess(rgb_init))
        rgb_end = (end_item[COLOR_KEY].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rgb_end = Image.fromarray(rgb_end)
        rgb_end = np.asarray(self.rgb_preprocess(rgb_end))
        rgbs = np.array([rgb_init, rgb_end])

        DEPTH_KEY = "observation.images.cam_azure_kinect.transformed_depth"
        depth_init = Image.fromarray(start_item[DEPTH_KEY].numpy()[0])
        depth_init = np.asarray(self.depth_preprocess(depth_init))
        depth_end = Image.fromarray(end_item[DEPTH_KEY].numpy()[0])
        depth_end = np.asarray(self.depth_preprocess(depth_end))
        depths = np.array([depth_init, depth_end])

        GRIPPER_PCD_KEY = "observation.points.gripper_pcds"
        gripper_pcd_init = start_item[GRIPPER_PCD_KEY]
        gripper_pcd_end = end_item[GRIPPER_PCD_KEY]
        gripper_pcds = np.array([gripper_pcd_init, gripper_pcd_end])

        return rgbs, depths, gripper_pcds, task, f"{episode_index}"

    @staticmethod
    def _load_camera_params():
        file_path = (
            Path(__file__).parent.parent / "rpad_foxglove/calibration/intrinsics.txt"
        )
        return np.loadtxt(file_path)

    def __getitem__(self, index):
        # Retrieve the item from the underlying LeRobot dataset
        start2end = torch.eye(4)  # Static camera

        rgbs, depths, gripper_pcds, caption, demo_name = self.load_transition(index)

        start_tracks, end_tracks = gripper_pcds[0], gripper_pcds[1]
        actual_caption = caption
        K = self._load_camera_params()
        H, W = (720, 1280)
        orig_shape = (H, W)
        K_ = RpadFoxgloveDataset.get_scaled_intrinsics(K, orig_shape, self.target_shape)

        rgb_embed, text_embed = self.rgb_text_featurizer.compute_rgb_text_feat(
            rgbs[0], caption
        )
        start_scene_pcd, start_scene_feat_pcd, augment_tf = self.get_scene_pcd(
            rgb_embed, depths[0], K_, self.num_points, self.max_depth
        )

        # We only use this dataset with aloha gripper, human version is collected w/ foxglove.
        gripper_idx = self.GRIPPER_IDX["aloha"]

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
            "actual_caption": actual_caption,
        }
        return item

    def __len__(self):
        # Return the length of the underlying LeRobot dataset
        return len(self.lerobot_dataset)


class RpadLeRobotDataModule(BaseDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg, seed):
        super().__init__(batch_size, val_batch_size, num_workers, dataset_cfg, seed)
        self.val_tags = ["aloha"]
        # Subset of train to use for eval
        self.TRAIN_SUBSET_SIZE = 20

    def setup(self, stage: str = "fit"):
        self.stage = stage
        self.val_datasets = {}
        self.test_datasets = {}

        self.train_dataset = RpadLeRobotDataset(
            dataset_cfg=self.dataset_cfg, root=self.root, split="train"
        )
        for tag in self.val_tags:
            dataset_cfg = self.dataset_cfg.copy()
            dataset_cfg.data_sources = [tag]
            self.val_datasets[tag] = RpadLeRobotDataset(
                dataset_cfg=dataset_cfg, root=self.root, split="val"
            )
            self.test_datasets[tag] = RpadLeRobotDataset(
                dataset_cfg=dataset_cfg, split="test"
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


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg_dict = {
        "dataset": {
            "name": "rpadFoxglove",
            "data_dir": None,
            "cache_dir": None,
            "cache_invalidation_rate": 0.05,
            "train_size": None,
            "val_size": None,
            "num_points": 8192,
            "max_depth": 1.5,
            "normalize": False,
            "augment_train": False,
            "augment_cfg": {
                "augment_prob": 0.75,
                "augment_transform": True,
                "pcd_sample": ["voxel", "fps"],
                "fps_num_points": [4096, 8192],
                "voxel_size": [0.01, 0.02],
            },
            "data_name": "0627",
            "additional_img_dir": None,
            "data_sources": ["aloha"],
            "oversample_robot": True,
            "rgb_feat": False,
            "use_full_text": True,
            "use_intermediate_frames": False,
            "use_gemini_subgoals": False,
            "repo_id": "beisner/aloha_plate_placement_goal",
        },
        "model": {
            "name": "articubot",
            "type": "cross_displacement",
            "in_channels": 4,
            "num_classes": 13,
            "keep_gripper_in_fps": False,
            "add_action_pcd_masked": True,
            "use_text_embedding": True,
            "use_rgb": False,
            "is_gmm": False,
            "fixed_variance": [0.01, 0.05, 0.1, 0.25, 0.5],
            "uniform_weights_coeff": 0.1,
        },
        "training": {
            "lr": 0.0001,
            "lr_warmup_steps": 100,
            "weight_decay": 1e-05,
            "epochs": 100,
            "precision": 32,
            "batch_size": 8,
            "val_batch_size": 8,
            "grad_clip_norm": 0.3,
            "num_training_steps": "None",
            "check_val_every_n_epochs": 10,
            "additional_train_logging_period": 100,
            "n_samples_wta": 5,
            "save_wta_to_disk": False,
        },
        "log_dir": "${hydra:runtime.cwd}/logs",
        "output_dir": "${hydra:runtime.output_dir}",
        "job_type": "${mode}_${dataset.name}",
        "lightning": {"checkpoint_dir": "${output_dir}/checkpoints"},
        "wandb": {
            "entity": "r-pad",
            "project": "lfd3d",
            "group": None,
            "save_dir": "${output_dir}",
            "artifact_dir": "${hydra:runtime.cwd}/wandb_artifacts",
        },
        "mode": "train",
        "seed": 42,
        "resources": {"num_workers": 0, "gpus": [0]},
        "load_checkpoint": False,
        "checkpoint": {
            "run_id": False,
            "type": "rmse",
            "reference": "${wandb.entity}/${wandb.project}/best_${checkpoint.type}_model-${checkpoint.run_id}:best",
        },
        "lora": {"enable": False, "rank": 4, "target_modules": "all", "dropout": 0.1},
    }
    cfg = OmegaConf.create(cfg_dict)
    lr_dset = RpadLeRobotDataset(dataset_cfg=cfg.dataset)
    item = lr_dset[0]

    datamodule = RpadLeRobotDataModule(
        batch_size=cfg.training.batch_size,
        val_batch_size=cfg.training.val_batch_size,
        num_workers=cfg.resources.num_workers,
        dataset_cfg=cfg.dataset,
        seed=cfg.seed,
    )
    datamodule.setup()
    train_dloader = datamodule.train_dataloader()
    val_dloader = datamodule.val_dataloader()

    for batch in tqdm(train_dloader):
        print(batch)

    for batch in tqdm(val_dloader):
        print(batch)
