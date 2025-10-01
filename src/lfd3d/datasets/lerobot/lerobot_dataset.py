"""This file adapts a LeRobot dataset to the LFD3D format."""

import random
from pathlib import Path

import numpy as np
import torch
import torchdatasets as td
import torchvision.transforms as T
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiHomogeneousLeRobotDataset,
)
from lerobot.common.datasets.utils import get_episode_data_index
from lfd3d.datasets.base_data import BaseDataModule, BaseDataset
from lfd3d.datasets.rgb_text_featurizer import RGBTextFeaturizer
from lfd3d.utils.data_utils import collate_pcd_fn
from lfd3d.utils.viz_utils import generate_heatmap_from_points, project_pcd_on_image
from PIL import Image
from torch.utils import data
from tqdm import tqdm


def make_dataset(repo_id, root):
    if isinstance(repo_id, str):
        lerobot_dataset = LeRobotDataset(repo_id=repo_id, root=root, tolerance_s=0.0004)
    else:
        datasets = []
        for dataset_id in repo_id:
            ds = LeRobotDataset(repo_id=dataset_id, root=root, tolerance_s=0.0004)
            datasets.append(ds)
        lerobot_dataset = MultiHomogeneousLeRobotDataset(datasets)
    return lerobot_dataset


class RpadLeRobotDataset(BaseDataset):
    def __init__(
        self,
        dataset_cfg,
        root: str | None = None,
        split: str = "train",
        split_indices: list = [],
    ):
        super().__init__()
        repo_id = dataset_cfg.repo_id

        self.lerobot_dataset = make_dataset(repo_id, root)

        # Store the same dataset configuration...
        self.cache_dir = dataset_cfg.cache_dir
        self.dataset_cfg = dataset_cfg
        self.num_points = dataset_cfg.num_points
        self.max_depth = dataset_cfg.max_depth
        self.pixel_score = dataset_cfg.pixel_score
        self.split = split

        self.split_indices = split_indices
        self.color_key = dataset_cfg.color_key
        self.depth_key = dataset_cfg.depth_key
        self.gripper_pcd_key = dataset_cfg.gripper_pcd_key

        self.rgb_text_featurizer = RGBTextFeaturizer(
            target_shape=self.target_shape, rgb_feat=self.dataset_cfg.rgb_feat
        )

        self.rgb_normalizer = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # indexes of selected gripper points -> handpicked
        self.GRIPPER_IDX = {
            "aloha": np.array([6, 197, 174]),
            "human": np.array([343, 763, 60]),
            "libero_franka": np.array([0, 1, 2]),
        }

        self.GRIPPER_PCD_IDX = {"top": 0, "left": 1, "right": 2, "grasp_center": 3}

    def extract_goal(self, item):
        if self.dataset_cfg.use_subgoals:
            return item["subgoal"]
        else:
            return item["task"]

    def load_transition(
        self, idx
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
        start_item = self.lerobot_dataset[idx]
        task = self.extract_goal(start_item)
        episode_index = start_item["episode_index"]
        # The next_event_idx is relative to the episode, so we calculate the absolute index
        end_idx = (
            start_item["next_event_idx"] - start_item["frame_index"] + idx
        ).item()

        # Off-by-one error in the data generation code on lerobot side
        # Fixed, but this change remains here because I don't want to regenerate
        # the datasets. Can probably be removed at some later point.
        if (end_idx == len(self.lerobot_dataset)) or (
            self.lerobot_dataset[end_idx]["episode_index"]
            != start_item["episode_index"]
        ):
            end_idx = end_idx - 1

        end_item = self.lerobot_dataset[end_idx]
        COLOR_KEY = self.color_key
        rgb_init = (start_item[COLOR_KEY].permute(1, 2, 0).numpy() * 255).astype(
            np.uint8
        )
        orig_shape = rgb_init.shape[:2]

        rgb_init = Image.fromarray(rgb_init)
        rgb_init = np.asarray(self.rgb_preprocess(rgb_init))
        rgb_end = (end_item[COLOR_KEY].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rgb_end = Image.fromarray(rgb_end)
        rgb_end = np.asarray(self.rgb_preprocess(rgb_end))
        rgbs = np.array([rgb_init, rgb_end])

        DEPTH_KEY = self.depth_key
        depth_init = Image.fromarray(start_item[DEPTH_KEY].numpy()[0])
        depth_init = np.asarray(self.depth_preprocess(depth_init))
        depth_end = Image.fromarray(end_item[DEPTH_KEY].numpy()[0])
        depth_end = np.asarray(self.depth_preprocess(depth_end))
        depths = np.array([depth_init, depth_end])

        GRIPPER_PCD_KEY = self.gripper_pcd_key
        gripper_pcd_init = start_item[GRIPPER_PCD_KEY]
        gripper_pcd_end = end_item[GRIPPER_PCD_KEY]
        gripper_pcds = np.array([gripper_pcd_init, gripper_pcd_end])

        return rgbs, depths, orig_shape, gripper_pcds, task, f"{episode_index}"

    def _load_camera_params(self, data_source):
        file_path = (
            Path(__file__).parent.parent / f"{data_source}_calibration/intrinsics.txt"
        )
        return np.loadtxt(file_path)

    def source_of_data(self, idx):
        """
        Return the source of the current demo i.e. where the data is from
        """
        item = self.lerobot_dataset[idx]
        # Currently identifying demo type by checking number of vertices
        # For human data, we have 778 vertices from the MANO mesh
        # For aloha data, we're sampling 500 points from the mesh.
        # For libero data, we have 4 points calculated analytically
        # HACK: This is pretty hacky, should find a better way to do this.
        n_points = item["observation.points.gripper_pcds"].shape[0]
        if n_points == 778:
            demo_type = "human"
        elif n_points == 500:
            demo_type = "aloha"
        elif n_points == 4:
            demo_type = "libero_franka"
        else:
            raise NotImplementedError

        return demo_type

    def __getitem__(self, index):
        # Map the dataset index to the actual LeRobot dataset index using split_indices
        actual_index = self.split_indices[index]
        data_source = self.source_of_data(actual_index)

        start2end = torch.eye(4)  # Static camera

        # Retrieve the item from the underlying LeRobot dataset
        rgbs, depths, orig_shape, gripper_pcds, caption, demo_name = (
            self.load_transition(actual_index)
        )

        K = self._load_camera_params(data_source)
        K_ = BaseDataset.get_scaled_intrinsics(K, orig_shape, self.target_shape)

        start_tracks, end_tracks = gripper_pcds[0], gripper_pcds[1]
        actual_caption = caption

        rgb_embed, text_embed = self.rgb_text_featurizer.compute_rgb_text_feat(
            rgbs[0], caption
        )
        start_scene_pcd, start_scene_feat_pcd, augment_tf = self.get_scene_pcd(
            rgb_embed, depths[0], K_, self.num_points, self.max_depth
        )

        gripper_idx = self.GRIPPER_IDX[data_source]

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
            "data_source": data_source,
        }

        if self.pixel_score:
            pixel_dict = {}

            # CNN input
            normalized_rgbs = np.array(
                [self.rgb_normalizer(rgbs[0]), self.rgb_normalizer(rgbs[1])]
            )

            # Get Last channel coords as pixel score label Grasp Center
            grasp_mask = np.array([False, False, False, False])
            grasp_mask[self.GRIPPER_PCD_IDX["grasp_center"]] = True

            init_track_2d, _ = project_pcd_on_image(
                gripper_pcds[0],
                grasp_mask,
                rgbs[0],
                K_,
                color=(0, 255, 0),
                return_coords=True,
            )
            end_track_2d, pixel_score_vis = project_pcd_on_image(
                gripper_pcds[1],
                grasp_mask,
                rgbs[0],
                K_,
                color=(255, 255, 0),
                return_coords=True,
            )

            pixel_score_init = generate_heatmap_from_points(
                np.stack([init_track_2d] * 3, axis=0),
                np.array([self.target_shape, self.target_shape]),
            )
            pixel_score_end = generate_heatmap_from_points(
                np.stack([end_track_2d] * 3, axis=0),
                np.array([self.target_shape, self.target_shape]),
            )

            # For mse loss
            normalized_pixel_scores = np.array(
                [
                    np.transpose(
                        (pixel_score_init / 255).astype(np.float64), (2, 0, 1)
                    ),
                    np.transpose((pixel_score_end / 255).astype(np.float64), (2, 0, 1)),
                ]
            )  # Normalize to [0,1], H W 3 -> 3 H W

            pixel_score_label = np.zeros((self.target_shape, self.target_shape, 3))
            pixel_score_label[end_track_2d[:, 1], end_track_2d[:, 0], :] = 1.0

            pixel_dict["normalized_rgbs"] = normalized_rgbs  # 2 3 H W
            pixel_dict["normalized_pixel_scores"] = normalized_pixel_scores  # 2 3 H W
            pixel_dict["pixel_score_label"] = np.transpose(
                pixel_score_label, (2, 0, 1)
            ).astype(float)  # 3 H W
            pixel_dict["pixel_score_vis"] = pixel_score_vis  # H W 3
            pixel_dict["pixel_coord"] = end_track_2d.reshape(-1)

            item.update(pixel_dict)

        return item

    def __len__(self):
        # Return the length based on split indices
        return len(self.split_indices)


class RpadLeRobotDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size,
        val_batch_size,
        num_workers,
        dataset_cfg,
        seed,
        val_episode_ratio=0.1,
    ):
        super().__init__(batch_size, val_batch_size, num_workers, dataset_cfg, seed)
        self.val_tags = ["lerobot"]
        # Subset of train to use for eval
        self.TRAIN_SUBSET_SIZE = 20
        self.val_episode_ratio = val_episode_ratio
        self.train_indices = None
        self.val_indices = None

    def _generate_episode_splits(self):
        """Generate train/val splits based on episodes using LeRobot's episode_data_index."""
        # Handle multiple datasets
        if isinstance(self.dataset_cfg.repo_id, str):
            # Load metadata to get episode information
            temp_meta = LeRobotDatasetMetadata(
                repo_id=self.dataset_cfg.repo_id, root=self.root
            )

            # Get episode data index which maps episodes to their frame ranges
            episode_data_index = get_episode_data_index(temp_meta.episodes)
            # Get all episode indices
            episode_list = list(temp_meta.episodes.keys())
        else:
            tmp_multi_dataset = make_dataset(self.dataset_cfg.repo_id, self.root)
            episode_data_index = tmp_multi_dataset.episode_data_index
            episode_list = list(range(tmp_multi_dataset.num_episodes))

        # Sample episodes for validation
        num_val_episodes = max(1, int(len(episode_list) * self.val_episode_ratio))
        val_episodes = random.sample(episode_list, num_val_episodes)
        train_episodes = [ep for ep in episode_list if ep not in val_episodes]

        # Get frame indices for train and val episodes using episode_data_index
        train_indices = []
        val_indices = []
        test_indices = {}

        for ep_idx in train_episodes:
            start_frame = episode_data_index["from"][ep_idx].item()
            end_frame = episode_data_index["to"][ep_idx].item()
            train_indices.extend(range(start_frame, end_frame))

        for ep_idx in val_episodes:
            start_frame = episode_data_index["from"][ep_idx].item()
            end_frame = episode_data_index["to"][ep_idx].item()
            val_indices.extend(range(start_frame, end_frame))
            test_indices[ep_idx] = val_indices

        return sorted(train_indices), sorted(val_indices), test_indices

    def setup(self, stage: str = "fit"):
        self.stage = stage
        self.val_datasets = {}
        self.test_datasets = {}

        self.train_indices, self.val_indices, self.test_indices = (
            self._generate_episode_splits()
        )

        self.train_dataset = RpadLeRobotDataset(
            dataset_cfg=self.dataset_cfg,
            root=self.root,
            split="train",
            split_indices=self.train_indices,
        )
        for tag in self.val_tags:
            dataset_cfg = self.dataset_cfg.copy()
            test_datasets = {}
            self.val_datasets[tag] = RpadLeRobotDataset(
                dataset_cfg=dataset_cfg,
                root=self.root,
                split="val",
                split_indices=self.val_indices,
            )

            for ep_id, indices in self.test_indices.items():
                test_datasets[ep_id] = RpadLeRobotDataset(
                    dataset_cfg=dataset_cfg,
                    root=self.root,
                    split="test",
                    split_indices=indices,
                )

            self.test_datasets[tag] = test_datasets

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
                for ep_id, test_dataset in self.test_datasets[tag].items():
                    test_dataset.cache(
                        td.cachers.HDF5(
                            Path(self.train_dataset.cache_dir) / f"test_{tag}"
                        )
                    )

    def test_dataloader(self):
        if not hasattr(self, "test_datasets"):
            raise AttributeError(
                "test_datasets has not been set. Make sure to call setup() first."
            )
        return {
            tag: {
                id: data.DataLoader(
                    episode,
                    batch_size=self.val_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=collate_pcd_fn,
                )
                for id, episode in dataset.items()
            }
            for tag, dataset in self.test_datasets.items()
        }


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
