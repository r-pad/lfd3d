"""This file adapts a LeRobot dataset to the LFD3D format."""

import random
from pathlib import Path

import numpy as np
import torch
import torchdatasets as td
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    MultiHomogeneousLeRobotDataset,
)
from lfd3d.datasets.base_data import BaseDataModule, BaseDataset
from lfd3d.datasets.rgb_text_featurizer import RGBTextFeaturizer
from lfd3d.utils.data_utils import collate_pcd_fn
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


def source_of_data(item):
    """
    Return the source of the current demo i.e. where the data is from
    """
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


class RpadLeRobotDataset(BaseDataset):
    def __init__(
        self,
        dataset_cfg,
        root: str | None = None,
        split: str = "train",
        split_indices: list = [],
        augment_train: str = "image_color_only",
        augment_cfg: dict = None,
    ):
        super().__init__(augment_train=augment_train, augment_cfg=augment_cfg)
        repo_id = dataset_cfg.repo_id

        self.lerobot_dataset = make_dataset(repo_id, root)

        # Store the same dataset configuration...
        self.cache_dir = dataset_cfg.cache_dir
        self.dataset_cfg = dataset_cfg
        self.num_points = dataset_cfg.num_points
        self.max_depth = dataset_cfg.max_depth
        self.split = split

        if len(split_indices) == 0:
            split_indices = list(range(len(self.lerobot_dataset)))
        self.split_indices = split_indices

        # Multi-camera configuration: first camera in list is primary
        self.cameras = dataset_cfg.cameras
        self.gripper_pcd_key = dataset_cfg.gripper_pcd_key

        # Validate augmentation mode for multi-camera
        if len(self.cameras) > 1:
            if augment_train == "image":
                raise ValueError(
                    "Multi-camera setup does not allow augment_train to be 'image'."
                    "Geometric augmentations (rotation, hflip) are not "
                    "compatible with world-frame ground truth."
                )

        self.rgb_text_featurizer = RGBTextFeaturizer(
            target_shape=self.target_shape, rgb_feat=self.dataset_cfg.rgb_feat
        )

        # indexes of selected gripper points -> handpicked
        self.GRIPPER_IDX = {
            "aloha": np.array([6, 197, 174]),
            "human": np.array([343, 763, 60]),
            "libero_franka": np.array(
                [0, 1, 2]
            ),  # gripper pcd in dataset: [left right top grasp-center] in agentview; (right gripper, left gripper, top, grasp-center)
        }

    def extract_goal(self, item):
        if self.dataset_cfg.use_subgoals:
            return item["subgoal"]
        else:
            return item["task"]

    def load_transition(self, idx):
        """Load a transition from the dataset.

        Returns:
            - all_rgbs: list of (2, H, W, 3) arrays, one per camera
            - all_depths: list of (2, H, W) arrays, one per camera
            - orig_shape: (H, W) from first camera
            - gripper_pcds: (2, N, 3) array
            - task: str
            - episode_index: str
            - cam_names: list of camera names
        """
        start_item = self.lerobot_dataset[idx]
        task = self.extract_goal(start_item)
        episode_index = start_item["episode_index"]
        # The next_event_idx is relative to the episode, so we calculate the absolute index
        end_idx = (
            start_item["next_event_idx"] - start_item["frame_index"] + idx
        ).item()

        end_item = self.lerobot_dataset[end_idx]

        all_rgbs = []
        all_depths = []
        cam_names = []
        orig_shape = None

        for cam_cfg in self.cameras:
            COLOR_KEY = cam_cfg.color_key
            DEPTH_KEY = cam_cfg.depth_key

            # Load RGB for this camera
            rgb_init = (start_item[COLOR_KEY].permute(1, 2, 0).numpy() * 255).astype(
                np.uint8
            )
            if orig_shape is None:
                orig_shape = rgb_init.shape[:2]

            rgb_init = Image.fromarray(rgb_init)
            rgb_init = np.asarray(self.rgb_preprocess(rgb_init))
            rgb_end = (end_item[COLOR_KEY].permute(1, 2, 0).numpy() * 255).astype(
                np.uint8
            )
            rgb_end = Image.fromarray(rgb_end)
            rgb_end = np.asarray(self.rgb_preprocess(rgb_end))
            rgbs = np.array([rgb_init, rgb_end])

            # Load depth for this camera
            depth_init = Image.fromarray(start_item[DEPTH_KEY].numpy()[0])
            depth_init = np.asarray(self.depth_preprocess(depth_init))
            depth_end = Image.fromarray(end_item[DEPTH_KEY].numpy()[0])
            depth_end = np.asarray(self.depth_preprocess(depth_end))
            depths = np.array([depth_init, depth_end])

            all_rgbs.append(rgbs)
            all_depths.append(depths)
            cam_names.append(cam_cfg.name)

        GRIPPER_PCD_KEY = self.gripper_pcd_key
        gripper_pcd_init = start_item[GRIPPER_PCD_KEY]
        gripper_pcd_end = end_item[GRIPPER_PCD_KEY]
        gripper_pcds = np.array([gripper_pcd_init, gripper_pcd_end])

        return (
            all_rgbs,
            all_depths,
            orig_shape,
            gripper_pcds,
            task,
            f"{episode_index}",
            cam_names,
        )

    def _load_camera_intrinsics(self, intrinsics_path, data_source):
        """Load camera intrinsics from file.

        Args:
            intrinsics_path: Relative path to intrinsics file (e.g., "aloha_calibration/intrinsics_xxx.txt")
            data_source: Data source name (e.g., 'aloha', 'human', 'libero_franka')

        Returns:
            np.ndarray: 3x3 intrinsics matrix
        """
        file_path = Path(__file__).parent.parent / intrinsics_path
        return np.loadtxt(file_path)

    def _load_camera_extrinsics(self, extrinsics_path, data_source):
        """Load camera extrinsics (T_world_from_camera) from file.

        Args:
            extrinsics_path: Relative path to extrinsics file (e.g., "aloha_calibration/T_world_from_camera_xxx.txt")
            data_source: Data source name (e.g., 'aloha', 'human', 'libero_franka')

        Returns:
            np.ndarray: 4x4 transformation matrix (T_world_from_camera)
        """
        file_path = Path(__file__).parent.parent / extrinsics_path
        T = np.loadtxt(file_path).astype(np.float32)
        return T.reshape(4, 4)

    def _transform_to_world_frame(self, points_cam, T_world_from_cam):
        """Transform points from camera frame to world frame.

        Args:
            points_cam: (N, 3) array of points in camera frame
            T_world_from_cam: (4, 4) transformation matrix

        Returns:
            (N, 3) array of points in world frame
        """
        # Convert to homogeneous coordinates
        N = points_cam.shape[0]
        points_hom = np.concatenate([points_cam, np.ones((N, 1))], axis=1)  # (N, 4)

        # Apply transformation
        points_world_hom = (T_world_from_cam @ points_hom.T).T  # (N, 4)

        # Convert back to 3D
        points_world = points_world_hom[:, :3]  # (N, 3)

        return points_world

    def __getitem__(self, index):
        # Map the dataset index to the actual LeRobot dataset index using split_indices
        actual_index = self.split_indices[index]
        data_source = source_of_data(self.lerobot_dataset[actual_index])

        start2end = torch.eye(4)  # Static camera

        # Retrieve the item from the underlying LeRobot dataset
        (
            all_rgbs,
            all_depths,
            orig_shape,
            gripper_pcds,
            caption,
            demo_name,
            cam_names,
        ) = self.load_transition(actual_index)

        # Load intrinsics and extrinsics for all cameras
        all_intrinsics = []
        all_extrinsics = []
        for cam_cfg in self.cameras:
            # Load intrinsics
            K = self._load_camera_intrinsics(cam_cfg.intrinsics, data_source)
            K_scaled = BaseDataset.get_scaled_intrinsics(
                K, orig_shape, self.target_shape
            )
            all_intrinsics.append(K_scaled)

            # Load extrinsics
            T = self._load_camera_extrinsics(cam_cfg.extrinsics, data_source)
            all_extrinsics.append(T)

        # Gripper tracks
        start_tracks, end_tracks = gripper_pcds[0], gripper_pcds[1]
        actual_caption = caption

        # Apply augmentation to all cameras
        # Note: Each camera gets independent random augmentations
        all_rgbs_aug = []
        all_depths_aug = []
        all_intrinsics_aug = []
        for i, (rgbs, depths, K) in enumerate(
            zip(all_rgbs, all_depths, all_intrinsics)
        ):
            # For image_color_only mode, tracks are not transformed (they're in world frame)
            rgbs_aug, depths_aug, start_tracks, end_tracks, K_aug = (
                self.apply_image_augmentation(
                    rgbs, depths, start_tracks, end_tracks, K, self.augment_cfg
                )
            )
            all_rgbs_aug.append(rgbs_aug)
            all_depths_aug.append(depths_aug)
            all_intrinsics_aug.append(K_aug)

        # Primary camera is first in list
        primary_rgbs = all_rgbs_aug[0]
        primary_depths = all_depths_aug[0]
        primary_K = all_intrinsics_aug[0]
        primary_T = all_extrinsics[0]

        # Compute scene PCD from primary camera
        rgb_embed, text_embed = self.rgb_text_featurizer.compute_rgb_text_feat(
            primary_rgbs[0], caption
        )
        start_scene_pcd, start_scene_feat_pcd, augment_tf = self.get_scene_pcd(
            rgb_embed, primary_depths[0], primary_K, self.num_points, self.max_depth
        )

        # Transform scene PCD from camera frame to world frame
        # start_scene_pcd is (N, 3) in camera frame, transform using primary extrinsics
        start_scene_pcd_world = self._transform_to_world_frame(
            start_scene_pcd, primary_T
        )

        gripper_idx = self.GRIPPER_IDX[data_source]

        # Normalize in world frame
        action_pcd_mean, scene_pcd_std = self.get_normalize_mean_std(
            start_tracks, start_scene_pcd_world, self.dataset_cfg
        )
        start_tracks, end_tracks, start_scene_pcd_world = self.transform_pcds(
            start_tracks,
            end_tracks,
            start_scene_pcd_world,
            action_pcd_mean,
            scene_pcd_std,
            augment_tf,
        )

        # collate_pcd_fn handles batching of the point clouds
        # Prepare auxiliary camera data
        if len(self.cameras) > 1:
            aux_rgbs = np.stack(all_rgbs_aug[1:], axis=0)  # (num_aux, 2, H, W, 3)
            aux_depths = np.stack(all_depths_aug[1:], axis=0)  # (num_aux, 2, H, W)
            aux_intrinsics = np.stack(all_intrinsics_aug[1:], axis=0)  # (num_aux, 3, 3)
            aux_extrinsics = np.stack(all_extrinsics[1:], axis=0)  # (num_aux, 4, 4)
        else:
            # No auxiliary cameras
            aux_rgbs = np.zeros((0, 2, *self.target_shape, 3), dtype=np.uint8)
            aux_depths = np.zeros((0, 2, *self.target_shape), dtype=np.float32)
            aux_intrinsics = np.zeros((0, 3, 3), dtype=np.float32)
            aux_extrinsics = np.zeros((0, 4, 4), dtype=np.float32)

        item = {
            # Primary camera data
            "rgbs": primary_rgbs,
            "depths": primary_depths,
            "intrinsics": primary_K,
            "extrinsics": primary_T,
            # Auxiliary camera data
            "aux_rgbs": aux_rgbs,
            "aux_depths": aux_depths,
            "aux_intrinsics": aux_intrinsics,
            "aux_extrinsics": aux_extrinsics,
            # Point clouds (in world frame)
            "action_pcd": start_tracks,
            "anchor_pcd": start_scene_pcd_world,
            "anchor_feat_pcd": start_scene_feat_pcd,
            # Labels
            "cross_displacement": end_tracks - start_tracks,
            # Text/metadata
            "caption": caption,
            "text_embed": text_embed,
            "actual_caption": actual_caption,
            # Transforms/normalization
            "start2end": start2end,
            "pcd_mean": action_pcd_mean,
            "pcd_std": scene_pcd_std,
            "augment_R": augment_tf["R"],
            "augment_t": augment_tf["t"],
            "augment_C": augment_tf["C"],
            # Other
            "gripper_idx": gripper_idx,
            "vid_name": demo_name,
            "data_source": data_source,
        }
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
        augment_train="image",
        augment_cfg=None,
    ):
        super().__init__(
            batch_size,
            val_batch_size,
            num_workers,
            dataset_cfg,
            seed,
            augment_train,
            augment_cfg,
        )
        self.val_tags = []  # populated in _generate_episode_splits
        # Subset of train to use for eval
        self.TRAIN_SUBSET_SIZE = 20
        self.val_episode_ratio = val_episode_ratio
        self.train_indices = None
        self.val_indices = None

    def _generate_episode_splits(self):
        """Generate train/val splits based on episodes and data sources using LeRobot's episode_data_index."""
        tmp_dataset = make_dataset(self.dataset_cfg.repo_id, self.root)
        episode_data_index = tmp_dataset.episode_data_index
        episode_list = list(range(tmp_dataset.num_episodes))

        # Group episodes by data source
        episodes_by_source = {}
        for ep_idx in episode_list:
            # Get first frame of episode to determine data source
            start_frame = episode_data_index["from"][ep_idx].item()
            data_source = source_of_data(tmp_dataset[start_frame])

            if data_source not in episodes_by_source:
                episodes_by_source[data_source] = []
                self.val_tags.append(data_source)
            episodes_by_source[data_source].append(ep_idx)

        # Split each data source independently
        train_episodes_by_source = {}
        val_episodes_by_source = {}

        for data_source, episodes in episodes_by_source.items():
            num_val_episodes = max(1, int(len(episodes) * self.val_episode_ratio))
            val_episodes = random.sample(episodes, num_val_episodes)
            train_episodes = [ep for ep in episodes if ep not in val_episodes]

            train_episodes_by_source[data_source] = train_episodes
            val_episodes_by_source[data_source] = val_episodes

        # Collect train indices (concatenated from all sources)
        train_indices = []
        for data_source, episodes in train_episodes_by_source.items():
            for ep_idx in episodes:
                start_frame = episode_data_index["from"][ep_idx].item()
                end_frame = episode_data_index["to"][ep_idx].item()
                train_indices.extend(range(start_frame, end_frame))

        # Collect val indices by source (kept separate for independent validation)
        val_indices_dict = {}
        test_indices_dict = {}
        for data_source, episodes in val_episodes_by_source.items():
            val_indices_dict[data_source] = []
            test_indices_dict[data_source] = {}
            for ep_idx in episodes:
                start_frame = episode_data_index["from"][ep_idx].item()
                end_frame = episode_data_index["to"][ep_idx].item()
                val_indices_dict[data_source].extend(range(start_frame, end_frame))
                test_indices_dict[data_source][ep_idx] = range(start_frame, end_frame)

        return train_indices, val_indices_dict, test_indices_dict

    def setup(self, stage: str = "fit"):
        self.stage = stage
        self.val_datasets = {}
        self.test_datasets = {}

        self.train_indices, self.val_indices_dict, self.test_indices_dict = (
            self._generate_episode_splits()
        )
        self.train_dataset = RpadLeRobotDataset(
            dataset_cfg=self.dataset_cfg,
            root=self.root,
            split="train",
            split_indices=self.train_indices,
            augment_train=self.augment_train,
            augment_cfg=self.augment_cfg,
        )
        for tag in self.val_tags:
            val_indices_for_tag = self.val_indices_dict.get(tag, [])
            if len(val_indices_for_tag) == 0:
                continue

            dataset_cfg = self.dataset_cfg.copy()
            self.val_datasets[tag] = RpadLeRobotDataset(
                dataset_cfg=dataset_cfg,
                root=self.root,
                split="val",
                split_indices=val_indices_for_tag,
                augment_train=None,  # Never augment validation
                augment_cfg=self.augment_cfg,
            )
            # Store metadata for lazy test dataset creation
            self.test_datasets[tag] = {}

        if self.train_dataset.cache_dir:
            invalidating_cacher = td.cachers.ProbabilisticCacherWrapper(
                td.cachers.HDF5(Path(self.train_dataset.cache_dir)),
                invalidation_rate=self.dataset_cfg.cache_invalidation_rate,
                seed=self.seed,
            )
            self.train_dataset.cache(invalidating_cacher)
            for tag in self.val_datasets:
                self.val_datasets[tag].cache(
                    td.cachers.HDF5(Path(self.train_dataset.cache_dir) / f"val_{tag}")
                )

    def _create_test_dataset(self, ep_id, indices):
        """Lazily create a test dataset for a single episode."""
        dataset_cfg = self.dataset_cfg.copy()
        test_dataset = RpadLeRobotDataset(
            dataset_cfg=dataset_cfg,
            root=self.root,
            split="test",
            split_indices=indices,
        )
        if self.train_dataset.cache_dir:
            # Get the tag for this episode
            tmp_dataset = make_dataset(self.dataset_cfg.repo_id, self.root)
            start_frame = indices[0]
            data_source = source_of_data(tmp_dataset[start_frame])
            test_dataset.cache(
                td.cachers.HDF5(
                    Path(self.train_dataset.cache_dir) / f"test_{data_source}"
                )
            )
        return test_dataset

    def test_dataloader(self):
        if not hasattr(self, "test_datasets"):
            raise AttributeError(
                "test_datasets has not been set. Make sure to call setup() first."
            )

        # Lazily create test datasets only when test_dataloader is called
        test_dataloaders = {}
        for tag in self.test_datasets.keys():
            test_dataloaders[tag] = {}
            for ep_id, indices in self.test_indices_dict[tag].items():
                # Create dataset on-demand
                episode_dataset = self._create_test_dataset(ep_id, list(indices))
                test_dataloaders[tag][ep_id] = data.DataLoader(
                    episode_dataset,
                    batch_size=self.val_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=collate_pcd_fn,
                )

        return test_dataloaders


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg_dict = {
        "dataset": {
            "name": "rpadLerobot",
            "cache_dir": None,
            "cache_invalidation_rate": 0.05,
            "train_size": None,
            "val_size": None,
            "num_points": 8192,
            "max_depth": 1.5,
            "normalize": False,
            "augment_train": None,
            "augment_cfg": {
                "augment_prob": 0.75,
                "augment_transform": True,
                "pcd_sample": ["voxel", "fps"],
                "fps_num_points": [4096, 8192],
                "voxel_size": [0.01, 0.02],
            },
            "color_key": "observation.images.cam_azure_kinect.color",
            "depth_key": "observation.images.cam_azure_kinect.transformed_depth",
            "gripper_pcd_key": "observation.points.gripper_pcds",
            "data_name": "0627",
            "additional_img_dir": None,
            "data_sources": ["aloha"],
            "oversample_robot": True,
            "rgb_feat": False,
            "use_full_text": True,
            "use_intermediate_frames": False,
            "use_subgoals": False,
            "repo_id": [
                "sriramsk/fold_onesie_20250831_subsampled_heatmapGoal",
                "sriramsk/fold_shirt_20250918_subsampled_heatmapGoal",
                "sriramsk/fold_towel_20250919_subsampled_heatmapGoal",
                "sriramsk/fold_bottoms_20250919_human_heatmapGoal",
            ],
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
    lr_dset = RpadLeRobotDataset(dataset_cfg=cfg.dataset, augment_train=None)
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
