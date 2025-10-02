import os
import pathlib
import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import Dict, List, Union, cast

import torch
import torch.utils._pytree as pytree
import wandb
from lfd3d.datasets import (
    DroidDataModule,
    GenGoalGenDataModule,
    HOI4DDataModule,
    MultiDatasetDataModule,
    RpadFoxgloveDataModule,
    RT1DataModule,
    SynthBlockDataModule,
)
from lfd3d.datasets.lerobot.lerobot_dataset import RpadLeRobotDataModule
from lfd3d.models.articubot import ArticubotNetwork, GoalRegressionModule
from lfd3d.models.dino_3dgp import Dino3DGPGoalRegressionModule, Dino3DGPNetwork
from lfd3d.models.dino_heatmap import DinoHeatmapNetwork, HeatmapSamplerModule
from lfd3d.models.diptv3 import DiPTv3, DiPTv3Adapter
from lfd3d.models.tax3d import (
    CrossDisplacementModule,
    DiffusionTransformerNetwork,
    SceneDisplacementModule,
)
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent.parent.parent.resolve())


class ModelCheckpointExplicit(ModelCheckpoint):
    """
    Custom callback to save model with a specific name.
    Useful if we're logging models for multiple (non-dominating) metrics
    """

    def __init__(self, artifact_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.artifact_name = artifact_name

    def on_train_end(self, trainer, pl_module):
        if self.best_model_path:
            artifact = wandb.Artifact(self.artifact_name, type="model")
            artifact.add_file(local_path=self.best_model_path, name="model.ckpt")
            wandb.log_artifact(artifact, aliases=["latest", "best"])


def create_model(cfg):
    if cfg.model.name == "df_base":
        network_fn = DiffusionTransformerNetwork
        # module_fn = SceneDisplacementTrainingModule
        module_fn = SceneDisplacementModule
    elif cfg.model.name == "df_cross":
        network_fn = DiffusionTransformerNetwork
        # module_fn = Tax3dModule
        module_fn = CrossDisplacementModule
    elif cfg.model.name == "diptv3_cross":
        network_fn = lambda model_cfg: DiPTv3Adapter(
            DiPTv3(in_channels=40), final_dimension=6
        )
        module_fn = CrossDisplacementModule
    elif cfg.model.name == "articubot":
        network_fn = ArticubotNetwork
        module_fn = GoalRegressionModule
    elif cfg.model.name == "dino_heatmap":
        network_fn = DinoHeatmapNetwork
        module_fn = HeatmapSamplerModule
    elif cfg.model.name == "dino_3dgp":
        network_fn = Dino3DGPNetwork
        module_fn = Dino3DGPGoalRegressionModule
    else:
        raise NotImplementedError(cfg.model.name)

    # create network and model
    network = network_fn(model_cfg=cfg.model)
    model = module_fn(network=network, cfg=cfg)

    return network, model


def create_datamodule(cfg):
    dataset_map = {
        "hoi4d": HOI4DDataModule,
        "droid": DroidDataModule,
        "rt1": RT1DataModule,
        "synth_block": SynthBlockDataModule,
        "multi": MultiDatasetDataModule,
        "genGoalGen": GenGoalGenDataModule,
        "rpadFoxglove": RpadFoxgloveDataModule,
        "rpadLerobot": RpadLeRobotDataModule,
        "liberoLerobot": RpadLeRobotDataModule,  # same module, just different default configs
    }

    datamodule_fn = dataset_map.get(cfg.dataset.name)
    if datamodule_fn is None:
        raise ValueError(f"Invalid dataset name: {cfg.dataset.name}")

    # job-specific datamodule pre-processing
    if cfg.mode == "eval":
        job_cfg = cfg.inference
        stage = "predict"
    elif cfg.mode == "train":
        job_cfg = cfg.training
        stage = "fit"
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")

    # setting up datamodule
    datamodule = datamodule_fn(
        batch_size=job_cfg.batch_size,
        val_batch_size=job_cfg.val_batch_size,
        num_workers=cfg.resources.num_workers,
        dataset_cfg=cfg.dataset,
        seed=cfg.seed,
    )
    datamodule.setup(stage)

    # training-specific job config setup
    if cfg.mode == "train":
        job_cfg.num_training_steps = len(datamodule.train_dataloader()) * job_cfg.epochs

    return cfg, datamodule


# This matching function
def match_fn(dirs: Sequence[str], extensions: Sequence[str], root: str = PROJECT_ROOT):
    def _match_fn(path: pathlib.Path):
        in_dir = any([str(path).startswith(os.path.join(root, d)) for d in dirs])

        if not in_dir:
            return False

        if not any([str(path).endswith(e) for e in extensions]):
            return False

        return True

    return _match_fn


TorchTree = Dict[str, Union[torch.Tensor, "TorchTree"]]


def flatten_outputs(outputs: List[TorchTree]) -> TorchTree:
    """Flatten a list of dictionaries into a single dictionary."""

    # Concatenate all leaf nodes in the trees.
    flattened_outputs = [pytree.tree_flatten(output) for output in outputs]
    flattened_list = [o[0] for o in flattened_outputs]
    flattened_spec = flattened_outputs[0][1]  # Spec definitely should be the same...
    cat_flat = [torch.cat(x) for x in list(zip(*flattened_list))]
    output_dict = pytree.tree_unflatten(cat_flat, flattened_spec)
    return cast(TorchTree, output_dict)


def load_checkpoint_config_from_wandb(
    current_cfg,
    task_overrides,
    entity,
    project,
    run_id,
    keys_to_preserve=[
        "dataset.data_dir",
        "dataset.cache_dir",
        "dataset.name",
        "dataset.additional_img_dir",
        "dataset.use_intermediate_frames",
        "dataset.data_sources",
    ],
):
    """
    Load and merge config from WandB, with controlled updates and overrides.
    A bit hacky, since we need to override some keys but not others :/
    """
    # grab run config from wandb
    api = wandb.Api()
    run_cfg = OmegaConf.create(api.run(f"{entity}/{project}/{run_id}").config)

    # check for consistency between task overrides and original run config
    inconsistent_keys = []
    for ovrd in task_overrides:
        key = ovrd.split("=")[0]
        if OmegaConf.select(current_cfg, key) != OmegaConf.select(run_cfg, key):
            inconsistent_keys.append(key)

    if inconsistent_keys:
        warnings.warn(
            f"Task overrides are inconsistent with original run config: {inconsistent_keys}",
            UserWarning,
        )

    # Store values to preserve before merging
    preserve_values = {
        key: deepcopy(OmegaConf.select(current_cfg, key)) for key in keys_to_preserve
    }

    # update run config with dataset and model configs from original run config
    for key in ["dataset", "model", "name"]:
        OmegaConf.update(
            current_cfg,
            key,
            OmegaConf.select(run_cfg, key),
            merge=True,
            force_add=True,
        )

    # Restore preserved values
    for key, value in preserve_values.items():
        OmegaConf.update(current_cfg, key, value, force_add=True)

    # small edge case - if 'eval', ignore 'train_size'/'val_size'
    if current_cfg.mode == "eval":
        current_cfg.dataset.train_size = None
        current_cfg.dataset.val_size = None
    return current_cfg
