import os
import pathlib
import warnings
from typing import Dict, List, Sequence, Union, cast

import torch
import torch.utils._pytree as pytree
import wandb
from lightning.pytorch import Callback
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from lfd3d.datasets.droid.droid_dataset import DroidDataModule
from lfd3d.datasets.genGoalGen_dataset import GenGoalGenDataModule
from lfd3d.datasets.hoi4d.hoi4d_dataset import HOI4DDataModule
from lfd3d.datasets.multi_dataset import MultiDatasetDataModule
from lfd3d.datasets.rt1_dataset import RT1DataModule
from lfd3d.datasets.synth_block_dataset import SynthBlockDataModule
from lfd3d.models.diptv3 import DiPTv3, DiPTv3Adapter
from lfd3d.models.tax3d import (
    CrossDisplacementModule,
    DiffusionTransformerNetwork,
    SceneDisplacementModule,
)

PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent.parent.parent.resolve())


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
    else:
        raise NotImplementedError(cfg.model.name)

    # create network and model
    network = network_fn(model_cfg=cfg.model)
    model = module_fn(network=network, cfg=cfg)

    return network, model


def create_datamodule(cfg):
    # check that dataset and model types are compatible
    if cfg.model.type != cfg.dataset.type:
        raise ValueError(
            f"Model type: '{cfg.model.type}' and dataset type: '{cfg.dataset.type}' are incompatible."
        )

    # check dataset name
    elif cfg.dataset.name == "hoi4d":
        datamodule_fn = HOI4DDataModule
    elif cfg.dataset.name == "droid":
        datamodule_fn = DroidDataModule
    elif cfg.dataset.name == "rt1":
        datamodule_fn = RT1DataModule
    elif cfg.dataset.name == "synth_block":
        datamodule_fn = SynthBlockDataModule
    elif cfg.dataset.name == "multi":
        datamodule_fn = MultiDatasetDataModule
    elif cfg.dataset.name == "genGoalGen":
        datamodule_fn = GenGoalGenDataModule
    else:
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


class LogPredictionSamplesCallback(Callback):
    def __init__(self, logger: WandbLogger):
        self.logger = logger

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            outs = outputs["preds"][:n].argmax(dim=1)
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outs)
            ]

            # Option 1: log images with `WandbLogger.log_image`
            self.logger.log_image(key="sample_images", images=images, caption=captions)

            # Option 2: log images and predictions as a W&B Table
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred]
                for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outs))
            ]
            self.logger.log_table(key="sample_table", columns=columns, data=data)


class CustomModelPlotsCallback(Callback):
    def __init__(self, logger: WandbLogger):
        self.logger = logger

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""
        # assert trainer.logger is not None and isinstance(
        #     trainer.logger, WandbLogger
        # ), "This callback only works with WandbLogger."
        plots = pl_module.make_plots()
        trainer.logger.experiment.log(
            {
                "mode_distribution": plots["mode_distribution"],
            },
            step=trainer.global_step,
        )


def load_checkpoint_config_from_wandb(
    current_cfg, task_overrides, entity, project, run_id
):
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

    # Keep some overrides
    current_data_dir = current_cfg.dataset.data_dir
    current_cache_dir = current_cfg.dataset.cache_dir
    current_dataset_name = current_cfg.dataset.name

    # update run config with dataset and model configs from original run config
    for key in ["dataset", "model", "name"]:
        OmegaConf.update(
            current_cfg,
            key,
            OmegaConf.select(run_cfg, key),
            merge=True,
            force_add=True,
        )

    # small edge case - if 'eval', ignore 'train_size'/'val_size'
    if current_cfg.mode == "eval":
        current_cfg.dataset.train_size = None
        current_cfg.dataset.val_size = None
    current_cfg.dataset.data_dir = current_data_dir
    current_cfg.dataset.cache_dir = current_cache_dir
    current_cfg.dataset.name = current_dataset_name
    return current_cfg
