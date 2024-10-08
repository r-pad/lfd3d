import os
from functools import partial
import pathlib
from typing import Dict, List, Sequence, Union, cast

import torch
import torch.utils._pytree as pytree
import torchvision as tv
import wandb
from lightning.pytorch import Callback
from pytorch_lightning.loggers import WandbLogger


from non_rigid.models.df_base import (
    DiffusionFlowBase,
    FlowPredictionInferenceModule,
    FlowPredictionTrainingModule,
    PointPredictionInferenceModule,
    PointPredictionTrainingModule,
)
from non_rigid.models.regression import (
    LinearRegression,
    LinearRegressionInferenceModule,
    LinearRegressionTrainingModule,
    RegressionNetwork,
    RegressionModule,
)
from non_rigid.models.tax3d import (
    DiffusionTransformerNetwork,
    SceneDisplacementModule,
    CrossDisplacementModule,
)

from non_rigid.datasets.proc_cloth_flow import ProcClothFlowDataModule
from non_rigid.datasets.hoi4d import HOI4DDataModule


PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent.parent.parent.resolve())

    
def create_model_legacy(cfg):
    if cfg.model.name in ["df_base", "df_cross"]:
        network_fn = DiffusionFlowBase
        if cfg.mode == "train":
            # tax3d training modules
            if cfg.model.type == "flow":
                module_fn = partial(FlowPredictionTrainingModule, training_cfg=cfg.training)
            elif cfg.model.type == "point":
                module_fn = partial(PointPredictionTrainingModule, task_type=cfg.task_type, training_cfg=cfg.training)
            else:
                raise ValueError(f"Invalid model type: {cfg.model.type}")
        elif cfg.mode == "eval":
            # tax3d inference modules
            if cfg.model.type == "flow":
                module_fn = partial(FlowPredictionInferenceModule, inference_cfg=cfg.inference)
            elif cfg.model.type == "point":
                module_fn = partial(PointPredictionInferenceModule, task_type=cfg.task_type, inference_cfg=cfg.inference)
            else:
                raise ValueError(f"Invalid model type: {cfg.model.type}")
        else:
            raise ValueError(f"Invalid mode: {cfg.mode}")
    elif cfg.model.name == "linear_regression":
        assert cfg.model.type == "point", "Only point regression is supported."
        network_fn = LinearRegression
        if cfg.mode == "train":
            # linear training module
            module_fn = partial(LinearRegressionTrainingModule, training_cfg=cfg.training)
        elif cfg.mode == "eval":
            # linear inference module
            module_fn = partial(LinearRegressionInferenceModule, inference_cfg=cfg.inference)
        else:
            raise ValueError(f"Invalid mode: {cfg.mode}")
    else:
        raise ValueError(f"Invalid model name: {cfg.model.name}")
    # create network
    network = network_fn(model_cfg=cfg.model)
    model = module_fn(network=network, model_cfg=cfg.model)

    # TODO: this should also check for a checkpoint id, and setup the network
    return network, model


def create_model(cfg):
    if cfg.model.name == "df_base":
        network_fn = DiffusionTransformerNetwork
        # module_fn = SceneDisplacementTrainingModule
        module_fn = SceneDisplacementModule
    elif cfg.model.name == "df_cross":
        network_fn = DiffusionTransformerNetwork
        # module_fn = Tax3dModule
        module_fn = CrossDisplacementModule
    elif cfg.model.name == "regression":
        network_fn = RegressionNetwork
        module_fn = RegressionModule

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
    if cfg.dataset.name == "proc_cloth":
        datamodule_fn = ProcClothFlowDataModule
    elif cfg.dataset.name == "hoi4d":
        datamodule_fn = HOI4DDataModule
    else:
        raise ValueError(f"Invalid dataset name: {cfg.dataset.name}")

    # job-specific datamodule pre-processing
    if cfg.mode == "eval":
        job_cfg = cfg.inference
        # check for full action
        if job_cfg.action_full:
            cfg.dataset.sample_size_action = -1
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

    # updating job config sample sizes
    if cfg.dataset.scene:
        job_cfg.sample_size = cfg.dataset.sample_size_action + cfg.dataset.sample_size_anchor
    else:
        job_cfg.sample_size = cfg.dataset.sample_size_action
        job_cfg.sample_size_anchor = cfg.dataset.sample_size_anchor

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