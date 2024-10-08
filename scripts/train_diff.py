import json
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as L
import omegaconf
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# from non_rigid.datasets.microwave_flow import MicrowaveFlowDataModule
# from non_rigid.datasets.proc_cloth_flow import ProcClothFlowDataModule
# from non_rigid.datasets.rigid import RigidDataModule
# from non_rigid.models.df_base import (
#     DiffusionFlowBase,
#     FlowPredictionTrainingModule,
#     PointPredictionTrainingModule,
# )
# from non_rigid.models.regression import (
#     LinearRegression,
#     LinearRegressionTrainingModule,
# )
from non_rigid.utils.script_utils import (
    PROJECT_ROOT,
    CustomModelPlotsCallback,
    LogPredictionSamplesCallback,
    create_model,
    create_datamodule,
    match_fn,
)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    print(
        json.dumps(
            omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            sort_keys=True,
            indent=4,
        )
    )

    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    # Global seed for reproducibility.
    L.seed_everything(cfg.seed)

    ######################################################################
    # Create the datamodule.
    # The datamodule is responsible for all the data loading, including
    # downloading the data, and splitting it into train/val/test.
    #
    # This could be swapped out for a different datamodule in-place,
    # or with an if statement, or by using hydra.instantiate.
    ######################################################################
    cfg, datamodule = create_datamodule(cfg)

    ######################################################################
    # Create the network(s) which will be trained by the Training Module.
    # The network should (ideally) be lightning-independent. This allows
    # us to use the network in other projects, or in other training
    # configurations. Also create the Training Module.
    #
    # This might get a bit more complicated if we have multiple networks,
    # but we can just customize the training module and the Hydra configs
    # to handle that case. No need to over-engineer it. You might
    # want to put this into a "create_network" function somewhere so train
    # and eval can be the same.
    #
    # If it's a custom network, a good idea is to put the custom network
    # in `python_ml_project_template.nets.my_net`.
    ######################################################################

    # Model architecture is dataset-dependent, so we have a helper
    # function to create the model (while separating out relevant vals).
    network, model = create_model(cfg)


    # datamodule.setup(stage="fit")
    # cfg.training.num_training_steps = (
    #     len(datamodule.train_dataloader()) * cfg.training.epochs
    # )
    # # updating the training sample size
    # # cfg.training.training_sample_size = cfg.dataset.sample_size

    # TODO: compiling model doesn't work with lightning out of the box?
    # model = torch.compile(model)

    ######################################################################
    # Set up logging in WandB.
    # This is a bit complicated, because we want to log the codebase,
    # the model, and the checkpoints.
    ######################################################################

    # If no group is provided, then we should create a new one (so we can allocate)
    # evaluations to this group later.
    if cfg.wandb.group is None:
        id = wandb.util.generate_id()
        group = "experiment-" + id
    else:
        group = cfg.wandb.group

    logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        log_model=True,  # Only log the last checkpoint to wandb, and only the LAST model checkpoint.
        save_dir=cfg.wandb.save_dir,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        job_type=cfg.job_type,
        save_code=True,  # This just has the main script.
        group=group,
    )

    ######################################################################
    # Create the trainer.
    # The trainer is responsible for running the training loop, and
    # logging the results.
    #
    # There are a few callbacks (which we could customize):
    # - LogPredictionSamplesCallback: Logs some examples from the dataset,
    #       and the model's predictions.
    # - ModelCheckpoint #1: Saves the latest model.
    # - ModelCheckpoint #2: Saves the best model (according to validation
    #       loss), and logs it to wandb.
    ######################################################################

    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.resources.gpus,
        # precision="16-mixed",
        precision="32-true",
        max_epochs=cfg.training.epochs,
        logger=logger,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epochs,
        # log_every_n_steps=2, # TODO: MOVE THIS TO TRAINING CFG
        log_every_n_steps=len(datamodule.train_dataloader()),
        gradient_clip_val=cfg.training.grad_clip_norm,
        callbacks=[
            # Callback which logs whatever visuals (i.e. dataset examples, preds, etc.) we want.
            # LogPredictionSamplesCallback(logger),
            # This checkpoint callback saves the latest model during training, i.e. so we can resume if it crashes.
            # It saves everything, and you can load by referencing last.ckpt.
            # CustomModelPlotsCallback(logger),
            ModelCheckpoint(
                dirpath=cfg.lightning.checkpoint_dir,
                filename="{epoch}-{step}",
                monitor="step",
                mode="max",
                save_weights_only=False,
                save_last=True,
            ),
            ModelCheckpoint(
                dirpath=cfg.lightning.checkpoint_dir,
                filename="{epoch}-{step}-{val_wta_rmse_0:.3f}",
                monitor="val_rmse_wta_0",
                mode="min",
                save_weights_only=False,
                save_last=False,
                # auto_insert_metric_name=False,
            ),
            # ModelCheckpoint(
            #     dirpath=cfg.lightning.checkpoint_dir,
            #     filename="{epoch}-{step}-{val_rmse_0:.3f}",
            #     monitor="val_rmse_0",
            #     mode="min",
            #     save_weights_only=False,
            #     save_last=False,
            # )
            # This checkpoint will get saved to WandB. The Callback mechanism in lightning is poorly designed, so we have to put it last.
            # ModelCheckpoint(
            #     dirpath=cfg.lightning.checkpoint_dir,
            #     filename="{epoch}-{step}-{val_loss:.2f}-weights-only",
            #     monitor="val_loss",
            #     mode="min",
            #     save_weights_only=True,
            # ),
        ],
        num_sanity_val_steps=0,
    )

    ######################################################################
    # Log the code to wandb.
    # This is somewhat custom, you'll have to edit this to include whatever
    # additional files you want, but basically it just logs all the files
    # in the project root inside dirs, and with extensions.
    ######################################################################

    # Log the code used to train the model. Make sure not to log too much, because it will be too big.
    wandb.run.log_code(
        root=PROJECT_ROOT,
        include_fn=match_fn(
            dirs=["configs", "scripts", "src"],
            extensions=[".py", ".yaml"],
        ),
    )

    ######################################################################
    # Train the model.
    ######################################################################

    # this might be a little too "pythonic"
    if cfg.checkpoint.run_id:
        print(
            "Attempting to resume training from checkpoint: ", cfg.checkpoint.reference
        )

        api = wandb.Api()
        artifact_dir = cfg.wandb.artifact_dir
        artifact = api.artifact(cfg.checkpoint.reference, type="model")
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
        # ckpt = torch.load(ckpt_file)
        # # model.load_state_dict(
        # #     {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
        # # )
        # model.load_state_dict(ckpt["state_dict"])
    else:
        print("Starting training from scratch.")
        ckpt_file = None

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_file)


if __name__ == "__main__":
    main()
