import json

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from lfd3d.utils.lora_utils import apply_lora
from lfd3d.utils.script_utils import (
    PROJECT_ROOT,
    ModelCheckpointExplicit,
    create_datamodule,
    create_model,
    match_fn,
)
from pytorch_lightning.loggers import WandbLogger


def create_checkpoint_callbacks(cfg, experiment_id):
    """Create ModelCheckpoint callbacks based on config."""
    callbacks = []

    # Get checkpoint configs from config file, with defaults
    checkpoint_configs = cfg.training.checkpoints

    for name, config in checkpoint_configs.items():
        monitor = config["monitor"]
        mode = config.get("mode", "min")

        # Extract metric name for filename (remove val/ prefix if present)
        metric_name = monitor.replace("val/", "")

        callback = ModelCheckpointExplicit(
            artifact_name=f"best_{name}_model-{experiment_id}",
            dirpath=cfg.lightning.checkpoint_dir,
            filename=name + "{epoch}-{step}-{" + monitor + ":.3f}",
            monitor=monitor,
            mode=mode,
            save_weights_only=False,
            save_last=True,
        )
        callbacks.append(callback)

    return callbacks


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_float32_matmul_precision("highest")

    # Global seed for reproducibility.
    pl.seed_everything(cfg.seed)

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
    # configurations.
    #
    # This might get a bit more complicated if we have multiple networks,
    # but we can just customize the training module and the Hydra configs
    # to handle that case. No need to over-engineer it. You might
    # want to put this into a "create_network" function somewhere so train
    # and eval can be the same.
    #
    # If it's a custom network, a good idea is to put the custom network
    # in `lfd3d.nets.my_net`.
    ######################################################################

    # Model architecture is dataset-dependent, so we have a helper
    # function to create the model (while separating out relevant vals).
    ######################################################################
    # Create the training module.
    # The training module is responsible for all the different parts of
    # training, including the network, the optimizer, the loss function,
    # and the logging.
    ######################################################################
    network, model = create_model(cfg)

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
        log_model=False,  # handled explicitly by the callbacks
        save_dir=cfg.wandb.save_dir,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        job_type=cfg.job_type,
        save_code=True,  # This just has the main script.
        group=group,
        name=cfg.wandb.name,
    )

    ######################################################################
    # Create the trainer.
    # The trainer is responsible for running the training loop, and
    # logging the results.
    ######################################################################

    # For multi-dataset training, we define our own distributed sampler
    # to handle issues described in src/lfd3d/datasets/multi_dataset.py
    use_distributed_sampler = False if cfg.dataset.name == "multi" else True

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.resources.gpus,
        precision=cfg.training.precision,
        max_epochs=cfg.training.epochs,
        logger=logger,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epochs,
        gradient_clip_val=cfg.training.grad_clip_norm,
        use_distributed_sampler=use_distributed_sampler,
        callbacks=create_checkpoint_callbacks(cfg, logger.experiment.id),
        num_sanity_val_steps=0,
    )

    ######################################################################
    # Log the code to wandb.
    # This is somewhat custom, you'll have to edit this to include whatever
    # additional files you want, but basically it just logs all the files
    # in the project root inside dirs, and with extensions.
    ######################################################################

    # Log the code used to train the model. Make sure not to log too much, because it will be too big.
    if trainer.is_global_zero:
        print(
            json.dumps(
                omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=False
                ),
                sort_keys=True,
                indent=4,
            )
        )

        logger.experiment.log_code(
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

        ckpt = torch.load(ckpt_file)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print("Starting training from scratch.")
        ckpt_file = None

    if cfg.lora.enable:
        assert (
            cfg.checkpoint.run_id is not None
        ), "Doesn't make sense to enable LoRA without initializing from a pretrained model"
        model = apply_lora(model, cfg.lora)

    trainer.fit(model, datamodule=datamodule)
    wandb.run.finish()  # Sometimes the job didn't finish after max epochs reaches
    wandb.finish()


if __name__ == "__main__":
    main()
