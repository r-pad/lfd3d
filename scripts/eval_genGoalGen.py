import json

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from hydra.core.hydra_config import HydraConfig

from lfd3d.utils.script_utils import (
    create_datamodule,
    create_model,
    load_checkpoint_config_from_wandb,
)


@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def main(cfg):
    task_overrides = HydraConfig.get().overrides.task
    cfg = load_checkpoint_config_from_wandb(
        cfg, task_overrides, cfg.wandb.entity, cfg.wandb.project, cfg.checkpoint.run_id
    )
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
    pl.seed_everything(42)

    device = f"cuda:{cfg.resources.gpus[0]}"
    dataset_name = cfg.dataset.name

    ######################################################################
    # Create the datamodule.
    ######################################################################
    cfg, datamodule = create_datamodule(cfg)

    ######################################################################
    # Create the network(s) which will be evaluated (same as training).
    # You might want to put this into a "create_network" function
    # somewhere so train and eval can be the same.
    #
    # We'll also load the weights.
    ######################################################################

    # Model architecture is dataset-dependent, so we have a helper
    # function to create the model (while separating out relevant vals).
    network, model = create_model(cfg)

    # get checkpoint file (for now, this does not log a run)
    checkpoint_reference = cfg.checkpoint.reference
    if checkpoint_reference.startswith(cfg.wandb.entity):
        api = wandb.Api()
        artifact_dir = cfg.wandb.artifact_dir
        artifact = api.artifact(checkpoint_reference, type="model")
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference
    # Load the network weights.
    ckpt = torch.load(ckpt_file, map_location=device)
    state_dict = {k: v for k, v in ckpt["state_dict"].items()}
    network.load_state_dict({k.partition(".")[2]: v for k, v, in state_dict.items()})
    # set model to eval mode
    network.eval()
    model.eval()

    ######################################################################
    # Create the trainer.
    # Bit of a misnomer here, we're not doing training. But we are gonna
    # use it to set up the model appropriately and do all the batching
    # etc.
    #
    # If this is a different kind of downstream eval, chuck this block.
    ######################################################################

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.resources.gpus,
        precision="16-mixed",
        logger=False,
    )

    dataloaders = [
        datamodule.test_dataloader(),
    ]

    # Upload output to wandb
    wandb.init(entity="r-pad", project="lfd3d", id=cfg.checkpoint.run_id, resume="must")

    preds = trainer.predict(model, dataloaders=dataloaders)

    wandb.finish()


if __name__ == "__main__":
    main()
