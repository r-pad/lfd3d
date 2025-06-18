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


class EvalDataModule(pl.LightningDataModule):
    def __init__(self, dataloaders, tags, inference_cfg):
        super().__init__()
        self.dataloaders = dataloaders
        self.eval_tags = tags
        self.n_samples_wta = inference_cfg.n_samples_wta
        self.save_wta_to_disk = inference_cfg.save_wta_to_disk

    def setup(self, stage=None):
        pass

    def predict_dataloader(self):
        return self.dataloaders


def get_eval_datamodule(datamodule, inference_cfg):
    tags = datamodule.val_tags
    eval_dataloaders, eval_tags = [], []
    for i, (tag, loader) in enumerate(datamodule.test_dataloader().items()):
        eval_dataloaders.append(loader)
        eval_tags.append(f"test_{tag}")
    # eval_dataloaders.append(datamodule.train_subset_dataloader())
    # eval_tags.append("train_subset")
    # for i, (tag, loader) in enumerate(datamodule.val_dataloader().items()):
    #     eval_dataloaders.append(loader)
    #     eval_tags.append(f"val_{tag}")

    eval_datamodule = EvalDataModule(eval_dataloaders, eval_tags, inference_cfg)
    return eval_datamodule


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
    # Should be the same one as in training, but we're gonna use val+test
    # dataloaders.
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
    network.load_state_dict({k.partition(".")[2]: v for k, v in state_dict.items()})
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
        logger=False,
    )

    ######################################################################
    # Run the model on the train/val/test sets.
    # This outputs a list of dictionaries, one for each batch. This
    # is annoying to work with, so later we'll flatten.
    #
    # If a downstream eval, you can swap it out with whatever the eval
    # function is.
    ######################################################################

    # Upload output to wandb
    wandb.init(entity="r-pad", project="lfd3d", id=cfg.checkpoint.run_id, resume="must")

    eval_datamodule = get_eval_datamodule(datamodule, cfg.inference)
    preds = trainer.predict(model, datamodule=eval_datamodule)
    preds_dict = {tag: {} for tag in eval_datamodule.eval_tags}

    # Keys to iterate over
    keys = preds[0][0].keys()
    for i, pred_split in enumerate(preds_dict.keys()):
        # Flatten the predictions for each split separately and store in preds_dict
        preds_split = preds[i]
        for key in keys:
            if type(preds_split[0][key]) == torch.Tensor:
                preds_dict[pred_split][key] = torch.cat([i[key] for i in preds_split])
            else:
                preds_dict[pred_split][key] = []
                for i in preds_split:
                    preds_dict[pred_split][key].extend(i[key])

    # Individual Statistics
    for pred_split in preds_dict.keys():
        table = wandb.Table(columns=list(preds_dict[pred_split].keys()))
        num_rows = len(next(iter(preds_dict[pred_split].values())))

        for i in range(num_rows):
            row = [
                preds_dict[pred_split][key][i] for key in preds_dict[pred_split].keys()
            ]
            table.add_data(*row)
        wandb.log({f"{pred_split}/{dataset_name}-eval_results": table})

    # Summary Statistics
    summary_stats = []
    metrics = []
    for tag in eval_datamodule.eval_tags:
        for postfix in ["rmse", "chamfer_dist", "wta_rmse", "wta_chamfer_dist"]:
            metric = f"{tag}/{postfix}"
            metrics.append(metric)
    for metric in metrics:
        pred_metric = preds_dict[metric.split("/")[0]][metric.split("/")[1]]
        summary_stats.append([metric, pred_metric.mean()])
    summary_table = wandb.Table(data=summary_stats, columns=["Metric", "Value"])
    wandb.log({f"{dataset_name}-eval_results_summary": summary_table})

    wandb.finish()


if __name__ == "__main__":
    main()
