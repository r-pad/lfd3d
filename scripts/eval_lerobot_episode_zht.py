import json
import os
import random
from datetime import datetime

import hydra
import numpy as np
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
from lfd3d.utils.viz_utils import (
    generate_heatmap_from_points,
    get_heatmap_viz,
    save_video,
)
from tqdm import tqdm


def save_path(cfg, episode_id):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    date_dir = os.path.join(cfg.log_dir, f"eval_{cfg.dataset.name}_episodes", date_str)
    time_dir = os.path.join(date_dir, time_str)
    episode_dir = os.path.join(time_dir, f"episode_{episode_id}")

    os.makedirs(date_dir, exist_ok=True)
    os.makedirs(time_dir, exist_ok=True)
    os.makedirs(episode_dir, exist_ok=True)

    return episode_dir


def random_episode(episode_idx, n):
    if n > len(episode_idx):
        return episode_idx
    return random.sample(episode_idx, n)


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


def get_eval_datamodule_episode(datamodule, inference_cfg):
    tags = datamodule.val_tags
    eval_dataloaders, eval_tags, episode_idx = [], [], []
    episode_num = inference_cfg.n_eval_episode if "n_eval_episode" in inference_cfg.keys() else 1

    for i, (tag, loader) in enumerate(datamodule.test_dataloader().items()):
        eval_dataloaders.extend(loader.values())
        episode_idx.extend(loader.keys())
        eval_tags.extend([f"test_{tag}" for i in range(len(loader.keys()))])
    # eval_dataloaders.append(datamodule.train_subset_dataloader())
    # eval_tags.append("train_subset")
    # for i, (tag, loader) in enumerate(datamodule.val_dataloader().items()):
    #     eval_dataloaders.append(loader)
    #     eval_tags.append(f"val_{tag}")

    random_id = random_episode(list(range(0, len(episode_idx))), episode_num)
    eval_dataloaders = [eval_dataloaders[i] for i in random_id]
    eval_tags = [eval_tags[i] for i in random_id]
    episode_idx = [episode_idx[i] for i in random_id]

    eval_datamodule = EvalDataModule(eval_dataloaders, eval_tags, inference_cfg)
    return episode_idx, eval_datamodule


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
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.checkpoint.run_id,
        resume="must",
    )

    episode_idx, eval_datamodule = get_eval_datamodule_episode(
        datamodule, cfg.inference
    )
    print(f"Total episode:{episode_idx}")
    preds = trainer.predict(model, datamodule=eval_datamodule)
    preds_dict = {tag: {} for tag in eval_datamodule.eval_tags}

    loader = eval_datamodule.predict_dataloader()
    for i, episode_id in enumerate(episode_idx):
        heatmaps = []
        raw_heatmaps = []

        for pred, batch in tqdm(zip(preds[i], loader[i]), total=len(loader[i])):
            rgb = batch["rgbs"][:, 0].cpu().numpy()  # B, H, W, 3
            batch_size = rgb.shape[0]

            if cfg.model.name == "dino_heatmap":
                pred_coord = pred["pred_coord"].cpu().numpy().astype(int)  # B, 2
                raw_heatmap = pred["outputs"]  # B, 1, H, W

                for j in range(batch_size):
                    heatmap_gray = generate_heatmap_from_points(
                        np.stack([pred_coord[j]] * 3, axis=0), rgb[j].shape
                    )  # H, W, 3
                    heatmap = get_heatmap_viz(rgb[j], heatmap_gray[:, :, 0])  # H, W, 3
                    raw_heatmap_viz = get_heatmap_viz(rgb[j], raw_heatmap[j])  # H, W, 3

                    heatmaps.append(heatmap)
                    raw_heatmaps.append(raw_heatmap_viz)

            else:
                raise NotImplementedError

        file_dir = save_path(cfg, episode_id)
        save_video(
            f"{file_dir}/raw_heatmaps_{cfg.model.name}_{cfg.inference.loss_type}.mp4",
            frames=raw_heatmaps,
        )
        save_video(
            f"{file_dir}/heatmap_{cfg.model.name}_{cfg.inference.loss_type}.mp4",
            frames=heatmaps,
        )


if __name__ == "__main__":
    main()
