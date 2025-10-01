import json
from collections import defaultdict

import numpy as np
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from lfd3d.datasets.lerobot.lerobot_dataset import InferenceLeRoBotDataModule
from lfd3d.utils.script_utils import (
    create_model,
    load_checkpoint_config_from_wandb,
)
from lfd3d.utils.viz_utils import get_heatmap_viz, save_video, generate_heatmap_from_points
from tqdm import tqdm
import cv2

def create_special_datamodule(cfg):
    if cfg.mode == "eval":
        job_cfg = cfg.inference
        stage = "predict"
    elif cfg.mode == "train":
        job_cfg = cfg.training
        stage = "fit"
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")
    
    datamodule = InferenceLeRoBotDataModule(
        batch_size=job_cfg.batch_size,
        val_batch_size=job_cfg.val_batch_size,
        num_workers=cfg.resources.num_workers,
        dataset_cfg=cfg.dataset,
        seed=cfg.seed,
    )
    datamodule.setup(stage)
    return cfg, datamodule

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


def get_eval_datamodule_episode(datamodule, inference_cfg, episode_idx):
    tags = datamodule.val_tags
    eval_dataloaders, eval_tags = [], []
    for episode_id in episode_idx:
        for i, (tag, loader) in enumerate(datamodule.test_dataloader(episode_id).items()):
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
    cfg, datamodule = create_special_datamodule(cfg)

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
        entity="hz2851-carnegie-mellon-university",
        project="lfd3d",
        id=cfg.checkpoint.run_id,
        resume="must",
    )
    

    eval_datamodule = get_eval_datamodule_episode(datamodule, cfg.inference, cfg.episode_idx)

    preds = trainer.predict(model, datamodule=eval_datamodule)

    preds_dict = {tag: {} for tag in eval_datamodule.eval_tags}

    loader = eval_datamodule.predict_dataloader()
    for i, episode_id in enumerate(cfg.episode_idx):
        heatmaps  = []
        raw_heatmaps = []
        for pred, batch in tqdm(zip(preds[i], loader[i]), total = len(loader[i])):
            rgb = batch["rgbs"][:,0].squeeze(0).cpu().numpy() # H, W, 3
            if cfg.model.name =="pixelscore":
                pixel_score = pred["pred"] # B H, W, 3
                pred_coord =  pred["pred_coord"].squeeze(0).cpu().numpy().astype(int) # 2

                heatmap = get_heatmap_viz(rgb, pixel_score[:,:,:,0])
                
                #episode_frames.append(cv2.circle(rgb, (pred_coord[1], pred_coord[0]), 1, (255,0,0),-1))
            elif cfg.model.name == "dino_heatmap":

                pred_coord = pred["pred_coord"].cpu().numpy().astype(int) # 1, 2
                raw_heatmap = pred["outputs"] # 1, H, W

                heatmap_gray = generate_heatmap_from_points(np.concatenate([pred_coord]*3, axis = 0), rgb.shape) # H, W, 3
                heatmap = get_heatmap_viz(rgb, heatmap_gray[:,:,0])

            else:
                raise ValueError
            heatmaps.append(heatmap)
            raw_heatmaps.append(get_heatmap_viz(rgb, raw_heatmap))
        save_video(f"{cfg.log_dir}/episode_{episode_id}_raw_heatmaps_{cfg.model.name}.mp4", frames = raw_heatmaps)
        save_video(f"{cfg.log_dir}/episode_{episode_id}_heatmap_{cfg.model.name}.mp4", frames = heatmaps)


if __name__ == "__main__":
    main()
