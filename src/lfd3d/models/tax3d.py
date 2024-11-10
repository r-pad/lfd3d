import os
from typing import Dict, List

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from diffusers import get_cosine_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from torch import nn, optim

from lfd3d.metrics.pcd_metrics import chamfer_distance, rmse_pcd
from lfd3d.models.dit.diffusion import create_diffusion
from lfd3d.models.dit.models import DiT_PointCloud, DiT_PointCloud_Cross
from lfd3d.models.dit.models import DiT_PointCloud_Unc as DiT_pcu
from lfd3d.models.dit.models import (
    DiT_PointCloud_Unc_Cross,
    Rel3D_DiT_PointCloud_Unc_Cross,
)
from lfd3d.utils.viz_utils import (
    create_point_cloud_frames,
    get_img_and_track_pcd,
    project_pcd_on_image,
)


def DiT_pcu_S(**kwargs):
    return DiT_pcu(depth=12, hidden_size=384, num_heads=6, **kwargs)


def DiT_pcu_xS(**kwargs):
    return DiT_pcu(depth=5, hidden_size=128, num_heads=4, **kwargs)


def DiT_pcu_cross_xS(**kwargs):
    return DiT_PointCloud_Unc_Cross(depth=5, hidden_size=128, num_heads=4, **kwargs)


def Rel3D_DiT_pcu_cross_xS(**kwargs):
    # Embed dim divisible by 3 for 3D positional encoding and divisible by num_heads for multi-head attention
    return Rel3D_DiT_PointCloud_Unc_Cross(
        depth=5, hidden_size=132, num_heads=4, **kwargs
    )


def DiT_PointCloud_Cross_xS(use_rotary, **kwargs):
    # hidden size divisible by 3 for rotary embedding, and divisible by num_heads for multi-head attention
    hidden_size = 132 if use_rotary else 128
    return DiT_PointCloud_Cross(depth=5, hidden_size=hidden_size, num_heads=4, **kwargs)


def DiT_PointCloud_xS(use_rotary, **kwargs):
    # hidden size divisible by 3 for rotary embedding, and divisible by num_heads for multi-head attention
    hidden_size = 132 if use_rotary else 128
    return DiT_PointCloud(depth=5, hidden_size=hidden_size, num_heads=4, **kwargs)


def encode_without_parallelism(text_embed_model, text):
    original_parallelism = os.environ.get("TOKENIZERS_PARALLELISM", None)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    embeddings = text_embed_model.encode(text, show_progress_bar=False)
    if original_parallelism is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = original_parallelism
    else:
        del os.environ["TOKENIZERS_PARALLELISM"]
    return embeddings


def calc_pcd_metrics(pred_dict, pcd, pred, gt):
    """
    Calculate pcd metrics and update pred_dict with the keys.
    Creates point clouds to be measured by applying the predicted
    and gt displacements to the metric pcd.

    pred_dict: Dictionary with keys to be updated
    pcd: Metric Point Cloud
    pred: Predicted cross displacement
    gt: GT cross displacement
    """
    pred_pcd = pcd + pred
    gt_pcd = pcd + gt

    pred_dict["rmse"] = rmse_pcd(pred_pcd, gt_pcd)
    pred_dict["chamfer_dist"] = chamfer_distance(pred_pcd, gt_pcd)
    return pred_dict


# TODO: clean up all unused functions
DiT_models = {
    "DiT_pcu_S": DiT_pcu_S,
    "DiT_pcu_xS": DiT_pcu_xS,
    "DiT_pcu_cross_xS": DiT_pcu_cross_xS,
    "Rel3D_DiT_pcu_cross_xS": Rel3D_DiT_pcu_cross_xS,
    # there is no Rel3D_DiT_pcu_xS
    "DiT_PointCloud_Cross_xS": DiT_PointCloud_Cross_xS,
    # TODO: add the SD model here
    "DiT_PointCloud_xS": DiT_PointCloud_xS,
}


def get_model(model_cfg):
    # rotary = "Rel3D_" if model_cfg.rotary else ""
    cross = "Cross_" if model_cfg.name == "df_cross" else ""
    # model_name = f"{rotary}DiT_pcu_{cross}{model_cfg.size}"
    model_name = f"DiT_PointCloud_{cross}{model_cfg.size}"
    return DiT_models[model_name]


class DiffusionTransformerNetwork(nn.Module):
    """
    Network containing the specified Diffusion Transformer architecture.
    """

    def __init__(self, model_cfg=None):
        super().__init__()
        self.dit = get_model(model_cfg)(
            use_rotary=model_cfg.rotary,
            in_channels=model_cfg.in_channels,
            learn_sigma=model_cfg.learn_sigma,
            model_cfg=model_cfg,
        )

    def forward(self, x, t, **kwargs):
        return self.dit(x, t, **kwargs)


class DenseDisplacementDiffusionModule(pl.LightningModule):
    """
    Generalized Dense Displacement Diffusion (DDD) module that handles model training, inference,
    evaluation, and visualization. This module is inherited and overriden by scene-level and
    object-centric modules.
    """

    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.prediction_type = self.model_cfg.type  # flow or point
        self.mode = cfg.mode  # train or eval
        self.val_outputs: List[Dict] = []
        self.train_outputs: List[Dict] = []

        # prediction type-specific processing
        # TODO: eventually, this should be removed by updating dataset to use "point" instead of "pc"
        if self.prediction_type == "flow":
            self.label_key = "flow"
        elif self.prediction_type == "point":
            self.label_key = "pc"
        elif self.prediction_type == "cross_displacement":
            self.label_key = "cross_displacement"
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")

        self.text_embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        for param in self.text_embed.parameters():
            param.requires_grad = False

        # mode-specific processing
        if self.mode == "train":
            self.run_cfg = cfg.training
            # training-specific params
            self.lr = self.run_cfg.lr
            self.weight_decay = self.run_cfg.weight_decay
            self.num_training_steps = self.run_cfg.num_training_steps
            self.lr_warmup_steps = self.run_cfg.lr_warmup_steps
            self.additional_train_logging_period = (
                self.run_cfg.additional_train_logging_period
            )
        elif self.mode == "eval":
            self.run_cfg = cfg.inference
            # inference-specific params
            self.num_trials = self.run_cfg.num_trials
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # data params
        self.batch_size = self.run_cfg.batch_size
        self.val_batch_size = self.run_cfg.val_batch_size
        # TODO: it is debatable if the module needs to know about the sample size
        self.sample_size = self.run_cfg.sample_size
        self.sample_size_anchor = self.run_cfg.sample_size_anchor

        # diffusion params
        # self.noise_schedule = model_cfg.diff_noise_schedule
        # self.noise_scale = model_cfg.diff_noise_scale
        self.diff_steps = self.model_cfg.diff_train_steps  # TODO: rename to diff_steps
        self.diffusion = create_diffusion(
            timestep_respacing=None,
            diffusion_steps=self.diff_steps,
            # noise_schedule=self.noise_schedule,
        )

    def configure_optimizers(self):
        assert self.mode == "train", "Can only configure optimizers in training mode."
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [lr_scheduler]

    def get_model_kwargs(self, batch):
        """
        Get the model kwargs for the forward pass.
        """
        raise NotImplementedError("This should be implemented in the derived class.")

    def get_viz_args(self, batch, viz_idx):
        """
        Get visualization arguments for wandb logging.
        """
        raise NotImplementedError("This should be implemented in the derived class.")

    def forward(self, batch, t):
        """
        Forward pass to compute diffusion training loss.
        """
        ground_truth = batch[self.label_key].permute(0, 2, 1)  # channel first
        model_kwargs = self.get_model_kwargs(batch)

        # run diffusion
        # noise = torch.randn_like(ground_truth) * self.noise_scale
        loss_dict = self.diffusion.training_losses(
            model=self.network,
            x_start=ground_truth,
            t=t,
            model_kwargs=model_kwargs,
            # noise=noise,
        )
        loss = loss_dict["loss"].mean()
        return None, loss

    @torch.no_grad()
    def predict(self, batch, progress=False):
        """
        Compute prediction for a given batch.

        Args:
            batch: the input batch
            progress: whether to show progress bar
        """
        # TODO: replace bs with batch_size?
        bs, sample_size = batch["start_pcd"].shape[:2]
        model_kwargs = self.get_model_kwargs(batch)

        # generating latents and running diffusion
        z = torch.randn(bs, 3, sample_size, device=self.device)
        pred, results = self.diffusion.p_sample_loop(
            self.network,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
            device=self.device,
        )
        pred = pred.permute(0, 2, 1)

        return {self.prediction_type: {"pred": pred}}

    def log_viz_to_wandb(self, batch, pred_dict, tag):
        """
        Log visualizations to wandb.

        Args:
            batch: the input batch
            pred_dict: the prediction dictionary
            tag: the tag to use for logging
        """
        # pick a random sample in the batch to visualize
        viz_idx = np.random.randint(0, batch["start_pcd"].shape[0])
        RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)

        pred = pred_dict[self.prediction_type]["pred"][viz_idx].cpu().numpy()
        end2start = np.linalg.inv(batch["start2end"][viz_idx].cpu().numpy())

        goal_text = batch["caption"][viz_idx]
        vid_name = batch["vid_name"][viz_idx]
        pcd = batch["start_pcd"][viz_idx].cpu().numpy()
        gt = batch[self.prediction_type][viz_idx].cpu().numpy()

        mask = ~np.all(pcd == 0, axis=1)  # Remove 0 points
        pred_pcd = pcd + pred
        gt_pcd = pcd + gt

        # All points cloud are in the start image's coordinate frame
        # We need to visualize the end image, therefore need to apply transform
        # Transform the point clouds to align with end image coordinate frame
        pcd_endframe = np.hstack((pcd, np.ones((pcd.shape[0], 1))))
        pcd_endframe = (end2start @ pcd_endframe.T).T[:, :3]
        pred_pcd = np.hstack((pred_pcd, np.ones((pred_pcd.shape[0], 1))))
        pred_pcd = (end2start @ pred_pcd.T).T[:, :3]
        gt_pcd = np.hstack((gt_pcd, np.ones((gt_pcd.shape[0], 1))))
        gt_pcd = (end2start @ gt_pcd.T).T[:, :3]

        K = batch["intrinsics"][viz_idx].cpu().numpy()

        rgb_init, rgb_end = (
            batch["rgbs"][viz_idx, 0].cpu().numpy(),
            batch["rgbs"][viz_idx, 1].cpu().numpy(),
        )
        depth_init, depth_end = (
            batch["depths"][viz_idx, 0].cpu().numpy(),
            batch["depths"][viz_idx, 1].cpu().numpy(),
        )

        ### Project tracks to image and save
        init_rgb_proj = project_pcd_on_image(pcd, mask, rgb_init, K, GREEN)
        end_rgb_proj = project_pcd_on_image(gt_pcd, mask, rgb_end, K, RED)
        pred_rgb_proj = project_pcd_on_image(pred_pcd, mask, rgb_end, K, BLUE)
        rgb_proj_viz = cv2.hconcat([init_rgb_proj, end_rgb_proj, pred_rgb_proj])
        rgb_proj_viz = cv2.resize(rgb_proj_viz, (0, 0), fx=0.25, fy=0.25)

        wandb_proj_img = wandb.Image(
            rgb_proj_viz,
            caption=f"Left: Initial Frame (GT Track)\n; Middle: Final Frame (GT Track)\n\
            ; Right: Final Frame (Pred Track)\n; Goal Description : {goal_text};\n\
            video path = {vid_name}",
        )
        ###

        # Visualize point cloud
        viz_pcd = get_img_and_track_pcd(
            rgb_end,
            depth_end,
            K,
            mask,
            pcd_endframe,
            gt_pcd,
            pred_pcd,
            GREEN,
            RED,
            BLUE,
        )
        ###

        # Render video of point cloud
        pcd_video = create_point_cloud_frames(viz_pcd)
        pcd_video = np.transpose(pcd_video, (0, 3, 1, 2))

        wandb.log(
            {
                f"{tag}/track_projected_to_rgb": wandb_proj_img,
                f"{tag}/image_and_tracks_pcd": wandb.Object3D(viz_pcd),
                f"{tag}/pcd_video": wandb.Video(pcd_video, fps=6, format="webm"),
                "trainer/global_step": self.global_step,
            },
        )
        ###

    def training_step(self, batch, batch_idx):
        """
        Training step for the module. Logs training metrics and visualizations to wandb.
        """
        self.train()
        t = torch.randint(
            0, self.diff_steps, (batch[self.label_key].shape[0],), device=self.device
        ).long()
        _, loss = self(batch, t)
        start_pcd = batch["start_pcd"]
        #########################################################
        # logging training metrics
        #########################################################
        train_metrics = {"loss": loss}

        # determine if additional logging should be done
        do_additional_logging = (
            self.global_step % self.additional_train_logging_period == 0
            and self.trainer.is_global_zero
        )

        # additional logging
        if do_additional_logging:
            pred_dict = self.predict(batch)
            pred = pred_dict[self.prediction_type]["pred"]
            ground_truth = batch[self.label_key].to(self.device)
            pred_dict = calc_pcd_metrics(pred_dict, start_pcd, pred, ground_truth)
            train_metrics.update(pred_dict)

            ####################################################
            # logging visualizations
            ####################################################
            self.log_viz_to_wandb(batch, pred_dict, "train")

        self.train_outputs.append(train_metrics)
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack([x["loss"] for x in self.train_outputs]).mean()
        rmse = torch.stack(
            [x["rmse"] for x in self.train_outputs if "rmse" in x]
        ).mean()
        chamfer_dist = torch.stack(
            [x["chamfer_dist"] for x in self.train_outputs if "chamfer_dist" in x]
        ).mean()
        ####################################################
        # logging training metrics
        ####################################################
        self.log_dict(
            {
                "train/loss": loss,
                "train/rmse": rmse,
                "train/chamfer_dist": chamfer_dist,
            },
            add_dataloader_idx=False,
            sync_dist=True,
            prog_bar=True,
        )
        self.train_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the module. Logs validation metrics and visualizations to wandb.
        """
        self.eval()
        with torch.no_grad():
            pred_dict = self.predict(batch)

        pred = pred_dict[self.prediction_type]["pred"]
        ground_truth = batch[self.label_key].to(self.device)
        start_pcd = batch["start_pcd"]
        pred_dict = calc_pcd_metrics(pred_dict, start_pcd, pred, ground_truth)
        self.val_outputs.append(pred_dict)

        ####################################################
        # logging visualizations
        ####################################################
        if batch_idx == 0 and self.trainer.is_global_zero:
            self.log_viz_to_wandb(batch, pred_dict, "val")
        return pred_dict

    def on_validation_epoch_end(self):
        rmse = torch.stack([x["rmse"].mean() for x in self.val_outputs]).mean()
        chamfer_dist = torch.stack(
            [x["chamfer_dist"].mean() for x in self.val_outputs]
        ).mean()
        ####################################################
        # logging validation metrics
        ####################################################
        self.log_dict(
            {
                f"val/rmse": rmse,
                f"val/chamfer_dist": chamfer_dist,
            },
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.val_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for model evaluation.
        """
        pred_dict = self.predict(batch)
        return {
            "rmse": pred_dict["rmse"],
        }


class SceneDisplacementModule(DenseDisplacementDiffusionModule):
    """
    Scene-level DDD module. Applies self-attention to the entire scene.
    """

    def __init__(self, network, cfg) -> None:
        super().__init__(network, cfg)

    def get_model_kwargs(self, batch):
        pc_action = batch["pc_action"]
        pc_action = pc_action.permute(0, 2, 1)  # channel first
        model_kwargs = dict(x0=pc_action)
        return model_kwargs

    def get_viz_args(self, batch, viz_idx):
        """
        Get visualization arguments for wandb logging.
        """
        pc_pos_viz = batch["pc"][viz_idx, :, :3]
        pc_action_viz = batch["pc_action"][viz_idx, :, :3]
        viz_args = {
            "pc_pos_viz": pc_pos_viz,
            "pc_action_viz": pc_action_viz,
        }
        return viz_args


class CrossDisplacementModule(DenseDisplacementDiffusionModule):
    """
    Object-centric DDD module. Applies cross attention between action and anchor objects.
    """

    def __init__(self, network, cfg) -> None:
        super().__init__(network, cfg)

    def get_model_kwargs(self, batch):
        pc_action = batch["start_pcd"]
        pc_anchor = batch["start_pcd"]

        text_embedding = torch.tensor(
            encode_without_parallelism(self.text_embed, batch["caption"]),
            device=self.device,
        )
        # Repeat embedding to apply to every point for conditioning
        text_embedding = text_embedding.unsqueeze(1).repeat(1, pc_action.shape[1], 1)

        pc_action = pc_action.permute(0, 2, 1)  # channel first
        pc_anchor = pc_anchor.permute(0, 2, 1)  # channel first
        model_kwargs = dict(x0=pc_action, y=pc_anchor, text_embed=text_embedding)
        return model_kwargs

    def get_viz_args(self, batch, viz_idx):
        """
        Get visualization arguments for wandb logging.
        """
        pc_pos_viz = batch["pc"][viz_idx, :, :3]
        pc_action_viz = batch["pc_action"][viz_idx, :, :3]
        pc_anchor_viz = batch["pc_anchor"][viz_idx, :, :3]
        viz_args = {
            "pc_pos_viz": pc_pos_viz,
            "pc_action_viz": pc_action_viz,
            "pc_anchor_viz": pc_anchor_viz,
        }
        return viz_args
