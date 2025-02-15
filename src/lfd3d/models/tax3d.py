import random
from typing import Dict, List

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from diffusers import get_cosine_schedule_with_warmup
from pytorch3d.structures import Pointclouds
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
    get_action_anchor_pcd,
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
    hidden_size = 132 if use_rotary else 192
    return DiT_PointCloud_Cross(depth=5, hidden_size=hidden_size, num_heads=4, **kwargs)


def DiT_PointCloud_Cross_S(use_rotary, **kwargs):
    hidden_size = 192 if use_rotary else 288
    return DiT_PointCloud_Cross(depth=7, hidden_size=hidden_size, num_heads=8, **kwargs)


def DiT_PointCloud_Cross_B(use_rotary, **kwargs):
    hidden_size = 240 if use_rotary else 384
    return DiT_PointCloud_Cross(
        depth=10, hidden_size=hidden_size, num_heads=12, **kwargs
    )


def DiT_PointCloud_xS(use_rotary, **kwargs):
    # hidden size divisible by 3 for rotary embedding, and divisible by num_heads for multi-head attention
    hidden_size = 132 if use_rotary else 128
    return DiT_PointCloud(depth=5, hidden_size=hidden_size, num_heads=4, **kwargs)


def calc_pcd_metrics(pred_dict, pcd, pred, gt, scale_factor, padding_mask):
    """
    Calculate pcd metrics and update pred_dict with the keys.
    Creates point clouds to be measured by applying the predicted
    and gt displacements to the metric pcd.

    pred_dict: Dictionary with keys to be updated
    pcd: Metric Point Cloud
    pred: Predicted cross displacement
    gt: GT cross displacement
    scale_factor: Scaling factor to bring to metric scale
    padding_mask: Padding mask
    """
    pred_pcd = (pcd + pred) * scale_factor[:, None, :]
    gt_pcd = (pcd + gt) * scale_factor[:, None, :]

    pred_dict["rmse"] = rmse_pcd(pred_pcd, gt_pcd, padding_mask)
    pred_dict["chamfer_dist"] = chamfer_distance(pred_pcd, gt_pcd, padding_mask)
    return pred_dict


# TODO: clean up all unused functions
DiT_models = {
    "DiT_pcu_S": DiT_pcu_S,
    "DiT_pcu_xS": DiT_pcu_xS,
    "DiT_pcu_cross_xS": DiT_pcu_cross_xS,
    "Rel3D_DiT_pcu_cross_xS": Rel3D_DiT_pcu_cross_xS,
    # there is no Rel3D_DiT_pcu_xS
    "DiT_PointCloud_Cross_xS": DiT_PointCloud_Cross_xS,
    "DiT_PointCloud_Cross_S": DiT_PointCloud_Cross_S,
    "DiT_PointCloud_Cross_B": DiT_PointCloud_Cross_B,
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
        self.predict_outputs: List[Dict] = []

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
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # data params
        self.batch_size = self.run_cfg.batch_size
        self.val_batch_size = self.run_cfg.val_batch_size

        # diffusion params
        self.noise_schedule = self.model_cfg.diff_noise_schedule
        # self.noise_scale = model_cfg.diff_noise_scale
        self.diff_steps = self.model_cfg.diff_train_steps  # TODO: rename to diff_steps
        self.diffusion = create_diffusion(
            timestep_respacing=None,
            diffusion_steps=self.diff_steps,
            noise_schedule=self.noise_schedule,
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # Step after every batch
            },
        }

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
        ground_truth = batch[self.label_key].points_padded()
        ground_truth = ground_truth.permute(0, 2, 1)  # channel first
        padding_mask = batch["padding_mask"]
        model_kwargs = self.get_model_kwargs(batch)

        # run diffusion
        # noise = torch.randn_like(ground_truth) * self.noise_scale
        loss_dict = self.diffusion.training_losses(
            model=self.network,
            x_start=ground_truth,
            t=t,
            model_kwargs=model_kwargs,
            padding_mask=padding_mask,
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
        batch_size, sample_size = batch[self.label_key].points_padded().shape[:2]
        model_kwargs = self.get_model_kwargs(batch)

        # generating latents and running diffusion
        z = torch.randn(batch_size, 3, sample_size, device=self.device)
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

        # Some opengl issues with creating the video on the cluster
        # It's not very useful anyway so keeping it disabled.
        log_pcd_video = False

        batch_size = batch[self.label_key].points_padded().shape[0]
        # pick a random sample in the batch to visualize
        viz_idx = np.random.randint(0, batch_size)
        RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)

        pred = pred_dict[self.prediction_type]["pred"][viz_idx].cpu().numpy()
        end2start = np.linalg.inv(batch["start2end"][viz_idx].cpu().numpy())

        goal_text = batch["caption"][viz_idx]
        vid_name = batch["vid_name"][viz_idx]
        rmse = pred_dict["rmse"][viz_idx]
        anchor_pcd = batch["anchor_pcd"].points_padded()[viz_idx].cpu().numpy()
        pcd = batch["action_pcd"].points_padded()[viz_idx].cpu().numpy()
        gt = batch[self.prediction_type].points_padded()[viz_idx].cpu().numpy()

        padding_mask = batch["padding_mask"][viz_idx].cpu().numpy()
        pred_pcd = pcd + pred
        gt_pcd = pcd + gt

        # Move center back from action_pcd to the camera frame before viz
        pcd_mean = batch["pcd_mean"][viz_idx].cpu().numpy()
        pcd_std = batch["pcd_std"][viz_idx].cpu().numpy()
        pcd = (pcd * pcd_std) + pcd_mean
        anchor_pcd = (anchor_pcd * pcd_std) + pcd_mean
        pred_pcd = (pred_pcd * pcd_std) + pcd_mean
        gt_pcd = (gt_pcd * pcd_std) + pcd_mean

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
        init_rgb_proj = project_pcd_on_image(pcd, padding_mask, rgb_init, K, GREEN)
        end_rgb_proj = project_pcd_on_image(gt_pcd, padding_mask, rgb_end, K, RED)
        pred_rgb_proj = project_pcd_on_image(pred_pcd, padding_mask, rgb_end, K, BLUE)
        rgb_proj_viz = cv2.hconcat([init_rgb_proj, end_rgb_proj, pred_rgb_proj])

        wandb_proj_img = wandb.Image(
            rgb_proj_viz,
            caption=f"Left: Initial Frame (GT Track)\n; Middle: Final Frame (GT Track)\n\
            ; Right: Final Frame (Pred Track)\n; Goal Description : {goal_text};\n\
            rmse={rmse};\nvideo path = {vid_name}; ",
        )
        ###

        # Visualize point cloud
        viz_pcd = get_img_and_track_pcd(
            rgb_end,
            depth_end,
            K,
            padding_mask,
            pcd_endframe,
            gt_pcd,
            pred_pcd,
            GREEN,
            RED,
            BLUE,
        )
        ###

        # Visualize action/anchor point cloud
        action_anchor_pcd = get_action_anchor_pcd(
            pcd,
            anchor_pcd,
            GREEN,
            RED,
        )
        ###

        viz_dict = {
            f"{tag}/track_projected_to_rgb": wandb_proj_img,
            f"{tag}/image_and_tracks_pcd": wandb.Object3D(viz_pcd),
            f"{tag}/action_anchor_pcd": wandb.Object3D(action_anchor_pcd),
            "trainer/global_step": self.global_step,
        }

        if log_pcd_video:
            # Render video of point cloud
            pcd_video = create_point_cloud_frames(viz_pcd)
            pcd_video = np.transpose(pcd_video, (0, 3, 1, 2))
            viz_dict[f"{tag}/pcd_video"] = wandb.Video(pcd_video, fps=6, format="webm")

        wandb.log(viz_dict)
        ###

    def training_step(self, batch, batch_idx):
        """
        Training step for the module. Logs training metrics and visualizations to wandb.
        """
        self.train()
        batch_size = batch[self.label_key].points_padded().shape[0]
        t = torch.randint(0, self.diff_steps, (batch_size,), device=self.device).long()
        _, loss = self(batch, t)
        action_pcd = batch["action_pcd"]
        #########################################################
        # logging training metrics
        #########################################################
        train_metrics = {"loss": loss}

        # determine if additional logging should be done
        do_additional_logging = (
            self.global_step % self.additional_train_logging_period == 0
            and self.global_step != 0
        )

        # additional logging
        if do_additional_logging:
            self.eval()
            with torch.no_grad():
                pred_dict = self.predict(batch)
                pred = pred_dict[self.prediction_type]["pred"]
            self.train()  # Switch back to training mode

            padding_mask = batch["padding_mask"]
            pcd_std = batch["pcd_std"]
            ground_truth = batch[self.label_key].to(self.device)
            pred_dict = calc_pcd_metrics(
                pred_dict,
                action_pcd.points_padded(),
                pred,
                ground_truth.points_padded(),
                pcd_std,
                padding_mask,
            )
            train_metrics.update(pred_dict)

            if self.trainer.is_global_zero:
                ####################################################
                # logging visualizations
                ####################################################
                self.log_viz_to_wandb(batch, pred_dict, "train")

        self.train_outputs.append(train_metrics)
        return loss

    def on_train_epoch_end(self):
        if len(self.train_outputs) == 0:
            return

        log_dictionary = {}
        loss = torch.stack([x["loss"] for x in self.train_outputs]).mean()
        log_dictionary["train/loss"] = loss

        extra_keys = any("rmse" in x.keys() for x in self.train_outputs)
        if extra_keys:
            rmse = torch.stack(
                [x["rmse"].mean() for x in self.train_outputs if "rmse" in x]
            ).mean()
            log_dictionary["train/rmse"] = rmse

            chamfer_dist = torch.stack(
                [
                    x["chamfer_dist"].mean()
                    for x in self.train_outputs
                    if "chamfer_dist" in x
                ]
            ).mean()
            log_dictionary["train/chamfer_dist"] = chamfer_dist

        ####################################################
        # logging training metrics
        ####################################################
        self.log_dict(
            log_dictionary,
            add_dataloader_idx=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_outputs.clear()

    def on_validation_epoch_start(self):
        # Choose a random batch index for each validation epoch
        self.random_val_viz_idx = random.randint(
            0, len(self.trainer.val_dataloaders) - 1
        )

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the module. Logs validation metrics and visualizations to wandb.
        """
        self.eval()
        with torch.no_grad():
            pred_dict = self.predict(batch)

        pred = pred_dict[self.prediction_type]["pred"]
        ground_truth = batch[self.label_key].to(self.device)
        action_pcd = batch["action_pcd"]
        padding_mask = batch["padding_mask"]
        pcd_std = batch["pcd_std"]
        pred_dict = calc_pcd_metrics(
            pred_dict,
            action_pcd.points_padded(),
            pred,
            ground_truth.points_padded(),
            pcd_std,
            padding_mask,
        )
        self.val_outputs.append(pred_dict)

        ####################################################
        # logging visualizations
        ####################################################
        if batch_idx == self.random_val_viz_idx and self.trainer.is_global_zero:
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

        The test dataset is expected to be first dataloader_idx.
        Visualizations are logged with dataloader_idx=0
        """
        pred_dict = self.predict(batch)
        pred = pred_dict[self.prediction_type]["pred"]
        ground_truth = batch[self.label_key].to(self.device)
        action_pcd = batch["action_pcd"]
        padding_mask = batch["padding_mask"]
        pcd_std = batch["pcd_std"]
        pred_dict = calc_pcd_metrics(
            pred_dict,
            action_pcd.points_padded(),
            pred,
            ground_truth.points_padded(),
            pcd_std,
            padding_mask,
        )

        if dataloader_idx == 0:
            self.predict_outputs.append(pred_dict)

        return {
            "rmse": pred_dict["rmse"],
            "chamfer_dist": pred_dict["chamfer_dist"],
            "vid_name": batch["vid_name"],
            "caption": batch["caption"],
        }

    def on_predict_epoch_end(self):
        """
        TODO: Fix up this function. Assumes batch contains only one of best/worst
        """
        batch_size = self.predict_outputs[0]["rmse"].shape[0]
        rmse = torch.cat([x["rmse"] for x in self.predict_outputs])
        chamfer_dist = torch.cat([x["chamfer_dist"] for x in self.predict_outputs])
        cross_displacement = []
        for i in self.predict_outputs:
            cross_displacement.extend(i["cross_displacement"]["pred"])

        best_5_idx = rmse.argsort()[:5].tolist()
        best_batch_num = [i // batch_size for i in best_5_idx]
        worst_5_idx = rmse.argsort()[-5:].tolist()
        worst_batch_num = [i // batch_size for i in worst_5_idx]

        for i, batch in enumerate(self.trainer.predict_dataloaders[0]):
            if i in best_batch_num or i in worst_batch_num:
                if i in best_batch_num:
                    log_key = "eval/best_rmse"
                    idx = best_5_idx[best_batch_num.index(i)]
                else:
                    log_key = "eval/worst_rmse"
                    idx = worst_5_idx[worst_batch_num.index(i)]
                within_batch_idx = idx % batch_size

                pred_dict = self.compose_pred_dict_for_viz(
                    rmse, chamfer_dist, cross_displacement, idx
                )
                viz_batch = self.compose_batch_for_viz(batch, within_batch_idx)
                self.log_viz_to_wandb(viz_batch, pred_dict, log_key)
        self.predict_outputs.clear()

    def compose_pred_dict_for_viz(self, rmse, chamfer_dist, cross_displacement, idx):
        return {
            "rmse": [rmse[idx]],
            "chamfer_dist": [chamfer_dist[idx]],
            "cross_displacement": {"pred": cross_displacement[idx]},
        }

    def compose_batch_for_viz(self, batch, within_batch_idx):
        viz_batch = {}
        for key in batch.keys():
            if type(batch[key]) == Pointclouds:
                pcd = batch[key].points_padded()[within_batch_idx]
                viz_batch[key] = Pointclouds(points=pcd[None])
            elif key in ["rgbs", "depths"]:
                viz_batch[key] = batch[key][within_batch_idx][None]
            else:
                viz_batch[key] = [batch[key][within_batch_idx]]
        return viz_batch


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
        pc_action = batch["action_pcd"].points_padded()
        pc_anchor = batch["anchor_pcd"].points_padded()
        pc_anchor_feat = batch["anchor_pcd"].features_padded()
        text_embedding = (
            batch["text_embed"].unsqueeze(1).repeat(1, pc_action.shape[1], 1)
        )

        pc_action = pc_action.permute(0, 2, 1)  # channel first
        pc_anchor = pc_anchor.permute(0, 2, 1)  # channel first
        pc_anchor_feat = pc_anchor_feat.permute(0, 2, 1)  # channel first
        model_kwargs = dict(
            x0=pc_action, y=pc_anchor, y_feat=pc_anchor_feat, text_embed=text_embedding
        )
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
