import random
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from diffusers import get_cosine_schedule_with_warmup
from torch import nn, optim
from transformers import AutoImageProcessor, AutoModel

from lfd3d.models.tax3d import calc_pcd_metrics
from lfd3d.utils.viz_utils import (
    get_img_and_track_pcd,
    invert_augmentation_and_normalization,
    project_pcd_on_image,
)


class Dino3DGPNetwork(nn.Module):
    """
    DINOv2 + 3D positional encoding + Transformer for 3D goal prediction
    Architecture:
    - Image tokens: DINOv2 patches with 3D PE (x,y,z from depth)
    - Language token: SigLIP embedding
    - Gripper token: 6DoF pose + gripper width
    - Transformer: self-attention blocks
    - Output: 256 GMM components, each predicting 13-dim (4×3 coords + 1 weight)
    """

    def __init__(self, model_cfg):
        super(Dino3DGPNetwork, self).__init__()

        # DINOv2 backbone
        self.backbone_processor = AutoImageProcessor.from_pretrained(
            model_cfg.dino_model
        )
        self.backbone = AutoModel.from_pretrained(model_cfg.dino_model)
        self.backbone.requires_grad_(False)  # Freeze backbone

        # Get backbone dimensions
        self.pos_encoding_dim = 128
        self.hidden_dim = self.backbone.config.hidden_size + self.pos_encoding_dim
        self.patch_size = self.backbone.config.patch_size
        self.num_components = 256  # Fixed number of GMM components

        # 3D Positional encoding MLP
        # Input: (x, y, z) coordinates, output: hidden_dim
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, self.pos_encoding_dim),
        )

        # Language token encoder
        self.use_text_embedding = model_cfg.use_text_embedding
        if self.use_text_embedding:
            self.text_encoder = nn.Sequential(
                nn.Linear(1152, 256),  # SIGLIP input dim
                nn.ReLU(),
                nn.Linear(256, self.hidden_dim),
            )

        # Gripper token encoder (6DoF pose + gripper width = 7 dims)
        self.use_gripper_token = model_cfg.use_gripper_token
        if self.use_gripper_token:
            self.gripper_encoder = nn.Sequential(
                nn.Linear(7, 128),
                nn.ReLU(),
                nn.Linear(128, self.hidden_dim),
            )

        # Transformer blocks (self-attention only)
        self.num_layers = model_cfg.get("num_transformer_layers", 4)
        self.transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Output head: predicts 13 dims per component (12 for 4×3 coords + 1 weight)
        self.output_head = nn.Linear(self.hidden_dim, 13)

    def get_patch_centers(self, H, W, intrinsics, depth):
        """
        Compute 3D coordinates for patch centers using depth.
        Args:
            H, W: image height and width
            intrinsics: (B, 3, 3) camera intrinsics
            depth: (B, H, W) depth map
        Returns:
            patch_coords: (B, num_patches, 3) 3D coordinates
            valid_mask: (B, num_patches) mask for valid depth
        """
        B = depth.shape[0]
        device = depth.device

        # Calculate patch grid size (DINOv2 uses 16×16 patches for 224×224 image)
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_patches = h_patches * w_patches

        # Get center pixel of each patch
        y_centers = (
            torch.arange(h_patches, device=device) * self.patch_size
            + self.patch_size // 2
        )
        x_centers = (
            torch.arange(w_patches, device=device) * self.patch_size
            + self.patch_size // 2
        )
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")

        # Flatten to (num_patches, 2)
        pixel_coords = torch.stack(
            [xx.flatten(), yy.flatten()], dim=1
        )  # (num_patches, 2)

        # Sample depth at patch centers
        pixel_coords_batch = pixel_coords.unsqueeze(0).expand(
            B, -1, -1
        )  # (B, num_patches, 2)
        y_idx = pixel_coords_batch[:, :, 1].long()
        x_idx = pixel_coords_batch[:, :, 0].long()

        depth_values = depth[
            torch.arange(B, device=device).unsqueeze(1), y_idx, x_idx
        ]  # (B, num_patches)

        # Unproject to 3D
        fx = intrinsics[:, 0, 0].unsqueeze(1)  # (B, 1)
        fy = intrinsics[:, 1, 1].unsqueeze(1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)

        x_3d = (pixel_coords_batch[:, :, 0] - cx) * depth_values / fx
        y_3d = (pixel_coords_batch[:, :, 1] - cy) * depth_values / fy
        z_3d = depth_values

        patch_coords = torch.stack(
            [x_3d, y_3d, z_3d], dim=2
        ).float()  # (B, num_patches, 3)

        return patch_coords

    def forward(
        self, image, depth, intrinsics, gripper_token=None, text_embedding=None
    ):
        """
        Args:
            image: (B, 3, H, W) RGB image
            depth: (B, H, W) depth map
            intrinsics: (B, 3, 3) camera intrinsics
            gripper_token: (B, 7) [6DoF pose + gripper width]
            text_embedding: (B, 1152) SigLIP embedding
        Returns:
            outputs: (B, 256, 13) GMM parameters
        """
        B, _, H, W = image.shape

        # Extract DINOv2 features
        with torch.no_grad():
            inputs = self.backbone_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.backbone.device) for k, v in inputs.items()}
            dino_outputs = self.backbone(**inputs)

        # Get patch features (skip CLS token)
        patch_features = dino_outputs.last_hidden_state[
            :, 1:
        ]  # (B, 256, dino_hidden_dim)

        # Get 3D positional encoding for patches
        patch_coords = self.get_patch_centers(H, W, intrinsics, depth)
        pos_encoding = self.pos_encoder(patch_coords)  # (B, 256, 128)

        # Combine patch features with positional encoding
        tokens = torch.cat(
            [patch_features, pos_encoding], dim=-1
        )  # (B, 256, hidden_dim)

        # Add language token
        if self.use_text_embedding and text_embedding is not None:
            lang_token = self.text_encoder(text_embedding).unsqueeze(
                1
            )  # (B, 1, hidden_dim)
            tokens = torch.cat([tokens, lang_token], dim=1)  # (B, 257, hidden_dim)

        # Add gripper token
        if self.use_gripper_token and gripper_token is not None:
            grip_token = self.gripper_encoder(gripper_token).unsqueeze(
                1
            )  # (B, 1, hidden_dim)
            tokens = torch.cat([tokens, grip_token], dim=1)  # (B, 258, hidden_dim)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Take only the first 256 tokens (throw away language and gripper tokens)
        tokens = tokens[:, :256]  # (B, 256, hidden_dim)

        # Predict GMM parameters
        outputs = self.output_head(tokens)  # (B, 256, 13)

        return outputs, patch_coords


class Dino3DGPGoalRegressionModule(pl.LightningModule):
    """
    A goal generation module for 3D goal prediction with RGB+Depth.
    Similar to articubot.py but uses DINOv2 with RGB+depth instead of PointNet++.
    """

    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.prediction_type = self.model_cfg.type
        self.mode = cfg.mode  # train or eval
        self.val_outputs: defaultdict[str, List[Dict]] = defaultdict(list)
        self.train_outputs: List[Dict] = []
        self.predict_outputs: defaultdict[str, List[Dict]] = defaultdict(list)
        self.predict_weighted_displacements: defaultdict[str, List[Dict]] = defaultdict(
            list
        )

        self.fixed_variance = cfg.model.fixed_variance
        self.uniform_weights_coeff = cfg.model.uniform_weights_coeff
        self.is_gmm = cfg.model.is_gmm

        if self.prediction_type != "cross_displacement":
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")
        self.label_key = "cross_displacement"

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
        self.max_depth = cfg.dataset.max_depth

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

    def extract_gt_4_points(self, batch):
        """Extract ground truth goal points (4 gripper points)"""
        cross_displacement = batch[self.label_key].points_padded()
        initial_gripper = batch["action_pcd"].points_padded()
        ground_truth_gripper = initial_gripper + cross_displacement
        batch_indices = torch.arange(
            ground_truth_gripper.shape[0], device=ground_truth_gripper.device
        ).unsqueeze(1)

        # Select specific idxs to compute the loss over
        gt_primary_points = ground_truth_gripper[batch_indices, batch["gripper_idx"], :]
        # Assumes 0/1 are tips to be averaged
        gt_extra_point = (gt_primary_points[:, 0, :] + gt_primary_points[:, 1, :]) / 2
        gt = torch.cat([gt_primary_points, gt_extra_point[:, None, :]], dim=1)

        init_primary_points = initial_gripper[batch_indices, batch["gripper_idx"], :]
        init_extra_point = (
            init_primary_points[:, 0, :] + init_primary_points[:, 1, :]
        ) / 2
        init = torch.cat([init_primary_points, init_extra_point[:, None, :]], dim=1)
        return init, gt

    def get_gripper_token(self, batch):
        """
        Extract gripper state as a token (6DoF pose + gripper width).
        TODO: Implement proper 6DoF pose extraction, for now use placeholder.
        """
        B = batch["action_pcd"].points_padded().shape[0]
        device = batch["action_pcd"].points_padded().device

        # Placeholder: return zeros for now
        # TODO: Extract proper 6DoF pose from action_pcd or batch
        gripper_token = torch.zeros(B, 7, device=device)
        return gripper_token

    def nll_loss(
        self,
        pred_displacement,
        gt_displacement,
        weights,
        valid_mask,
        variance,
        use_weights=True,
    ):
        """
        Negative log-likelihood loss for GMM.
        Similar to articubot.py but adapted for 256 fixed components.
        """
        batch_size, num_components = pred_displacement.shape[:2]

        # Mask invalid components
        weights = weights.masked_fill(~valid_mask, float("-inf"))
        if use_weights is False:
            # Overwrite with uniform weights
            weights = weights.masked_fill(valid_mask, 1)

        # Compute Gaussian log-likelihoods
        diff = (pred_displacement - gt_displacement).reshape(
            batch_size, num_components, -1
        )  # Shape: (B, 256, 12)
        exponent = -0.5 * torch.sum((diff**2) / variance, dim=2)  # Shape: (B, 256)
        log_gaussians = exponent

        # Compute log mixing coefficients
        log_mixing_coeffs = torch.log_softmax(weights, dim=1)  # (B, 256)
        log_mixing_coeffs = torch.clamp(log_mixing_coeffs, min=-10)

        masked_sum = log_gaussians + log_mixing_coeffs  # [B, 256]
        masked_sum = masked_sum.masked_fill(~valid_mask, -1e9)

        max_log = torch.max(masked_sum, dim=1, keepdim=True).values  # (B, 1)
        log_probs = max_log.squeeze(1) + torch.logsumexp(
            masked_sum - max_log, dim=1
        )  # B,

        nll_loss = -torch.mean(log_probs)
        return nll_loss

    def forward(self, batch):
        """Forward pass with GMM loss"""
        initial_gripper = batch["action_pcd"].points_padded()
        text_embedding = batch["text_embed"]

        # Get gripper token (6DoF + width)
        gripper_token = self.get_gripper_token(batch)

        # RGBs is [B, 2, H, W, 3], depths is [B, 2, H, W]
        # Use the first image/depth for prediction
        rgb = batch["rgbs"][:, 0].permute(0, 3, 1, 2)  # (B, 3, H, W)
        depth = batch["depths"][:, 0]  # (B, H, W)
        depth[depth > self.max_depth] = 0
        intrinsics = batch["intrinsics"]  # (B, 3, 3)

        # Forward through network
        outputs, patch_coords = self.network(
            rgb,
            depth,
            intrinsics,
            gripper_token=gripper_token,
            text_embedding=text_embedding,
        )

        # outputs: (B, 256, 13) - last dim is [12 coords + 1 weight]
        B, num_components, _ = outputs.shape
        init, gt = self.extract_gt_4_points(batch)

        # Parse outputs
        pred_displacement = outputs[:, :, :-1].reshape(
            B, num_components, 4, 3
        )  # (B, 256, 4, 3)
        weights = outputs[:, :, -1]  # (B, 256)

        # Predictions are residuals from patch centers, add them to get absolute positions
        # Expand patch_coords to match pred shape
        patch_coords_expanded = patch_coords[:, :, None, :]  # (B, 256, 1, 3)
        pred = patch_coords_expanded + pred_displacement  # Residual to absolute

        # GT displacement relative to patch centers
        gt_displacement = gt[:, None, :, :] - patch_coords_expanded  # (B, 256, 4, 3)

        # All components are valid (256 fixed components)
        valid_mask = torch.ones(
            B, num_components, device=outputs.device, dtype=torch.bool
        )

        # Compute GMM loss
        if self.is_gmm:
            loss = 0
            for var in self.fixed_variance:
                loss += self.nll_loss(
                    pred_displacement,
                    gt_displacement,
                    weights,
                    valid_mask,
                    var,
                    use_weights=True,
                )
                loss += self.uniform_weights_coeff * self.nll_loss(
                    pred_displacement,
                    gt_displacement,
                    weights,
                    valid_mask,
                    var,
                    use_weights=False,
                )
        else:
            # Simple MSE loss (if not using GMM)
            # Get weighted prediction
            weights_norm = F.softmax(weights, dim=1)
            pred_points = (weights_norm[:, :, None, None] * pred_displacement).sum(
                dim=1
            )
            loss = F.mse_loss(pred_points, gt)

        return None, loss

    def training_step(self, batch, batch_idx):
        """Training step with 3D GMM prediction"""
        assert (
            batch["augment_t"].mean().item() == 0
        ), "Disable pcd augmentations when training image model!"

        self.train()
        batch_size = batch[self.label_key].points_padded().shape[0]

        _, loss = self(batch)
        train_metrics = {"loss": loss}

        # Additional logging
        do_additional_logging = (
            self.global_step % self.additional_train_logging_period == 0
            and self.global_step != 0
        )

        if do_additional_logging:
            n_samples_wta = self.run_cfg.n_samples_wta
            self.eval()
            with torch.no_grad():
                all_pred_dict = []
                if self.is_gmm:
                    for i in range(n_samples_wta):
                        all_pred_dict.append(self.predict(batch))
                else:
                    all_pred_dict = [self.predict(batch)]

                pred_dict, weighted_displacement = all_pred_dict[0]
                pred_dict[self.prediction_type]["all_pred"] = [
                    i[0][self.prediction_type]["pred"] for i in all_pred_dict
                ]
                pred_dict[self.prediction_type]["all_pred"] = torch.stack(
                    pred_dict[self.prediction_type]["all_pred"]
                ).permute(1, 0, 2, 3)
            self.train()

            init, gt = self.extract_gt_4_points(batch)
            gt_displacement = gt - init

            padding_mask = torch.ones(gt.shape[0], gt.shape[1]).bool()
            pcd_std = batch["pcd_std"]
            pred_dict = calc_pcd_metrics(
                pred_dict,
                init,
                pred_dict[self.prediction_type]["all_pred"],
                gt_displacement,
                pcd_std,
                padding_mask,
            )
            train_metrics.update(pred_dict)

            if self.trainer.is_global_zero:
                self.log_viz_to_wandb(batch, pred_dict, weighted_displacement, "train")

        self.train_outputs.append(train_metrics)
        return loss

    @torch.no_grad()
    def predict(self, batch, progress=False):
        """
        Predict 3D goal points using GMM sampling.
        Returns displacement from initial gripper position.
        """
        initial_gripper = batch["action_pcd"].points_padded()
        text_embedding = batch["text_embed"]
        gripper_token = self.get_gripper_token(batch)

        # Get inputs
        rgb = batch["rgbs"][:, 0].permute(0, 3, 1, 2)
        depth = batch["depths"][:, 0]
        intrinsics = batch["intrinsics"]

        # Forward
        outputs, patch_coords = self.network(
            rgb,
            depth,
            intrinsics,
            gripper_token=gripper_token,
            text_embedding=text_embedding,
        )

        init, gt = self.extract_gt_4_points(batch)

        if self.is_gmm:
            pred = self.sample_from_gmm(outputs, patch_coords)
        else:
            pred = self.get_weighted_prediction(outputs, patch_coords)

        pred_displacement = pred - init
        return {self.prediction_type: {"pred": pred_displacement}}, outputs

    def sample_from_gmm(self, outputs, patch_coords):
        """
        Sample from GMM by selecting a component and using its mean.
        Args:
            outputs: (B, 256, 13) GMM parameters
            patch_coords: (B, 256, 3) patch center coordinates
        Returns:
            pred_points: (B, 4, 3) sampled goal points
        """
        B, num_components, _ = outputs.shape
        device = outputs.device

        # Parse outputs
        pred_displacement = outputs[:, :, :-1].reshape(B, num_components, 4, 3)
        weights = outputs[:, :, -1]

        # Softmax weights
        weights = F.softmax(weights, dim=1)

        # Sample component indices
        sampled_indices = torch.multinomial(weights, num_samples=1)  # (B, 1)
        batch_indices = torch.arange(B, device=device).unsqueeze(1)

        # Get sampled displacement and add to patch center
        sampled_disp = pred_displacement[batch_indices, sampled_indices].squeeze(
            1
        )  # (B, 4, 3)
        sampled_patch = patch_coords[batch_indices, sampled_indices].squeeze(
            1
        )  # (B, 3)

        pred_points = sampled_disp + sampled_patch.unsqueeze(1)  # (B, 4, 3)
        return pred_points

    def get_weighted_prediction(self, outputs, patch_coords):
        """Get weighted average prediction (non-GMM mode)"""
        B, num_components, _ = outputs.shape

        pred_displacement = outputs[:, :, :-1].reshape(B, num_components, 4, 3)
        weights = F.softmax(outputs[:, :, -1], dim=1)

        # Weighted average
        patch_coords_expanded = patch_coords[:, :, None, :]
        pred_abs = pred_displacement + patch_coords_expanded
        pred_points = (weights[:, :, None, None] * pred_abs).sum(dim=1)
        return pred_points

    def log_viz_to_wandb(self, batch, pred_dict, weighted_displacement, tag):
        """
        Log 3D visualizations to wandb (similar to articubot.py).
        """
        batch_size = batch[self.label_key].points_padded().shape[0]
        viz_idx = np.random.randint(0, batch_size)
        RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)
        max_depth = self.max_depth

        all_pred = pred_dict[self.prediction_type]["all_pred"][viz_idx].cpu().numpy()
        N = all_pred.shape[0]
        end2start = np.linalg.inv(batch["start2end"][viz_idx].cpu().numpy())

        if N == 1:
            BLUES = [BLUE]
        else:
            BLUES = [
                (int(200 * (1 - i / (N - 1))), int(220 * (1 - i / (N - 1))), 255)
                for i in range(N)
            ]

        goal_text = batch["caption"][viz_idx]
        vid_name = batch["vid_name"][viz_idx]
        rmse = pred_dict["rmse"][viz_idx]

        pcd, gt = self.extract_gt_4_points(batch)
        pcd, gt = pcd.cpu().numpy()[viz_idx], gt.cpu().numpy()[viz_idx]
        all_pred_pcd = pcd + all_pred
        gt_pcd = gt
        padding_mask = torch.ones(gt.shape[0]).bool().numpy()

        # Invert augmentation transforms before viz
        pcd_mean = batch["pcd_mean"][viz_idx].cpu().numpy()
        pcd_std = batch["pcd_std"][viz_idx].cpu().numpy()
        R = batch["augment_R"][viz_idx].cpu().numpy()
        t = batch["augment_t"][viz_idx].cpu().numpy()
        scene_centroid = batch["augment_C"][viz_idx].cpu().numpy()

        pcd = invert_augmentation_and_normalization(
            pcd, pcd_mean, pcd_std, R, t, scene_centroid
        )
        all_pred_pcd = invert_augmentation_and_normalization(
            all_pred_pcd, pcd_mean, pcd_std, R, t, scene_centroid
        )
        gt_pcd = invert_augmentation_and_normalization(
            gt_pcd, pcd_mean, pcd_std, R, t, scene_centroid
        )

        # Transform to end frame
        pcd_endframe = np.hstack((pcd, np.ones((pcd.shape[0], 1))))
        pcd_endframe = (end2start @ pcd_endframe.T).T[:, :3]
        all_pred_pcd_tmp = []
        for i in range(N):
            tmp_pcd = np.hstack((all_pred_pcd[i], np.ones((all_pred_pcd.shape[1], 1))))
            tmp_pcd = (end2start @ tmp_pcd.T).T[:, :3]
            all_pred_pcd_tmp.append(tmp_pcd)
        all_pred_pcd = np.stack(all_pred_pcd_tmp)
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

        # Project tracks to image
        init_rgb_proj = project_pcd_on_image(pcd, padding_mask, rgb_init, K, GREEN)
        end_rgb_proj = project_pcd_on_image(gt_pcd, padding_mask, rgb_end, K, RED)
        pred_rgb_proj = project_pcd_on_image(
            all_pred_pcd[-1], padding_mask, rgb_end, K, BLUE
        )
        rgb_proj_viz = cv2.hconcat([init_rgb_proj, end_rgb_proj, pred_rgb_proj])

        wandb_proj_img = wandb.Image(
            rgb_proj_viz,
            caption=f"Left: Initial Frame (GT Track)\n; Middle: Final Frame (GT Track)\n\
            ; Right: Final Frame (Pred Track)\n; Goal Description : {goal_text};\n\
            rmse={rmse};\nvideo path = {vid_name}; ",
        )

        # Visualize point cloud
        viz_pcd, _ = get_img_and_track_pcd(
            rgb_end,
            depth_end,
            K,
            padding_mask,
            pcd_endframe,
            gt_pcd,
            all_pred_pcd,
            GREEN,
            RED,
            BLUES,
            max_depth,
            4096,
        )

        viz_dict = {
            f"{tag}/track_projected_to_rgb": wandb_proj_img,
            f"{tag}/image_and_tracks_pcd": wandb.Object3D(viz_pcd),
            "trainer/global_step": self.global_step,
        }

        wandb.log(viz_dict)

    def on_train_epoch_end(self):
        if len(self.train_outputs) == 0:
            return

        log_dictionary = {}
        loss = torch.stack([x["loss"] for x in self.train_outputs]).mean()
        log_dictionary["train/loss"] = loss

        def mean_metric(metric_name):
            return torch.stack(
                [x[metric_name].mean() for x in self.train_outputs if metric_name in x]
            ).mean()

        if any("rmse" in x for x in self.train_outputs):
            log_dictionary["train/rmse"] = mean_metric("rmse")
            log_dictionary["train/wta_rmse"] = mean_metric("wta_rmse")
            log_dictionary["train/chamfer_dist"] = mean_metric("chamfer_dist")
            log_dictionary["train/wta_chamfer_dist"] = mean_metric("wta_chamfer_dist")
            log_dictionary["train/sample_std"] = mean_metric("sample_std")

        self.log_dict(
            log_dictionary,
            add_dataloader_idx=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_outputs.clear()

    def on_validation_epoch_start(self):
        self.random_val_viz_idx = {
            k: random.randint(0, len(v) - 1)
            for k, v in self.trainer.val_dataloaders.items()
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step for 3D goal prediction"""
        val_tag = self.trainer.datamodule.val_tags[dataloader_idx]
        n_samples_wta = self.run_cfg.n_samples_wta
        self.eval()
        with torch.no_grad():
            all_pred_dict = []
            if self.is_gmm:
                for i in range(n_samples_wta):
                    all_pred_dict.append(self.predict(batch))
            else:
                all_pred_dict = [self.predict(batch)]
            pred_dict, weighted_displacement = all_pred_dict[0]

            pred_dict[self.prediction_type]["all_pred"] = [
                i[0][self.prediction_type]["pred"] for i in all_pred_dict
            ]
            pred_dict[self.prediction_type]["all_pred"] = torch.stack(
                pred_dict[self.prediction_type]["all_pred"]
            ).permute(1, 0, 2, 3)

        init, gt = self.extract_gt_4_points(batch)
        gt_displacement = gt - init

        padding_mask = torch.ones(gt.shape[0], gt.shape[1]).bool()
        pcd_std = batch["pcd_std"]
        pred_dict = calc_pcd_metrics(
            pred_dict,
            init,
            pred_dict[self.prediction_type]["all_pred"],
            gt_displacement,
            pcd_std,
            padding_mask,
        )
        self.val_outputs[val_tag].append(pred_dict)

        if (
            batch_idx == self.random_val_viz_idx[val_tag]
            and self.trainer.is_global_zero
        ):
            self.log_viz_to_wandb(
                batch, pred_dict, weighted_displacement, f"val_{val_tag}"
            )
        return pred_dict

    def on_validation_epoch_end(self):
        log_dict = {}
        all_metrics = {
            "rmse": [],
            "wta_rmse": [],
            "chamfer_dist": [],
            "wta_chamfer_dist": [],
            "sample_std": [],
        }

        for val_tag in self.trainer.datamodule.val_tags:
            val_outputs = self.val_outputs[val_tag]
            tag_metrics = {}

            if len(val_outputs) == 0:
                continue

            for metric in all_metrics.keys():
                values = torch.stack([x[metric].mean() for x in val_outputs]).mean()
                tag_metrics[metric] = values
                all_metrics[metric].append(values)

            for metric, value in tag_metrics.items():
                log_dict[f"val_{val_tag}/{metric}"] = value

        for metric, values in all_metrics.items():
            log_dict[f"val/{metric}"] = torch.stack(values).mean()

        # Combined metric
        alpha = 0.95
        log_dict["val/rmse_and_std_combi"] = alpha * log_dict["val/rmse"] + (
            1 - alpha
        ) * (-log_dict["val/sample_std"])

        self.log_dict(
            log_dict,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.val_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for model evaluation"""
        eval_tag = self.trainer.datamodule.eval_tags[dataloader_idx]
        n_samples_wta = self.trainer.datamodule.n_samples_wta

        all_pred_dict = []
        if self.is_gmm:
            for i in range(n_samples_wta):
                all_pred_dict.append(self.predict(batch))
        else:
            all_pred_dict = [self.predict(batch)]

        pred_dict, weighted_displacement = all_pred_dict[0]
        pred_dict[self.prediction_type]["all_pred"] = [
            i[0][self.prediction_type]["pred"] for i in all_pred_dict
        ]
        pred_dict[self.prediction_type]["all_pred"] = torch.stack(
            pred_dict[self.prediction_type]["all_pred"]
        ).permute(1, 0, 2, 3)

        init, gt = self.extract_gt_4_points(batch)
        gt_displacement = gt - init

        padding_mask = torch.ones(gt.shape[0], gt.shape[1]).bool()
        pcd_std = batch["pcd_std"]
        pred_dict = calc_pcd_metrics(
            pred_dict,
            init,
            pred_dict[self.prediction_type]["all_pred"],
            gt_displacement,
            pcd_std,
            padding_mask,
        )
        self.predict_outputs[eval_tag].append(pred_dict)
        self.predict_weighted_displacements[eval_tag].append(
            weighted_displacement.cpu()
        )

        return {
            "rmse": pred_dict["rmse"],
            "chamfer_dist": pred_dict["chamfer_dist"],
            "wta_rmse": pred_dict["wta_rmse"],
            "wta_chamfer_dist": pred_dict["wta_chamfer_dist"],
            "vid_name": batch["vid_name"],
            "caption": batch["caption"],
        }

    def on_predict_epoch_end(self):
        """Stub - implement if needed"""
        pass
