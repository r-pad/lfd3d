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
from matplotlib import cm
from torch import nn, optim
from transformers import AutoImageProcessor, AutoModel


def get_heatmap_viz(rgb_image, heatmap, alpha=0.4):
    """
    Overlay heatmap on RGB image with transparency.

    Args:
        rgb_image: (H, W, 3) RGB image
        heatmap: (1, H, W) predicted heatmap
        alpha: transparency factor for heatmap overlay

    Returns:
        wandb.Image: RGB image with heatmap overlay
    """
    # Get single heatmap
    heatmap = heatmap.squeeze().cpu().numpy()  # (H, W)

    # Normalize heatmap to 0-1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Convert to colormap (jet colormap)
    colormap = cm.jet(heatmap)[:, :, :3]  # (H, W, 3)
    colormap = (colormap * 255).astype(np.uint8)

    # Blend with original image
    overlay = cv2.addWeighted(rgb_image, 1 - alpha, colormap, alpha, 0)

    return wandb.Image(overlay)


def calc_pix_metrics(pred_dict, gt_idx, all_pred_idx, img_shape):
    """
    Calculate pixel distance metrics and update pred_dict with the keys.

    pred_dict: Dictionary with keys to be updated
    gt_idx: GT pixel location (B, 2) [x, y]
    all_pred_idx: Predicted goal pixel locations of multiple samples (B, N, 2)
    img_shape: (H, W) image shape for normalization
    """
    H, W = img_shape

    # Calculate L1 distances for all samples
    # gt_idx: (B, 2), all_pred_idx: (B, N, 2)
    gt_expanded = gt_idx.unsqueeze(1)  # (B, 1, 2)

    # L1 distance in pixel space
    pix_distances = torch.sum(
        torch.abs(all_pred_idx.float() - gt_expanded.float()), dim=2
    )  # (B, N)

    # Normalize by max Manhattan distance for 0-1 range
    max_dist = H + W
    normalized_distances = pix_distances / max_dist

    # Use first sample for single prediction metrics
    pred_dict["pix_dist"] = pix_distances[:, 0]  # (B,)
    pred_dict["normalized_pix_dist"] = normalized_distances[:, 0]  # (B,)

    # Winner-takes-all: best sample across all predictions
    pred_dict["wta_pix_dist"] = pix_distances.min(dim=1)[0]  # (B,)
    pred_dict["wta_normalized_pix_dist"] = normalized_distances.min(dim=1)[0]  # (B,)

    return pred_dict


class SimplePointNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimplePointNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Make translation invariant
        centroid = x.mean(dim=1, keepdim=True)
        x = x - centroid

        # x: (B, N, input_dim) -> (B, input_dim, N)
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        # Global max pooling
        x = torch.max(x, 2)[0]  # (B, output_dim)
        return x


class DinoHeatmapNetwork(nn.Module):
    """
    Dino + DPT network for dense heatmap prediction
    """

    def __init__(self, model_cfg):
        super(DinoHeatmapNetwork, self).__init__()

        # Dino backbone
        self.backbone_processor = AutoImageProcessor.from_pretrained(
            model_cfg.dino_model
        )
        self.backbone = AutoModel.from_pretrained(model_cfg.dino_model)
        self.backbone.requires_grad_(False)  # Freeze backbone

        # Get backbone dimensions
        hidden_dim = self.backbone.config.hidden_size

        # Point encoder for hand/gripper poses
        self.use_gripper_pcd = model_cfg.use_gripper_pcd
        self.encoded_point_dim = 128
        if self.use_gripper_pcd:
            self.point_encoder = SimplePointNet(3, self.encoded_point_dim)

        # Language conditioning
        self.use_text_embedding = model_cfg.use_text_embedding
        self.encoded_text_dim = 128
        if self.use_text_embedding:
            self.text_encoder = nn.Linear(
                1152, self.encoded_text_dim
            )  # SIGLIP input dim

        # Cross-attention fusion with layer norm for stability
        if self.use_text_embedding:
            self.text_cross_attn = nn.MultiheadAttention(
                hidden_dim, 8, batch_first=True
            )
            self.text_norm_pre = nn.LayerNorm(hidden_dim)
            self.text_norm_post = nn.LayerNorm(hidden_dim)

        if self.use_gripper_pcd:
            self.point_cross_attn = nn.MultiheadAttention(
                hidden_dim, 8, batch_first=True
            )
            self.point_norm_pre = nn.LayerNorm(hidden_dim)
            self.point_norm_post = nn.LayerNorm(hidden_dim)

        # Project conditioning features to hidden_dim
        if self.use_text_embedding:
            self.text_proj = nn.Linear(self.encoded_text_dim, hidden_dim)
        if self.use_gripper_pcd:
            self.point_proj = nn.Linear(self.encoded_point_dim, hidden_dim)

        # DPT-style decoder
        self.decoder = nn.Sequential(
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, 256, 3, 1, 1),
            nn.ReLU(),
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            # 64x64 -> 128x128
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            # 128x128 -> 224x224
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 1, 3, 1, 1),
        )

        # Initialize cross-attention weights for stability
        self._init_cross_attention_weights()

    def _init_cross_attention_weights(self):
        """Initialize cross-attention layers with smaller weights for training stability"""
        if self.use_text_embedding:
            # Scale down attention weights
            nn.init.xavier_uniform_(self.text_cross_attn.in_proj_weight, gain=0.1)
            nn.init.xavier_uniform_(self.text_cross_attn.out_proj.weight, gain=0.1)

        if self.use_gripper_pcd:
            # Scale down attention weights
            nn.init.xavier_uniform_(self.point_cross_attn.in_proj_weight, gain=0.1)
            nn.init.xavier_uniform_(self.point_cross_attn.out_proj.weight, gain=0.1)

    def forward(self, image, gripper_pcd=None, text_embedding=None):
        # Extract features from Dino
        with torch.no_grad():
            inputs = self.backbone_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.backbone.device) for k, v in inputs.items()}
            outputs = self.backbone(**inputs)

        features = outputs.last_hidden_state[:, 1:]  # skip CLS token, (B, 256, 768)
        B, L, D = features.shape

        # Cross-attention fusion
        fused = features

        # Text cross-attention
        if self.use_text_embedding and text_embedding is not None:
            text_feat = self.text_encoder(text_embedding)  # (B, 128)
            text_feat = self.text_proj(text_feat)  # (B, hidden_dim)
            text_feat = text_feat.unsqueeze(1)  # (B, 1, hidden_dim)

            # Cross-attention: query=visual_features, key/value=text_features
            normed_fused = self.text_norm_pre(fused)
            attn_out, _ = self.text_cross_attn(normed_fused, text_feat, text_feat)
            fused = self.text_norm_post(fused + attn_out)  # Residual connection

            # fused, _ = self.text_cross_attn(fused, text_feat, text_feat)

        # Point cross-attention
        if self.use_gripper_pcd and gripper_pcd is not None:
            point_feat = self.point_encoder(gripper_pcd)  # (B, 128)
            point_feat = self.point_proj(point_feat)  # (B, hidden_dim)
            point_feat = point_feat.unsqueeze(1)  # (B, 1, hidden_dim)

            # Cross-attention: query=visual_features, key/value=point_features
            normed_fused = self.point_norm_pre(fused)
            attn_out, _ = self.point_cross_attn(normed_fused, point_feat, point_feat)
            fused = self.point_norm_post(fused + attn_out)  # Residual connection

            # fused, _ = self.point_cross_attn(fused, point_feat, point_feat)

        # Reshape to spatial format (assuming square patches)
        h = w = int(L**0.5)
        fused = fused.transpose(1, 2).reshape(B, D, h, w)

        # Decode to heatmap
        heatmap = self.decoder(fused)  # (B, 1, H, W)

        return heatmap


class HeatmapSamplerModule(pl.LightningModule):
    """
    A goal generation module that handles model training, inference, evaluation and visualization.
    Based on GoalRegressionModule but reworked to use an image-based high-level model.
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
        if self.prediction_type != "heatmap":
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")
        self.label_key = "heatmap"

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
            self.loss_type = self.run_cfg.loss_type
        elif self.mode == "eval":
            self.run_cfg = cfg.inference
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # data params
        self.batch_size = self.run_cfg.batch_size
        self.val_batch_size = self.run_cfg.val_batch_size

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
        cross_displacement = batch["cross_displacement"].points_padded()
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

    def compute_gt_mask(self, batch, gt, channels=1):
        assert (
            batch["augment_t"].mean().item() == 0
        ), "Disable pcd augmentations when training image model!"
        device = batch["rgbs"].device

        gt = gt[:, :channels, :]  # only project as many channels as we have in the mask
        batch_size, _, height, width, _ = batch["rgbs"].shape
        img_limits = torch.tensor([width - 1, height - 1], device=device)
        K = batch["intrinsics"].float()

        mask = torch.zeros((batch_size, height, width, channels), device=device)

        projected_points = torch.bmm(K, gt.transpose(-1, -2))
        projected_points = projected_points[:, :2, :] / projected_points[:, 2:, :]
        projected_image_coords = projected_points.transpose(-1, -2).round().int()

        coords = torch.clamp(
            projected_image_coords, torch.tensor([0], device=device), img_limits
        )

        batch_idx = (
            torch.arange(batch_size, device=coords.device).unsqueeze(1).expand(-1, 4)
        )
        y_coords = coords[..., 1].long()
        x_coords = coords[..., 0].long()

        mask[batch_idx, y_coords, x_coords] = 1
        return mask

    def create_gaussian_target(self, coords, H, W, sigma=3.0):
        """Create smooth Gaussian blob around target pixel coordinates"""
        device = coords.device
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # coords is (B, 2) -> (x, y)
        dx = x_grid[None] - coords[:, 0, None, None]  # (B, H, W)
        dy = y_grid[None] - coords[:, 1, None, None]  # (B, H, W)

        dist_sq = dx**2 + dy**2
        target_dist = torch.exp(-dist_sq / (2 * sigma**2))

        # Normalize to probability distribution
        return target_dist / target_dist.sum(dim=(2, 3), keepdim=True)

    def forward(self, batch):
        _, gt = self.extract_gt_4_points(batch)
        gt_mask = self.compute_gt_mask(batch, gt)

        text_embedding = batch["text_embed"]

        # RGBs is [B, 2, H, W, 3] -> [B, 3, H, W], we only use the first image
        outputs = self.network(
            batch["rgbs"][:, 0].permute(0, 3, 1, 2),
            gripper_pcd=batch["action_pcd"].points_padded(),
            text_embedding=text_embedding,
        )

        if self.loss_type == "cross_entropy":
            # Flatten heatmap for cross entropy
            logits = outputs.flatten(2).squeeze(1)  # (B, H*W)

            # Create target indices from gt_mask
            # Find the pixel with value 1 in gt_mask
            gt_flat = gt_mask.flatten(1, 2)  # (B, H*W)
            target_idx = gt_flat.argmax(
                dim=1
            ).squeeze()  # (B, ) - index of target pixel
            loss = F.cross_entropy(logits, target_idx)
        elif self.loss_type == "kl_div":  # Goes to NaN?
            B, _, H, W = outputs.shape

            # Get target pixel coordinates
            gt_flat = gt_mask.flatten(1, 2)  # (B, H*W)
            target_flat_idx = gt_flat.argmax(dim=1)  # (B,) - flat index of target pixel

            # Convert to 2D coordinates
            target_y = target_flat_idx // W
            target_x = target_flat_idx % W
            target_coords = torch.stack(
                [target_x, target_y], dim=1
            ).float()  # (B, 2) [x, y]

            # Random sigma for robustness (1-3 pixel tolerance)
            sigma = 1.0 + torch.rand(1).item() * 2.0  # Random between 1.0-3.0

            # Create smooth Gaussian target distribution
            target_dist = self.create_gaussian_target(
                target_coords, H, W, sigma=sigma
            )  # (B, H, W)

            # Flatten for softmax, then reshape back
            logits_flat = outputs.squeeze(1).flatten(1)  # (B, H*W)
            pred_dist_flat = F.softmax(logits_flat, dim=1)  # (B, H*W)
            pred_dist = pred_dist_flat.view(B, H, W)  # (B, H, W)

            # KL divergence loss
            loss = F.kl_div(pred_dist.log(), target_dist, reduction="batchmean")
        else:
            raise NotImplementedError
        return None, loss

    def training_step(self, batch, batch_idx):
        """
        Training step for the module. Logs training metrics and visualizations to wandb.
        """
        self.train()
        batch_size = batch["rgbs"].shape[0]

        _, loss = self(batch)
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
            n_samples_wta = self.run_cfg.n_samples_wta
            self.eval()
            with torch.no_grad():
                all_pred_dict = []
                for i in range(n_samples_wta):
                    all_pred_dict.append(self.predict(batch))
                pred_dict, heatmap = all_pred_dict[0]

                # Store all sample preds for viz
                pred_dict[self.prediction_type]["all_pred"] = [
                    i[0][self.prediction_type]["pred"] for i in all_pred_dict
                ]
                pred_dict[self.prediction_type]["all_pred"] = torch.stack(
                    pred_dict[self.prediction_type]["all_pred"]
                ).permute(1, 0, 2)
            self.train()  # Switch back to training mode

            _, gt = self.extract_gt_4_points(batch)
            gt_mask = self.compute_gt_mask(batch, gt)
            # Find the pixel with value 1 in gt_mask and convert to 2D coords
            gt_flat = gt_mask.flatten(1, 2)  # (B, H*W)
            flat_idx = gt_flat.argmax(
                dim=1
            ).squeeze()  # (B,) - flat index of target pixel
            H, W = gt_mask.shape[1], gt_mask.shape[2]
            y_coords = flat_idx // W
            x_coords = flat_idx % W
            target_idx = torch.stack([x_coords, y_coords], dim=1)  # (B, 2) [x, y]

            pred_dict = calc_pix_metrics(
                pred_dict,
                target_idx,
                pred_dict[self.prediction_type]["all_pred"],
                (H, W),
            )
            train_metrics.update(pred_dict)

            if self.trainer.is_global_zero:
                ####################################################
                # logging visualizations
                ####################################################
                self.log_viz_to_wandb(batch, pred_dict, heatmap, "train")

        self.train_outputs.append(train_metrics)
        return loss

    @torch.no_grad()
    def predict(self, batch, progress=False):
        """
        Compute prediction for a given batch.

        Args:
            batch: the input batch
            progress: whether to show progress bar
        """
        _, gt = self.extract_gt_4_points(batch)
        gt_mask = self.compute_gt_mask(batch, gt)

        text_embedding = batch["text_embed"]

        # RGBs is [B, 2, H, W, 3] -> [B, 3, H, W], we only use the first image
        outputs = self.network(
            batch["rgbs"][:, 0].permute(0, 3, 1, 2),
            gripper_pcd=batch["action_pcd"].points_padded(),
            text_embedding=text_embedding,
        )

        pred_idx = self.sample_from_heatmap(outputs)
        return {self.prediction_type: {"pred": pred_idx}}, outputs

    def sample_from_heatmap(self, heatmap):
        """
        Sample 2D pixel coordinates from heatmap using multinomial sampling.

        Args:
            heatmap: (B, 1, H, W) predicted heatmap

        Returns:
            coords: (B, 2) sampled pixel coordinates [x, y]
        """
        B, _, H, W = heatmap.shape

        # Flatten spatial dimensions and apply softmax
        logits = heatmap.squeeze(1).flatten(1)  # (B, H*W)
        probs = F.softmax(logits, dim=1)

        # Multinomial sampling
        sampled_indices = torch.multinomial(probs, 1).squeeze(1)  # (B,)

        # Convert flat indices back to 2D coordinates
        y_coords = sampled_indices // W
        x_coords = sampled_indices % W

        # Stack to get (B, 2) [x, y] format
        coords = torch.stack([x_coords, y_coords], dim=1)

        return coords

    def log_viz_to_wandb(self, batch, pred_dict, heatmap, tag):
        """
        Log visualizations to wandb.

        Args:
            batch: the input batch
            pred_dict: the prediction dictionary
            heatmap: the output of the dino-heatmap model
            tag: the tag to use for logging
        """
        batch_size = batch["rgbs"].shape[0]
        # pick a random sample in the batch to visualize
        viz_idx = np.random.randint(0, batch_size)
        RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)

        all_pred = pred_dict[self.prediction_type]["all_pred"][viz_idx].cpu().numpy()
        N = all_pred.shape[0]

        if N == 1:
            BLUES = [BLUE]
        else:
            # Multiple shades of blue for different samples
            BLUES = [
                (int(200 * (1 - i / (N - 1))), int(220 * (1 - i / (N - 1))), 255)
                for i in range(N)
            ]

        goal_text = batch["caption"][viz_idx]
        vid_name = batch["vid_name"][viz_idx]
        pix_dist = pred_dict["pix_dist"][viz_idx]
        heatmap_ = heatmap[viz_idx]

        rgb_init, rgb_end = (
            batch["rgbs"][viz_idx, 0].cpu().numpy(),
            batch["rgbs"][viz_idx, 1].cpu().numpy(),
        )

        # Get GT pixel coordinates
        _, gt = self.extract_gt_4_points(batch)
        gt_mask = self.compute_gt_mask(batch, gt)
        gt_flat = gt_mask.flatten(1, 2)
        flat_idx = gt_flat.argmax(dim=1)
        H, W = gt_mask.shape[1], gt_mask.shape[2]
        gt_y = (flat_idx[viz_idx] // W).item()
        gt_x = (flat_idx[viz_idx] % W).item()

        # Overlay pixel coords on images
        init_rgb_pix = rgb_init.copy()
        cv2.circle(init_rgb_pix, (gt_x, gt_y), 3, GREEN, -1)

        end_rgb_pix = rgb_end.copy()
        cv2.circle(end_rgb_pix, (gt_x, gt_y), 3, RED, -1)

        pred_rgb_pix = rgb_end.copy()
        # Draw all predicted samples
        for i, pred_coord in enumerate(all_pred):
            x, y = int(pred_coord[0]), int(pred_coord[1])
            color = BLUES[i]
            cv2.circle(pred_rgb_pix, (x, y), 3, color, -1)
        rgb_pix_viz = cv2.hconcat([init_rgb_pix, end_rgb_pix, pred_rgb_pix])

        wandb_pix_img = wandb.Image(
            rgb_pix_viz,
            caption=f"Left: Initial Frame (GT Coord)\n; Middle: Final Frame (GT Coord)\n\
            ; Right: Final Frame (Pred Coord)\n; Goal Description : {goal_text};\n\
            pix_dist={pix_dist};\nvideo path = {vid_name}; ",
        )
        ###

        wandb_heatmap_img = get_heatmap_viz(rgb_end, heatmap_)

        viz_dict = {
            f"{tag}/track_projected_to_rgb": wandb_pix_img,
            f"{tag}/heatmap_on_image": wandb_heatmap_img,
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

        if any("pix_dist" in x for x in self.train_outputs):
            log_dictionary["train/pix_dist"] = mean_metric("pix_dist")
            log_dictionary["train/wta_pix_dist"] = mean_metric("wta_pix_dist")
            log_dictionary["train/normalized_pix_dist"] = mean_metric(
                "normalized_pix_dist"
            )
            log_dictionary["train/wta_normalized_pix_dist"] = mean_metric(
                "wta_normalized_pix_dist"
            )

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
        self.random_val_viz_idx = {
            k: random.randint(0, len(v) - 1)
            for k, v in self.trainer.val_dataloaders.items()
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step for the module. Logs validation metrics and visualizations to wandb.
        """
        val_tag = self.trainer.datamodule.val_tags[dataloader_idx]
        n_samples_wta = self.run_cfg.n_samples_wta
        self.eval()
        with torch.no_grad():
            all_pred_dict = []
            for i in range(n_samples_wta):
                all_pred_dict.append(self.predict(batch))
            pred_dict, heatmap = all_pred_dict[0]

            # Store all sample preds for viz
            pred_dict[self.prediction_type]["all_pred"] = [
                i[0][self.prediction_type]["pred"] for i in all_pred_dict
            ]
            pred_dict[self.prediction_type]["all_pred"] = torch.stack(
                pred_dict[self.prediction_type]["all_pred"]
            ).permute(1, 0, 2)

        _, gt = self.extract_gt_4_points(batch)
        gt_mask = self.compute_gt_mask(batch, gt)
        # Find the pixel with value 1 in gt_mask and convert to 2D coords
        gt_flat = gt_mask.flatten(1, 2)  # (B, H*W)
        flat_idx = gt_flat.argmax(dim=1).squeeze(1)  # (B,) - flat index of target pixel
        H, W = gt_mask.shape[1], gt_mask.shape[2]
        y_coords = flat_idx // W
        x_coords = flat_idx % W
        target_idx = torch.stack([x_coords, y_coords], dim=1)  # (B, 2) [x, y]

        pred_dict = calc_pix_metrics(
            pred_dict,
            target_idx,
            pred_dict[self.prediction_type]["all_pred"],
            (H, W),
        )
        self.val_outputs[val_tag].append(pred_dict)

        ####################################################
        # logging visualizations
        ####################################################
        if (
            batch_idx == self.random_val_viz_idx[val_tag]
            and self.trainer.is_global_zero
        ):
            self.log_viz_to_wandb(batch, pred_dict, heatmap, f"val_{val_tag}")
        return pred_dict

    def on_validation_epoch_end(self):
        log_dict = {}
        all_metrics = {
            "pix_dist": [],
            "normalized_pix_dist": [],
            "wta_pix_dist": [],
            "wta_normalized_pix_dist": [],
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

            # Per dataset metrics
            for metric, value in tag_metrics.items():
                log_dict[f"val_{val_tag}/{metric}"] = value

        # Avg over all datasets
        for metric, values in all_metrics.items():
            log_dict[f"val/{metric}"] = torch.stack(values).mean()

        self.log_dict(
            log_dict,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.val_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for model evaluation.
        """
        raise NotImplementedError("TBD after referring to articubot.py")

    def on_predict_epoch_end(self):
        """
        Visualize random 5 batches in the test sets.
        """
        raise NotImplementedError("TBD after referring to articubot.py")
