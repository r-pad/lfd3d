import random
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np
import ot
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from diffusers import get_cosine_schedule_with_warmup
from pytorch3d.transforms import (
    euler_angles_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from torch import nn, optim
from transformers import AutoImageProcessor, AutoModel, T5EncoderModel, T5Tokenizer

from lfd3d.models.dino_heatmap import calc_pix_metrics
from lfd3d.models.tax3d import calc_pcd_metrics
from lfd3d.utils.viz_utils import (
    get_img_and_track_pcd,
    invert_augmentation_and_normalization,
    project_pcd_on_image,
)


class FourierPositionalEncoding(nn.Module):
    """Fourier feature positional encoding for 3D coordinates"""

    def __init__(self, input_dim=3, num_frequencies=64, include_input=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Frequency bands (geometric progression)
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)

        # Output dimension: input_dim * num_frequencies * 2 (sin + cos) + input_dim (if include_input)
        self.output_dim = input_dim * num_frequencies * 2
        if include_input:
            self.output_dim += input_dim

    def forward(self, coords):
        """
        Args:
            coords: (..., input_dim) 3D coordinates
        Returns:
            encoded: (..., output_dim) Fourier encoded features
        """
        # coords: (..., 3)
        # freq_bands: (num_frequencies,)

        # Compute angular frequencies
        # (..., 3, 1) * (num_frequencies,) -> (..., 3, num_frequencies)
        scaled = coords.unsqueeze(-1) * self.freq_bands

        # Apply sin and cos
        sin_features = torch.sin(2 * np.pi * scaled)  # (..., 3, num_frequencies)
        cos_features = torch.cos(2 * np.pi * scaled)  # (..., 3, num_frequencies)

        # Interleave and flatten
        fourier_features = torch.cat(
            [sin_features, cos_features], dim=-1
        )  # (..., 3, 2*num_frequencies)
        fourier_features = fourier_features.reshape(
            *coords.shape[:-1], -1
        )  # (..., 3*2*num_frequencies)

        if self.include_input:
            fourier_features = torch.cat([coords, fourier_features], dim=-1)

        return fourier_features


class Dino3DGPNetwork(nn.Module):
    """
    DINOv2 + 3D positional encoding + Transformer for 3D goal prediction
    Architecture:
    - Image tokens: DINOv2 patches with 3D PE (x,y,z from depth)
    - Language tokens: Flan-T5 (optional)
    - Gripper token: 6DoF pose + gripper width (optional)
    - Source token: learnable embedding for human/robot (optional)
    - Transformer: self-attention blocks
    - Output: N*196 GMM components, each predicting 13-dim (4×3 coords + 1 weight)
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

        # Training augmentations
        self.image_token_dropout = model_cfg.image_token_dropout

        # 3D Positional encoding
        if model_cfg.use_fourier_pe:
            # Fourier positional encoding
            fourier_encoder = FourierPositionalEncoding(
                input_dim=3,
                num_frequencies=model_cfg.fourier_num_frequencies,
                include_input=model_cfg.fourier_include_input,
            )
            fourier_dim = fourier_encoder.output_dim
            # Fourier encoder + MLP projection
            self.pos_encoder = nn.Sequential(
                fourier_encoder,
                nn.Linear(fourier_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.pos_encoding_dim),
            )
        else:
            # 3D Positional encoding MLP
            # Input: (x, y, z) coordinates, output: hidden_dim
            self.pos_encoder = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(),
                nn.Linear(128, self.pos_encoding_dim),
            )

        # Language encoder
        self.use_text_embedding = model_cfg.use_text_embedding
        if self.use_text_embedding:
            self.text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            self.text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-base")
            self.text_encoder.requires_grad_(False)  # Freeze
            self.text_proj = nn.Sequential(
                nn.Linear(768, 256),  # Flan-T5 output dim
                nn.ReLU(),
                nn.Linear(256, self.hidden_dim),
            )

        # Gripper token encoder (6DoF pose + gripper width = xyz + 6D rot + width = 10dims)
        self.use_gripper_token = model_cfg.use_gripper_token
        if self.use_gripper_token:
            self.gripper_encoder = nn.Sequential(
                nn.Linear(10, 128),
                nn.ReLU(),
                nn.Linear(128, self.hidden_dim),
            )

        # Source token (learnable embeddings for human/robot)
        self.use_source_token = model_cfg.use_source_token
        if self.use_source_token:
            # Learnable embeddings: 0 = human, 1 = robot
            self.source_to_idx = {
                "human": 0,
                "aloha": 1,
                "libero_franka": 2,
                "droid": 3,
            }
            self.source_embeddings = nn.Embedding(
                len(self.source_to_idx), self.hidden_dim
            )

        # Transformer blocks (self-attention only)
        self.num_layers = model_cfg.num_transformer_layers
        self.transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=model_cfg.dropout,
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Output head: predicts 13 dims per component (12 for 4×3 coords + 1 weight)
        self.output_head = nn.Linear(self.hidden_dim, 13)

        # Register tokens
        self.num_registers = 4
        self.registers = nn.Parameter(
            torch.randn(1, self.num_registers, self.hidden_dim) * 0.02
        )

    def apply_image_token_dropout(self, tokens, patch_coords, num_cameras):
        """
        Apply image token dropout during training.

        Args:
            tokens: (B, N*196, hidden_dim) image tokens
            patch_coords: (B, N*196, 3) patch coordinates
            num_cameras: N - number of cameras

        Returns:
            tokens: (B, T, hidden_dim) tokens after dropout
            patch_coords: (B, T, 3) patch coords after dropout
        """
        if not self.training or not self.image_token_dropout:
            return tokens, patch_coords

        B, total_tokens, hidden_dim = tokens.shape
        tokens_per_camera = 196
        device = tokens.device

        # Sample dropout strategy: 0.6 = no dropout, 0.3 = token dropout, 0.1 = camera dropout
        dropout_type = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]

        if dropout_type == 0:
            # No dropout
            return tokens, patch_coords
        elif dropout_type == 1:
            # Drop 0-30% of all tokens randomly
            dropout_ratio = random.uniform(0.0, 0.3)
            num_tokens_to_keep = int(total_tokens * (1 - dropout_ratio))

            indices = torch.stack(
                [
                    torch.randperm(total_tokens, device=device)[:num_tokens_to_keep]
                    for _ in range(B)
                ]
            )
            batch_idx = torch.arange(B, device=device)[:, None]

            tokens = tokens[batch_idx, indices]
            patch_coords = patch_coords[batch_idx, indices]

            return tokens, patch_coords
        else:
            # Drop one entire camera (only if more than one camera)
            if num_cameras > 1:
                # Randomly select a camera to drop
                camera_to_drop = random.randint(0, num_cameras - 1)
                start_idx = camera_to_drop * tokens_per_camera
                end_idx = start_idx + tokens_per_camera

                # Create mask to keep all tokens except from dropped camera
                mask = torch.ones(total_tokens, dtype=torch.bool, device=device)
                mask[start_idx:end_idx] = False

                # Apply mask
                tokens = tokens[:, mask, :]
                patch_coords = patch_coords[:, mask, :]
            return tokens, patch_coords

    def transform_to_world(self, points_cam, T_world_from_cam):
        """Transform points from camera frame to world frame.

        Args:
            points_cam: (B, N, 3) - points in camera frame
            T_world_from_cam: (B, 4, 4) - transformation matrix

        Returns:
            points_world: (B, N, 3) - points in world frame
        """
        B, N, _ = points_cam.shape
        # Convert to homogeneous coordinates
        ones = torch.ones(B, N, 1, device=points_cam.device)
        points_hom = torch.cat([points_cam, ones], dim=-1)  # (B, N, 4)

        # Apply transformation: (B, 4, 4) @ (B, N, 4) -> (B, N, 4)
        points_world_hom = torch.einsum("bij,bnj->bni", T_world_from_cam, points_hom)

        # Convert back to 3D
        points_world = points_world_hom[:, :, :3]  # (B, N, 3)

        return points_world

    def get_patch_centers(self, H, W, intrinsics, depth, extrinsics):
        """
        Compute 3D coordinates for patch centers using depth (multi-camera support).

        Args:
            H, W: image height and width
            intrinsics: (B, N, 3, 3) camera intrinsics for N cameras
            depth: (B, N, H, W) depth maps for N cameras
            extrinsics: (B, N, 4, 4) camera-to-world transforms

        Returns:
            patch_coords: (B, N*num_patches, 3) 3D coordinates in WORLD frame
        """
        B, N, _, _ = depth.shape
        device = depth.device

        # Calculate patch grid size (DINOv3 uses 16×16 patches for 224×224 image)
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_patches = h_patches * w_patches  # 196 for 224x224 with patch_size=16

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

        # Process each camera
        all_coords_world = []
        for cam_idx in range(N):
            # Sample depth at patch centers for this camera
            pixel_coords_batch = pixel_coords.unsqueeze(0).expand(
                B, -1, -1
            )  # (B, num_patches, 2)
            y_idx = pixel_coords_batch[:, :, 1].long()
            x_idx = pixel_coords_batch[:, :, 0].long()

            depth_cam = depth[:, cam_idx, :, :]  # (B, H, W)
            depth_values = depth_cam[
                torch.arange(B, device=device).unsqueeze(1), y_idx, x_idx
            ]  # (B, num_patches)

            # Unproject to 3D in camera frame
            K = intrinsics[:, cam_idx, :, :]  # (B, 3, 3)
            fx = K[:, 0, 0].unsqueeze(1)  # (B, 1)
            fy = K[:, 1, 1].unsqueeze(1)
            cx = K[:, 0, 2].unsqueeze(1)
            cy = K[:, 1, 2].unsqueeze(1)

            x_3d = (pixel_coords_batch[:, :, 0] - cx) * depth_values / fx
            y_3d = (pixel_coords_batch[:, :, 1] - cy) * depth_values / fy
            z_3d = depth_values

            patch_coords_cam = torch.stack(
                [x_3d, y_3d, z_3d], dim=2
            ).float()  # (B, num_patches, 3)

            # Transform to world frame
            T_world_from_cam = extrinsics[:, cam_idx, :, :]  # (B, 4, 4)
            patch_coords_world = self.transform_to_world(
                patch_coords_cam, T_world_from_cam
            )

            all_coords_world.append(patch_coords_world)

        # Concatenate all cameras: (B, N*num_patches, 3)
        patch_coords = torch.cat(all_coords_world, dim=1)

        return patch_coords

    def forward(
        self,
        image,
        depth,
        intrinsics,
        extrinsics,
        gripper_token=None,
        text=None,
        source=None,
    ):
        """
        Multi-camera forward pass.

        Args:
            image: (B, N, 3, H, W) RGB images for N cameras
            depth: (B, N, H, W) depth maps for N cameras
            intrinsics: (B, N, 3, 3) camera intrinsics
            extrinsics: (B, N, 4, 4) camera-to-world transforms
            gripper_token: (B, 10) [6DoF pose (3 pos + 6 rot6d) + gripper width]
            text: (B, ) Text captions
            source: (B, ) ["human" or "aloha"]

        Returns:
            outputs: (B, T, 13) GMM parameters for all cameras
            patch_coords: (B, T, 3) patch center 3D coordinates in WORLD frame
        """
        B, N, C, H, W = image.shape

        # Extract DINOv2 features for each camera
        all_patch_features = []
        for cam_idx in range(N):
            with torch.no_grad():
                cam_image = image[:, cam_idx, :, :, :]  # (B, 3, H, W)
                inputs = self.backbone_processor(images=cam_image, return_tensors="pt")
                inputs = {k: v.to(self.backbone.device) for k, v in inputs.items()}
                dino_outputs = self.backbone(**inputs)

            # Get patch features (skip CLS and register tokens)
            patch_features = dino_outputs.last_hidden_state[
                :, 5:
            ]  # (B, 196, dino_hidden_dim)
            all_patch_features.append(patch_features)

        # Concatenate features from all cameras: (B, N*196, dino_hidden_dim)
        patch_features = torch.cat(all_patch_features, dim=1)

        # Get 3D positional encoding for patches (in world frame)
        patch_coords = self.get_patch_centers(
            H, W, intrinsics, depth, extrinsics
        )  # (B, N*196, 3)
        pos_encoding = self.pos_encoder(patch_coords)  # (B, N*196, 128)

        # Combine patch features with positional encoding
        tokens = torch.cat(
            [patch_features, pos_encoding], dim=-1
        )  # (B, N*196, hidden_dim)

        # Apply image token dropout (training only)
        tokens, patch_coords = self.apply_image_token_dropout(tokens, patch_coords, N)

        # Number of tokens T <= N*196
        num_patch_tokens = tokens.shape[1]
        mask = torch.zeros(B, num_patch_tokens, dtype=torch.bool, device=tokens.device)

        # Add language tokens
        if self.use_text_embedding:
            text_tokens = self.text_tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )
            text_tokens = {
                k: v.to(self.text_encoder.device) for k, v in text_tokens.items()
            }
            text_embedding = self.text_encoder(**text_tokens).last_hidden_state

            lang_tokens = self.text_proj(text_embedding)  # (B, J, hidden_dim)
            tokens = torch.cat([tokens, lang_tokens], dim=1)  # (B, T+J, hidden_dim)
            mask = torch.cat([mask, text_tokens["attention_mask"] == 0], dim=1)

        # Add gripper token
        if self.use_gripper_token:
            grip_token = self.gripper_encoder(gripper_token).unsqueeze(
                1
            )  # (B, 1, hidden_dim)
            tokens = torch.cat([tokens, grip_token], dim=1)  # (B, T+J+1, hidden_dim)
            mask = torch.cat(
                [mask, torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)], dim=1
            )

        # Add source token
        if self.use_source_token:
            source_indices = torch.tensor(
                [self.source_to_idx[s] for s in source], device=tokens.device
            )
            source_token = self.source_embeddings(source_indices).unsqueeze(
                1
            )  # (B, 1, hidden_dim)
            tokens = torch.cat([tokens, source_token], dim=1)  # (B, T+J+2, hidden_dim)
            mask = torch.cat(
                [mask, torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)], dim=1
            )

        tokens = torch.cat([tokens, self.registers.expand(B, -1, -1)], dim=1)
        mask = torch.cat(
            [
                mask,
                torch.zeros(
                    B, self.num_registers, dtype=torch.bool, device=tokens.device
                ),
            ],
            dim=1,
        )

        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens, src_key_padding_mask=mask)

        # Take only the patch tokens (throw away language, gripper, source, register tokens)
        tokens = tokens[:, :num_patch_tokens]  # (B, T, hidden_dim)

        # Predict GMM parameters
        outputs = self.output_head(tokens)  # (B, T, 13)

        return outputs, patch_coords, tokens


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

        # Optimal Transport loss parameters
        self.use_ot_loss = cfg.model.use_ot_loss
        self.ot_alpha = cfg.model.ot_alpha
        self.ot_lambda = cfg.model.ot_lambda
        self.ot_epsilon = cfg.model.ot_epsilon
        self.ot_percentile = cfg.model.ot_percentile

        # Gripper noise augmentation parameters
        self.gripper_noise_prob = cfg.model.gripper_noise_prob
        self.gripper_noise_translation = cfg.model.gripper_noise_translation
        self.gripper_noise_rotation = cfg.model.gripper_noise_rotation
        self.gripper_noise_width = cfg.model.gripper_noise_width

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

    def apply_gripper_noise_to_token(self, gripper_token):
        """
        Apply noise to gripper token during training.

        Args:
            gripper_token: (B, 10) [pos (3) + rot6d (6) + width (1)]

        Returns:
            noisy_token: (B, 10) gripper token with noise applied
        """
        if not self.training or random.random() > self.gripper_noise_prob:
            return gripper_token

        B = gripper_token.shape[0]
        device = gripper_token.device

        # Parse gripper token
        pos = gripper_token[:, :3]  # (B, 3)
        rot6d = gripper_token[:, 3:9]  # (B, 6)
        width = gripper_token[:, 9:10]  # (B, 1)

        # 1. Add translation noise
        translation_noise = torch.empty_like(pos).uniform_(
            -self.gripper_noise_translation, self.gripper_noise_translation
        )
        noisy_pos = pos + translation_noise

        # 2. Add rotation noise
        # Convert rot6d to rotation matrix
        rot_matrix = rotation_6d_to_matrix(rot6d)  # (B, 3, 3)

        # Generate random euler angles (XYZ convention)
        euler_noise = torch.empty(B, 3, device=device).uniform_(
            -np.deg2rad(self.gripper_noise_rotation),
            np.deg2rad(self.gripper_noise_rotation),
        )  # (B, 3) in radians

        # Convert euler angles to rotation matrix
        R_noise = euler_angles_to_matrix(euler_noise, convention="XYZ")  # (B, 3, 3)

        # Apply noise: R_new = R_noise @ R_original
        noisy_rot_matrix = torch.bmm(R_noise, rot_matrix)  # (B, 3, 3)

        # Convert back to rot6d
        noisy_rot6d = matrix_to_rotation_6d(noisy_rot_matrix)  # (B, 6)

        # 3. Add gripper width noise
        width_noise = torch.empty_like(width).uniform_(
            -self.gripper_noise_width, self.gripper_noise_width
        )
        noisy_width = width + width_noise

        # Reconstruct token
        noisy_token = torch.cat([noisy_pos, noisy_rot6d, noisy_width], dim=1)
        return noisy_token

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

    def project_3d_to_2d(
        self, points_3d_world, intrinsics, extrinsics, img_shape=(224, 224)
    ):
        """
        Project 3D points in world frame to 2D pixel coordinates.

        Args:
            points_3d_world: (B, N, 3) or (B, N, M, 3) 3D points in WORLD frame
            intrinsics: (B, 3, 3) camera intrinsics
            extrinsics: (B, 4, 4) camera-to-world transformation (T_world_from_cam)
            img_shape: (H, W) image shape for clamping

        Returns:
            pixel_coords: (B, N, 2) or (B, N, M, 2) pixel coordinates [x, y]
        """
        H, W = img_shape
        original_shape = points_3d_world.shape

        # Reshape to (B, -1, 3) for batch processing
        if len(original_shape) == 4:
            B, N, M, _ = original_shape
            points_3d_world = points_3d_world.reshape(B, N * M, 3)
        else:
            B, N, _ = original_shape

        # Transform from world frame to camera frame
        # extrinsics is T_world_from_cam, we need T_cam_from_world = inv(T_world_from_cam)
        T_cam_from_world = torch.inverse(extrinsics)  # (B, 4, 4)

        ones = torch.ones(B, points_3d_world.shape[1], 1, device=points_3d_world.device)
        points_world_hom = torch.cat([points_3d_world, ones], dim=-1)  # (B, N*M, 4)

        # Apply transformation: (B, 4, 4) @ (B, N*M, 4) -> (B, N*M, 4) ->  (B, N*M, 3)
        points_3d_cam = torch.einsum(
            "bij,bnj->bni", T_cam_from_world, points_world_hom
        )[:, :, :3]

        fx = intrinsics[:, 0, 0].unsqueeze(1)  # (B, 1)
        fy = intrinsics[:, 1, 1].unsqueeze(1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)

        # Project: [x, y, z] -> [u, v]
        # Add epsilon to avoid division by zero
        z = points_3d_cam[:, :, 2].clamp(min=1e-6)
        u = (points_3d_cam[:, :, 0] * fx / z + cx).clamp(0, W - 1)  # (B, N*M)
        v = (points_3d_cam[:, :, 1] * fy / z + cy).clamp(0, H - 1)  # (B, N*M)

        pixel_coords = torch.stack([u, v], dim=2)  # (B, N*M, 2)

        # Reshape back to original shape
        if len(original_shape) == 4:
            pixel_coords = pixel_coords.reshape(B, N, M, 2)

        return pixel_coords

    def gripper_points_to_rotation(self, gripper_center, palm_point, finger_point):
        # Always use palm->gripper as primary axis (more stable)
        forward = gripper_center - palm_point
        x_axis = forward / torch.linalg.norm(forward, dim=1, keepdim=True)

        # Use finger relative to the forward direction for secondary axis
        finger_vec = gripper_center - finger_point

        # Project finger vector onto plane perpendicular to forward
        finger_projected = (
            finger_vec - torch.sum(finger_vec * x_axis, dim=1, keepdim=True) * x_axis
        )
        y_axis = finger_projected / torch.linalg.norm(
            finger_projected, dim=1, keepdim=True
        )

        # Z completes the frame
        z_axis = torch.cross(x_axis, y_axis)

        return torch.stack([x_axis, y_axis, z_axis], dim=-1)

    def get_gripper_token(self, gripper_points):
        """
        Extract gripper state as a token (6DoF pose + gripper width).
        """
        gripper_pos = (gripper_points[:, 0, :] + gripper_points[:, 1, :]) / 2
        gripper_width = torch.linalg.norm(
            gripper_points[:, 0, :] - gripper_points[:, 1, :], dim=1
        )[:, None]
        # eef pose, base, right finger
        gripper_rot = self.gripper_points_to_rotation(
            gripper_pos, gripper_points[:, 2, :], gripper_points[:, 0, :]
        )
        gripper_rot6d = matrix_to_rotation_6d(gripper_rot)

        gripper_token = torch.cat([gripper_pos, gripper_rot6d, gripper_width], dim=-1)
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
        Similar to articubot.py but adapted for T <= 196*N fixed components.
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
        )  # Shape: (B, T, 12)
        exponent = -0.5 * torch.sum((diff**2) / variance, dim=2)  # Shape: (B, T)
        log_gaussians = exponent

        # Compute log mixing coefficients
        log_mixing_coeffs = torch.log_softmax(weights, dim=1)  # (B, T)
        log_mixing_coeffs = torch.clamp(log_mixing_coeffs, min=-10)

        masked_sum = log_gaussians + log_mixing_coeffs  # [B, T]
        masked_sum = masked_sum.masked_fill(~valid_mask, -1e9)

        max_log = torch.max(masked_sum, dim=1, keepdim=True).values  # (B, 1)
        log_probs = max_log.squeeze(1) + torch.logsumexp(
            masked_sum - max_log, dim=1
        )  # B,

        nll_loss = -torch.mean(log_probs)
        return nll_loss

    def forward(self, batch):
        """Forward pass with GMM loss"""
        init, gt = self.extract_gt_4_points(batch)

        # Get gripper token (6DoF pose + gripper width)
        gripper_token = self.get_gripper_token(init)

        # Apply gripper noise augmentation (training only)
        gripper_token = self.apply_gripper_noise_to_token(gripper_token)

        # Combine primary + auxiliary cameras
        # Primary: batch["rgbs"][:, 0] is (B, H, W, 3), batch["depths"][:, 0] is (B, H, W)
        # Auxiliary: batch["aux_rgbs"] is (B, N_aux, 2, H, W, 3), batch["aux_depths"] is (B, N_aux, 2, H, W)

        primary_rgb = batch["rgbs"][:, 0]  # (B, H, W, 3)
        primary_depth = batch["depths"][:, 0]  # (B, H, W)
        aux_rgbs = batch["aux_rgbs"][:, :, 0, :, :, :]  # (B, N_aux, H, W, 3)
        aux_depths = batch["aux_depths"][:, :, 0, :, :]  # (B, N_aux, H, W)

        # Stack along camera dimension
        all_rgbs = torch.cat(
            [primary_rgb.unsqueeze(1), aux_rgbs], dim=1
        )  # (B, N, H, W, 3)
        all_depths = torch.cat(
            [primary_depth.unsqueeze(1), aux_depths], dim=1
        )  # (B, N, H, W)

        # Clip depths
        all_depths[all_depths > self.max_depth] = 0

        # Permute RGB to (B, N, 3, H, W)
        rgb = all_rgbs.permute(0, 1, 4, 2, 3)
        depth = all_depths

        # Combine intrinsics and extrinsics
        all_intrinsics = torch.cat(
            [
                batch["intrinsics"].unsqueeze(1),  # (B, 1, 3, 3)
                batch["aux_intrinsics"],  # (B, N_aux, 3, 3)
            ],
            dim=1,
        )  # (B, N, 3, 3)

        all_extrinsics = torch.cat(
            [
                batch["extrinsics"].unsqueeze(1),  # (B, 1, 4, 4)
                batch["aux_extrinsics"],  # (B, N_aux, 4, 4)
            ],
            dim=1,
        )  # (B, N, 4, 4)

        # Forward through network
        outputs, patch_coords, tokens = self.network(
            rgb,
            depth,
            all_intrinsics,
            all_extrinsics,
            gripper_token=gripper_token,
            text=batch["caption"],
            source=batch["data_source"],
        )

        # outputs: (B, T, 13) - last dim is [12 coords + 1 weight]
        B, num_components, _ = outputs.shape

        # Parse outputs
        pred_displacement = outputs[:, :, :-1].reshape(
            B, num_components, 4, 3
        )  # (B, T, 4, 3)
        weights = outputs[:, :, -1]  # (B, T)

        # Predictions are residuals from patch centers, add them to get absolute positions
        # Expand patch_coords to match pred shape
        patch_coords_expanded = patch_coords[:, :, None, :]  # (B, T, 1, 3)
        pred = patch_coords_expanded + pred_displacement  # Residual to absolute

        # GT displacement relative to patch centers
        gt_displacement = gt[:, None, :, :] - patch_coords_expanded  # (B, T, 4, 3)

        # All components are valid (T fixed components)
        valid_mask = torch.ones(
            B, num_components, device=outputs.device, dtype=torch.bool
        )

        loss_dict = {}
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
            loss_dict["gmm_loss"] = loss
        else:
            # Simple MSE loss (if not using GMM)
            # Get weighted prediction
            weights_norm = F.softmax(weights, dim=1)
            pred_points = (weights_norm[:, :, None, None] * pred_displacement).sum(
                dim=1
            )
            loss = F.mse_loss(pred_points, gt)
            loss_dict["mse_loss"] = loss

        if self.use_ot_loss:
            ot_loss = self.ot_loss(
                tokens,
                embodiment=batch["data_source"],
                caption=batch["caption"],
                goal_vec=(gt - init)[:, 0, :],
            )
            loss_dict["ot_loss"] = ot_loss
            loss = loss + (self.ot_alpha * ot_loss)

        return None, loss, loss_dict

    def ot_loss(self, tokens, embodiment, caption, goal_vec):
        """
        Optimal Transport-based loss for domain adaptation based on EgoBridge.
        Aligns distributions of the latent representations of human and robot data.

        Similar latents are expected when we have similar tasks (pick-place, fold)
        and the goal vectors (goal_pos - current_pos) are similar.

        For this to work, batch size needs to be reasonably large and contain similar amounts
        and types of human and robot data, careful!
        """
        # Only compute OT loss if minibatch contains aloha and human data.
        if set(embodiment) != {"aloha", "human"}:
            return 0.0

        human_mask = [i == "human" for i in embodiment]
        robot_mask = [i == "aloha" for i in embodiment]
        n_h, n_r = sum(human_mask), sum(robot_mask)

        # Group the captions by the first word
        # [Fold the onesie, Fold the shirt] -> Fold
        # Somewhat hacky, should probably do semantic similarity?
        task = np.array([c.split(" ")[0] for c in caption])
        task_h, task_r = task[human_mask], task[robot_mask]
        task_match = torch.tensor(
            task_h[:, None] == task_r[None, :], device=tokens.device
        )

        # Similarity matrix of residual vectors (goal - current)
        # Considered a match if the distance is less than the percentile threshold
        res_h, res_r = goal_vec[human_mask], goal_vec[robot_mask]
        R = torch.cdist(res_h, res_r) ** 2
        best_match = (R < R.quantile(self.ot_percentile)) & task_match

        z = tokens.mean(dim=1)  # (B, T, D) -> (B, D)
        z_h, z_r = z[human_mask], z[robot_mask]

        # Compute cost matrix of latents
        C = torch.cdist(z_h, z_r) ** 2
        C = C / (C.max() + 1e-8)
        C[best_match] *= self.ot_lambda  # Discount latents which should align
        C[~task_match] = 1.0  # Penalize cross-task

        # Optimal Transport loss
        a = torch.ones(n_h, device=tokens.device) / n_h
        b = torch.ones(n_r, device=tokens.device) / n_r
        T = ot.sinkhorn(a, b, C, reg=self.ot_epsilon)
        loss = (T * C).sum()

        return loss

    def training_step(self, batch, batch_idx):
        """Training step with 3D GMM prediction"""
        assert (
            batch["augment_t"].mean().item() == 0.0
        ), "Disable pcd augmentations when training image model!"

        self.train()
        batch_size = batch[self.label_key].points_padded().shape[0]

        _, loss, loss_dict = self(batch)
        train_metrics = {"loss": loss}
        train_metrics.update(loss_dict)

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

                pred_dict, weighted_displacement, _ = all_pred_dict[0]
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

            # Calculate pixel metrics
            intrinsics = batch["intrinsics"]
            extrinsics = batch["extrinsics"]
            H, W = batch["rgbs"].shape[2:4]

            # Project GT to 2D (take first point only)
            gt_2d = (
                self.project_3d_to_2d(gt[:, :1, :], intrinsics, extrinsics, (H, W))
                .squeeze(1)
                .long()
            )  # (B, 2)

            # Project all predictions to 2D (take first point only)
            all_pred_3d = (
                init[:, None, :, :] + pred_dict[self.prediction_type]["all_pred"]
            )  # (B, N, 4, 3)
            all_pred_2d = (
                self.project_3d_to_2d(
                    all_pred_3d[:, :, :1, :], intrinsics, extrinsics, (H, W)
                )
                .squeeze(2)
                .long()
            )  # (B, N, 2)

            pred_dict = calc_pix_metrics(pred_dict, gt_2d, all_pred_2d, (H, W))
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
        init, gt = self.extract_gt_4_points(batch)
        gripper_token = self.get_gripper_token(init)

        # Combine primary + auxiliary cameras (same as forward)
        primary_rgb = batch["rgbs"][:, 0]  # (B, H, W, 3)
        primary_depth = batch["depths"][:, 0]  # (B, H, W)
        aux_rgbs = batch["aux_rgbs"][:, :, 0, :, :, :]  # (B, N_aux, H, W, 3)
        aux_depths = batch["aux_depths"][:, :, 0, :, :]  # (B, N_aux, H, W)

        all_rgbs = torch.cat(
            [primary_rgb.unsqueeze(1), aux_rgbs], dim=1
        )  # (B, N, H, W, 3)
        all_depths = torch.cat(
            [primary_depth.unsqueeze(1), aux_depths], dim=1
        )  # (B, N, H, W)

        rgb = all_rgbs.permute(0, 1, 4, 2, 3)  # (B, N, 3, H, W)
        depth = all_depths

        all_intrinsics = torch.cat(
            [batch["intrinsics"].unsqueeze(1), batch["aux_intrinsics"]], dim=1
        )

        all_extrinsics = torch.cat(
            [batch["extrinsics"].unsqueeze(1), batch["aux_extrinsics"]], dim=1
        )

        # Forward
        outputs, patch_coords, tokens = self.network(
            rgb,
            depth,
            all_intrinsics,
            all_extrinsics,
            gripper_token=gripper_token,
            text=batch["caption"],
            source=batch["data_source"],
        )

        z = tokens.mean(dim=1)  # (B, T, D) -> (B, D)

        if self.is_gmm:
            pred = self.sample_from_gmm(outputs, patch_coords)
        else:
            pred = self.get_weighted_prediction(outputs, patch_coords)

        pred_displacement = pred - init
        return {self.prediction_type: {"pred": pred_displacement}}, outputs, z

    def sample_from_gmm(self, outputs, patch_coords):
        """
        Sample from GMM by selecting a component and using its mean.
        Args:
            outputs: (B, T, 13) GMM parameters
            patch_coords: (B, T, 3) patch center coordinates
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

        # Transform from world frame to primary camera frame for projection
        # Primary camera extrinsics: T_world_from_cam, we need T_cam_from_world
        primary_extrinsics = batch["extrinsics"][viz_idx].cpu().numpy()  # (4, 4)
        T_cam_from_world = np.linalg.inv(primary_extrinsics)

        # Transform points to primary camera frame
        # Transform initial pcd (for init_rgb_proj)
        pcd_cam = np.hstack((pcd, np.ones((pcd.shape[0], 1))))
        pcd_cam = (T_cam_from_world @ pcd_cam.T).T[:, :3]

        pcd_endframe = np.hstack((pcd_endframe, np.ones((pcd_endframe.shape[0], 1))))
        pcd_endframe = (T_cam_from_world @ pcd_endframe.T).T[:, :3]

        all_pred_pcd_tmp = []
        for i in range(N):
            tmp_pcd = np.hstack((all_pred_pcd[i], np.ones((all_pred_pcd.shape[1], 1))))
            tmp_pcd = (T_cam_from_world @ tmp_pcd.T).T[:, :3]
            all_pred_pcd_tmp.append(tmp_pcd)
        all_pred_pcd = np.stack(all_pred_pcd_tmp)

        gt_pcd = np.hstack((gt_pcd, np.ones((gt_pcd.shape[0], 1))))
        gt_pcd = (T_cam_from_world @ gt_pcd.T).T[:, :3]

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
        init_rgb_proj = project_pcd_on_image(pcd_cam, padding_mask, rgb_init, K, GREEN)
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

        # Log loss_dict components
        if any("gmm_loss" in x for x in self.train_outputs):
            log_dictionary["train/gmm_loss"] = mean_metric("gmm_loss")
        if any("mse_loss" in x for x in self.train_outputs):
            log_dictionary["train/mse_loss"] = mean_metric("mse_loss")
        if any("ot_loss" in x for x in self.train_outputs):
            log_dictionary["train/ot_loss"] = mean_metric("ot_loss")

        if any("rmse" in x for x in self.train_outputs):
            log_dictionary["train/rmse"] = mean_metric("rmse")
            log_dictionary["train/wta_rmse"] = mean_metric("wta_rmse")
            log_dictionary["train/chamfer_dist"] = mean_metric("chamfer_dist")
            log_dictionary["train/wta_chamfer_dist"] = mean_metric("wta_chamfer_dist")
            log_dictionary["train/sample_std"] = mean_metric("sample_std")
            log_dictionary["train/pix_dist"] = mean_metric("pix_dist")
            log_dictionary["train/wta_pix_dist"] = mean_metric("wta_pix_dist")
            log_dictionary["train/normalized_pix_dist"] = mean_metric(
                "normalized_pix_dist"
            )
            log_dictionary["train/wta_normalized_pix_dist"] = mean_metric(
                "wta_normalized_pix_dist"
            )

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
            pred_dict, weighted_displacement, _ = all_pred_dict[0]

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

        # Calculate pixel metrics
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        H, W = batch["rgbs"].shape[2:4]

        # Project GT to 2D (take first point only)
        gt_2d = (
            self.project_3d_to_2d(gt[:, :1, :], intrinsics, extrinsics, (H, W))
            .squeeze(1)
            .long()
        )  # (B, 2)

        # Project all predictions to 2D (take first point only)
        all_pred_3d = (
            init[:, None, :, :] + pred_dict[self.prediction_type]["all_pred"]
        )  # (B, N, 4, 3)
        all_pred_2d = (
            self.project_3d_to_2d(
                all_pred_3d[:, :, :1, :], intrinsics, extrinsics, (H, W)
            )
            .squeeze(2)
            .long()
        )  # (B, N, 2)

        pred_dict = calc_pix_metrics(pred_dict, gt_2d, all_pred_2d, (H, W))
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
            "pix_dist": [],
            "wta_pix_dist": [],
            "normalized_pix_dist": [],
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

        pred_dict, weighted_displacement, z = all_pred_dict[0]
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

        # Calculate pixel metrics
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        H, W = batch["rgbs"].shape[2:4]

        # Project GT to 2D (take first point only)
        gt_2d = (
            self.project_3d_to_2d(gt[:, :1, :], intrinsics, extrinsics, (H, W))
            .squeeze(1)
            .long()
        )  # (B, 2)

        # Project all predictions to 2D (take first point only)
        all_pred_3d = (
            init[:, None, :, :] + pred_dict[self.prediction_type]["all_pred"]
        )  # (B, N, 4, 3)
        all_pred_2d = (
            self.project_3d_to_2d(
                all_pred_3d[:, :, :1, :], intrinsics, extrinsics, (H, W)
            )
            .squeeze(2)
            .long()
        )  # (B, N, 2)

        pred_dict = calc_pix_metrics(pred_dict, gt_2d, all_pred_2d, (H, W))
        self.predict_outputs[eval_tag].append(pred_dict)
        self.predict_weighted_displacements[eval_tag].append(
            weighted_displacement.cpu()
        )

        # Get pred_coord for visualization (first sample, first 3 points)
        pred_3d = (
            init + pred_dict[self.prediction_type]["pred"]
        )  # (B, 4, 3) absolute positions
        pred_3d_first3 = pred_3d[:, :3, :]  # (B, 3, 3) first 3 points
        pred_coord_viz = self.project_3d_to_2d(
            pred_3d_first3, intrinsics, extrinsics, (H, W)
        ).long()  # (B, 3, 2)

        return {
            "pred_coord": pred_coord_viz,
            "rmse": pred_dict["rmse"],
            "chamfer_dist": pred_dict["chamfer_dist"],
            "wta_rmse": pred_dict["wta_rmse"],
            "wta_chamfer_dist": pred_dict["wta_chamfer_dist"],
            "pix_dist": pred_dict["pix_dist"],
            "wta_pix_dist": pred_dict["wta_pix_dist"],
            "vid_name": batch["vid_name"],
            "caption": batch["caption"],
            "z": z,  # (B, D) mean-pooled token representation
        }

    def on_predict_epoch_end(self):
        """Stub - implement if needed"""
        pass
