import random
import types
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn.functional as F
import wandb
from diffusers import get_cosine_schedule_with_warmup
from pytorch3d.transforms import (
    euler_angles_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from torch import nn, optim

from lfd3d.utils.viz_utils import (
    get_img_and_track_pcd,
    interpolate_colors,
    invert_augmentation_and_normalization,
    project_pcd_on_image,
)


def calc_traj_metrics(pred_dict, all_pred, gt_trajectory):
    """
    Calculate trajectory metrics and update pred_dict with the keys.

    pred_dict: Dictionary with keys to be updated
    all_pred: Predicted trajectories multiple samples
    gt_trajectory: GT trajectory
    """
    all_rmse = []
    for i in range(all_pred.shape[1]):
        pred = all_pred[:, i]
        all_rmse.append(((pred - gt_trajectory) ** 2).mean((1, 2)) ** 0.5)
    pred_dict["rmse"] = all_rmse[0]
    pred_dict["wta_rmse"] = torch.stack(all_rmse).min(0)[0]
    return pred_dict


def calc_traj_pix_metrics(pred_dict, gt_traj, all_pred_traj, img_shape):
    """
    Calculate pixel distance metrics and update pred_dict with the keys.

    pred_dict: Dictionary with keys to be updated
    gt_idx: GT pixel trajectory (B, 10, 2) [x, y]
    all_pred_idx: Predicted goal pixel trajectories of multiple samples (B, N, 10, 2)
    img_shape: (H, W) image shape for normalization
    """
    H, W = img_shape

    # Calculate L1 distances for all samples
    # gt_idx: (B, 2), all_pred_idx: (B, N, 2)
    gt_expanded = gt_traj.unsqueeze(1)  # (B, 1, 2)

    # L1 distance in pixel space
    pix_distances = torch.sum(
        torch.abs(all_pred_traj.float() - gt_expanded.float()), dim=(2, 3)
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


def monkey_patch_mimicplay(network):
    """
    Monkey-patch in alternate functionality to train Mimicplay baseline.
    """

    def mimicplay_forward(
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
        Modified version of forward() for Dino3DGP
        """
        B, N, C, H, W = image.shape

        # Extract DINOv3 features for each camera
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

        # NEW ###
        # Add CLS token to hold latent plan
        tokens = torch.cat([tokens, self.cls_token.expand(B, -1, -1)], dim=1)
        mask = torch.cat(
            [
                mask,
                torch.zeros(B, 1, dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )

        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens, src_key_padding_mask=mask)

        # NEW ###
        # Take only the CLS token
        latent_plan = tokens[:, -1, :]  # (B, hidden_dim)
        # Predict GMM parameters
        outputs = self.gmm_decoder(latent_plan)

        means = outputs[:, :150].reshape(-1, 5, 30)  # (B, 5, 30)
        raw_scales = outputs[:, 150:300].reshape(-1, 5, 30)  # (B, 5, 30)
        logits = outputs[:, 300:305].reshape(-1, 5)  # (B, 5)

        scales = F.softplus(raw_scales) + 0.0001
        component_dist = D.Independent(D.Normal(means, scales), 1)
        mixture_dist = D.Categorical(logits=logits)
        gmm_dist = D.MixtureSameFamily(mixture_dist, component_dist)

        return latent_plan, gmm_dist

    # Add extra params for latent plan and GMM decoder
    network.cls_token = nn.Parameter(torch.randn(1, 1, network.hidden_dim) * 0.02)
    network.num_modes = 5
    network.pred_timesteps = 10
    # Predict 10-step trajectory of 4th gripper point (xyz only)
    # Output: means (5, 30) + scales (5, 30) + logits (5) = 305 total
    network.gmm_decoder = nn.Sequential(
        nn.Linear(network.hidden_dim, 400),
        nn.Softplus(),
        nn.Linear(400, 400),
        nn.Softplus(),
        nn.Linear(
            400,
            network.num_modes * (network.pred_timesteps * 3 * 2) + network.num_modes,
        ),
        # 5 modes × (10 timesteps × 3 coords × 2 [mean+scale]) + 5 logits = 305
    )

    # Add in a separate forward pass for MimicPlay
    network.mimicplay_forward = types.MethodType(mimicplay_forward, network)

    return network


class MimicplayModule(pl.LightningModule):
    """
    MimicPlay baseline building on top of Dino3DGP
    Monkey-patches a different forward pass into the model
    and uses a different loss function.
    """

    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.network = monkey_patch_mimicplay(network)
        self.model_cfg = cfg.model
        self.prediction_type = self.model_cfg.type
        self.mode = cfg.mode  # train or eval
        self.val_outputs: defaultdict[str, List[Dict]] = defaultdict(list)
        self.train_outputs: List[Dict] = []
        self.predict_outputs: defaultdict[str, List[Dict]] = defaultdict(list)

        # Gripper noise augmentation parameters
        self.gripper_noise_prob = cfg.model.gripper_noise_prob
        self.gripper_noise_translation = cfg.model.gripper_noise_translation
        self.gripper_noise_rotation = cfg.model.gripper_noise_rotation
        self.gripper_noise_width = cfg.model.gripper_noise_width

        # KL divergence loss parameters (MimicPlay-style)
        self.kl_lambda = cfg.model.kl_lambda
        self.min_std = cfg.model.gmm_min_std

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

    def extract_gt_trajectory(self, batch):
        """Extract ground truth goal points (4 gripper points)"""
        gripper_trajectory = batch["gripper_trajectory"]
        # Assumes 0/1 are tips to be averaged
        gt_trajectory = (
            gripper_trajectory[:, :, 0, :] + gripper_trajectory[:, :, 1, :]
        ) / 2
        return gt_trajectory

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

    def combine_camera_data(self, batch):
        """
        Combine primary and auxiliary camera data.

        Returns:
            rgb: (B, N, 3, H, W)
            depth: (B, N, H, W)
            all_intrinsics: (B, N, 3, 3)
            all_extrinsics: (B, N, 4, 4)
        """
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

        return rgb, depth, all_intrinsics, all_extrinsics

    def collect_and_stack_predictions(self, batch, n_samples):
        """
        Collect multiple predictions and stack them.

        Returns:
            pred_dict: Dictionary with stacked predictions in "all_pred" key
            pred_gmm: GMM distribution from first prediction
        """
        all_pred_dict = []
        for i in range(n_samples):
            all_pred_dict.append(self.predict(batch))

        pred_dict, pred_gmm = all_pred_dict[0]
        pred_dict[self.prediction_type]["all_pred"] = [
            i[0][self.prediction_type]["pred"] for i in all_pred_dict
        ]
        pred_dict[self.prediction_type]["all_pred"] = torch.stack(
            pred_dict[self.prediction_type]["all_pred"]
        ).permute(1, 0, 2, 3)

        return pred_dict, pred_gmm

    def calculate_pixel_metrics(self, pred_dict, batch, gt_trajectory):
        """
        Calculate pixel-based metrics by projecting 3D predictions to 2D.

        Args:
            pred_dict: Prediction dictionary to update
            batch: Batch data
            gt_trajectory: Ground truth trajectory (B, 10, 3)

        Returns:
            Updated pred_dict with pixel metrics
        """
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        H, W = batch["rgbs"].shape[2:4]

        # Project GT to 2D
        gt_2d = self.project_3d_to_2d(
            gt_trajectory, intrinsics, extrinsics, (H, W)
        ).long()  # (B, 10, 2)

        # Project all predictions to 2D
        all_pred_3d = pred_dict[self.prediction_type]["all_pred"]  # (B, N, 10, 3)
        all_pred_2d = self.project_3d_to_2d(
            all_pred_3d, intrinsics, extrinsics, (H, W)
        ).long()  # (B, N, 10, 2)

        pred_dict = calc_traj_pix_metrics(pred_dict, gt_2d, all_pred_2d, (H, W))
        return pred_dict

    def transform_points_homogeneous(self, points, transform_matrix):
        """
        Transform 3D points using a 4x4 homogeneous transformation matrix.

        Args:
            points: (N, 3) or (M, N, 3) array of 3D points
            transform_matrix: (4, 4) transformation matrix

        Returns:
            Transformed points with same shape as input
        """
        original_shape = points.shape
        if points.ndim == 2:
            points = points[np.newaxis, ...]

        # Add homogeneous coordinate
        points_hom = np.hstack(
            (points.reshape(-1, 3), np.ones((points.reshape(-1, 3).shape[0], 1)))
        )
        # Transform
        points_transformed = (transform_matrix @ points_hom.T).T[:, :3]

        # Reshape back
        if len(original_shape) == 2:
            return points_transformed
        else:
            return points_transformed.reshape(original_shape)

    def kl_loss(self, latent_plan, embodiment):
        """
        KL divergence loss between robot and human latent plan distributions.
        Domain adaptation loss that encourages human and robot latent plans to align.
        Args:
            latent_plan: (B, hidden_dim) latent representations from CLS token
            embodiment: list of strings ["human", "aloha", ...] indicating data source
        """
        # Separate by embodiment
        human_mask = torch.tensor(
            [e == "human" for e in embodiment], device=latent_plan.device
        )
        robot_mask = torch.tensor(
            [e == "aloha" for e in embodiment], device=latent_plan.device
        )

        # Need both human and robot samples in the batch
        if not (human_mask.any() and robot_mask.any()):
            return torch.tensor(0.0, device=latent_plan.device)

        human_latents = latent_plan[human_mask]  # (N_h, D)
        robot_latents = latent_plan[robot_mask]  # (N_r, D)

        # Compute distribution statistics for each embodiment
        mu_h = human_latents.mean(dim=0)  # (D,)
        mu_r = robot_latents.mean(dim=0)  # (D,)

        sigma_h = human_latents.std(dim=0) + 1e-6
        sigma_r = robot_latents.std(dim=0) + 1e-6

        # DKL(Qr || Qh)
        kl = (
            0.5
            * (
                2 * torch.log(sigma_h / sigma_r)
                + (sigma_r**2 + (mu_r - mu_h) ** 2) / (sigma_h**2)
                - 1.0
            ).sum()
        )

        return kl

    def forward(self, batch):
        """Forward pass with GMM loss"""
        init, gt = self.extract_gt_4_points(batch)

        # Get gripper token (6DoF pose + gripper width)
        gripper_token = self.get_gripper_token(init)

        # Apply gripper noise augmentation (training only)
        gripper_token = self.apply_gripper_noise_to_token(gripper_token)

        # Combine primary + auxiliary cameras
        rgb, depth, all_intrinsics, all_extrinsics = self.combine_camera_data(batch)

        # Forward through network
        latent_plan, gmm_dist = self.network.mimicplay_forward(
            rgb,
            depth,
            all_intrinsics,
            all_extrinsics,
            gripper_token=gripper_token,
            text=batch["caption"],
            source=batch["data_source"],
        )

        gt_trajectory = self.extract_gt_trajectory(batch).reshape(-1, 30)
        gmm_loss = -gmm_dist.log_prob(gt_trajectory).mean()
        kl_div = self.kl_loss(latent_plan, batch["data_source"])

        loss = gmm_loss + self.kl_lambda * kl_div
        loss_dict = {
            "gmm_loss": gmm_loss,
            "kl_div": kl_div,
        }

        return None, loss, loss_dict

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
                pred_dict, pred_gmm = self.collect_and_stack_predictions(
                    batch, n_samples_wta
                )
            self.train()

            gt_trajectory = self.extract_gt_trajectory(batch)
            pred_dict = calc_traj_metrics(
                pred_dict,
                pred_dict[self.prediction_type]["all_pred"],
                gt_trajectory,
            )

            pred_dict = self.calculate_pixel_metrics(pred_dict, batch, gt_trajectory)
            train_metrics.update(pred_dict)

            if self.trainer.is_global_zero:
                self.log_viz_to_wandb(batch, pred_dict, "train")

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

        # Combine primary + auxiliary cameras
        rgb, depth, all_intrinsics, all_extrinsics = self.combine_camera_data(batch)

        # Forward
        latent_plan, gmm_dist = self.network.mimicplay_forward(
            rgb,
            depth,
            all_intrinsics,
            all_extrinsics,
            gripper_token=gripper_token,
            text=batch["caption"],
            source=batch["data_source"],
        )

        pred_traj = gmm_dist.sample().reshape(-1, self.network.pred_timesteps, 3)
        return {self.prediction_type: {"pred": pred_traj}}, gmm_dist

    def log_viz_to_wandb(self, batch, pred_dict, tag):
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

        goal_text = batch["caption"][viz_idx]
        vid_name = batch["vid_name"][viz_idx]
        rmse = pred_dict["rmse"][viz_idx]

        gt_trajectory = self.extract_gt_trajectory(batch)

        pcd, gt = self.extract_gt_4_points(batch)
        pcd, gt = pcd.cpu().numpy()[viz_idx], gt.cpu().numpy()[viz_idx]
        all_pred_pcd = all_pred
        gt_pcd = self.extract_gt_trajectory(batch)[viz_idx].cpu().numpy()
        padding_mask = torch.ones(gt_pcd.shape[0]).bool().numpy()

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
        pcd_endframe = self.transform_points_homogeneous(pcd, end2start)
        all_pred_pcd = np.stack(
            [
                self.transform_points_homogeneous(all_pred_pcd[i], end2start)
                for i in range(N)
            ]
        )
        gt_pcd = self.transform_points_homogeneous(gt_pcd, end2start)

        # Transform from world frame to primary camera frame for projection
        # Primary camera extrinsics: T_world_from_cam, we need T_cam_from_world
        primary_extrinsics = batch["extrinsics"][viz_idx].cpu().numpy()  # (4, 4)
        T_cam_from_world = np.linalg.inv(primary_extrinsics)

        # Transform points to primary camera frame
        pcd_endframe = self.transform_points_homogeneous(pcd_endframe, T_cam_from_world)
        all_pred_pcd = np.stack(
            [
                self.transform_points_homogeneous(all_pred_pcd[i], T_cam_from_world)
                for i in range(N)
            ]
        )
        gt_pcd = self.transform_points_homogeneous(gt_pcd, T_cam_from_world)

        K = batch["intrinsics"][viz_idx].cpu().numpy()

        rgb_init, rgb_end = (
            batch["rgbs"][viz_idx, 0].cpu().numpy(),
            batch["rgbs"][viz_idx, 1].cpu().numpy(),
        )
        depth_init, depth_end = (
            batch["depths"][viz_idx, 0].cpu().numpy(),
            batch["depths"][viz_idx, 1].cpu().numpy(),
        )

        # Project tracks to image with color interpolation
        YELLOW = (255, 255, 0)
        GREEN = (0, 255, 0)

        # GT trajectory: RED to YELLOW gradient
        RED2YELLOW = interpolate_colors(RED, YELLOW, gt_pcd.shape[0])
        end_rgb_proj = project_pcd_on_image(
            gt_pcd, padding_mask, rgb_end, K, RED2YELLOW, radius=3
        )

        # Predicted trajectory: BLUE to GREEN gradient
        BLUE2GREEN = interpolate_colors(BLUE, GREEN, all_pred_pcd[-1].shape[0])
        pred_rgb_proj = project_pcd_on_image(
            all_pred_pcd[-1], padding_mask, rgb_end, K, BLUE2GREEN, radius=3
        )
        rgb_proj_viz = cv2.hconcat([rgb_init, end_rgb_proj, pred_rgb_proj])

        wandb_proj_img = wandb.Image(
            rgb_proj_viz,
            caption=f"Left: Initial Frame (GT Track)\n; Middle: Final Frame (GT Track)\n\
            ; Right: Final Frame (Pred Track)\n; Goal Description : {goal_text};\n\
            rmse={rmse};\nvideo path = {vid_name}; ",
        )

        # Create BLUE to GREEN gradients for all predictions
        # Each prediction gets a gradient from a shade of blue to corresponding shade of green
        BLUES2GREENS = []
        for i in range(N):
            start_blue = (
                (int(200 * (1 - i / (N - 1))), int(220 * (1 - i / (N - 1))), 255)
                if N > 1
                else (200, 220, 255)
            )
            end_green = (
                (int(200 * (1 - i / (N - 1))), 255, int(220 * (1 - i / (N - 1))))
                if N > 1
                else (200, 255, 220)
            )
            gradient = interpolate_colors(
                start_blue, end_green, all_pred_pcd[i].shape[0]
            )
            BLUES2GREENS.append(gradient)

        # Visualize point cloud
        viz_pcd, _ = get_img_and_track_pcd(
            rgb_end,
            depth_end,
            K,
            padding_mask,
            gt_pcd,  # repeating twice
            gt_pcd,
            all_pred_pcd,
            GREEN,
            RED2YELLOW,
            BLUES2GREENS,
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
        if any("kl_div" in x for x in self.train_outputs):
            log_dictionary["train/kl_div"] = mean_metric("kl_div")
        if any("mse_loss" in x for x in self.train_outputs):
            log_dictionary["train/mse_loss"] = mean_metric("mse_loss")
        if any("ot_loss" in x for x in self.train_outputs):
            log_dictionary["train/ot_loss"] = mean_metric("ot_loss")

        if any("rmse" in x for x in self.train_outputs):
            log_dictionary["train/rmse"] = mean_metric("rmse")
            log_dictionary["train/wta_rmse"] = mean_metric("wta_rmse")
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
            pred_dict, pred_gmm = self.collect_and_stack_predictions(
                batch, n_samples_wta
            )

        gt_trajectory = self.extract_gt_trajectory(batch)
        pred_dict = calc_traj_metrics(
            pred_dict,
            pred_dict[self.prediction_type]["all_pred"],
            gt_trajectory,
        )

        pred_dict = self.calculate_pixel_metrics(pred_dict, batch, gt_trajectory)
        self.val_outputs[val_tag].append(pred_dict)

        if (
            batch_idx == self.random_val_viz_idx[val_tag]
            and self.trainer.is_global_zero
        ):
            self.log_viz_to_wandb(batch, pred_dict, f"val_{val_tag}")
        return pred_dict

    def on_validation_epoch_end(self):
        log_dict = {}
        all_metrics = {
            "rmse": [],
            "wta_rmse": [],
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

        pred_dict, pred_gmm = self.collect_and_stack_predictions(batch, n_samples_wta)

        gt_trajectory = self.extract_gt_trajectory(batch)
        pred_dict = calc_traj_metrics(
            pred_dict,
            pred_dict[self.prediction_type]["all_pred"],
            gt_trajectory,
        )

        pred_dict = self.calculate_pixel_metrics(pred_dict, batch, gt_trajectory)
        self.predict_outputs[eval_tag].append(pred_dict)

        # Get pred_coord for visualization (take first point of trajectory)
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        H, W = batch["rgbs"].shape[2:4]

        # First sample prediction, first timestep
        pred_first_timestep = pred_dict[self.prediction_type]["pred"][
            :, :1, :
        ]  # (B, 1, 3)
        pred_coord_viz = (
            self.project_3d_to_2d(pred_first_timestep, intrinsics, extrinsics, (H, W))
            .squeeze(1)
            .long()
        )  # (B, 2)

        return {
            "pred_coord": pred_coord_viz,
            "rmse": pred_dict["rmse"],
            "wta_rmse": pred_dict["wta_rmse"],
            "pix_dist": pred_dict["pix_dist"],
            "wta_pix_dist": pred_dict["wta_pix_dist"],
            "vid_name": batch["vid_name"],
            "caption": batch["caption"],
        }

    def on_predict_epoch_end(self):
        """Stub - implement if needed"""
        pass
