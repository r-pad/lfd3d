"""
ViT3DGP: Vision Transformer + 3D Positional Encoding for Goal Prediction

This module is similar to DINO_3DGP but uses a trainable ViT (from scratch)
instead of a frozen DINO backbone. The ViT is sized to be comparable to ResNet-18.

For maximum code reuse, we import FourierPositionalEncoding and reuse
Dino3DGPGoalRegressionModule as the training module.
"""

import random

import torch
from einops import rearrange
from torch import nn
from transformers import AutoImageProcessor, T5EncoderModel, T5Tokenizer

from lfd3d.models.dino_3dgp import FourierPositionalEncoding


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, "b c h w -> b (h w) c")  # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone trained from scratch.

    Sized to be comparable to ResNet-18 (~11M params):
    - embed_dim = 384 (matches DINO ViT-B output for compatibility)
    - depth = 6
    - num_heads = 6
    - mlp_ratio = 4
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize patch embedding
        nn.init.xavier_uniform_(
            self.patch_embed.proj.weight.view(self.patch_embed.proj.weight.shape[0], -1)
        )
        nn.init.zeros_(self.patch_embed.proj.bias)

        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.zeros_(block.attn.in_proj_bias)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.out_proj.bias)
            for m in block.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input image

        Returns:
            patch_features: (B, num_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        return x


class ViT3DGPNetwork(nn.Module):
    """
    ViT + 3D positional encoding + Transformer for 3D goal prediction.

    Architecture:
    - Image tokens: ViT patches (trainable) with 3D PE (x,y,z from depth)
    - Language tokens: Flan-T5 (optional, frozen)
    - Gripper token: 6DoF pose + gripper width (optional)
    - Source token: learnable embedding for human/robot (optional)
    - Transformer: self-attention blocks
    - Output: N*196 GMM components, each predicting 13-dim (4x3 coords + 1 weight)

    Key difference from Dino3DGPNetwork:
    - Uses trainable ViT backbone instead of frozen DINO
    """

    def __init__(self, model_cfg):
        super(ViT3DGPNetwork, self).__init__()

        # Image processor for normalization (use standard ViT processor)
        self.image_processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )

        # ViT backbone (trainable, from scratch)
        self.backbone = ViTBackbone(
            img_size=model_cfg.img_size,
            patch_size=model_cfg.patch_size,
            in_channels=3,
            embed_dim=model_cfg.vit_embed_dim,
            depth=model_cfg.vit_depth,
            num_heads=model_cfg.vit_num_heads,
            mlp_ratio=model_cfg.vit_mlp_ratio,
            dropout=model_cfg.dropout,
        )

        # Get backbone dimensions
        self.pos_encoding_dim = 128
        self.hidden_dim = model_cfg.vit_embed_dim + self.pos_encoding_dim
        self.patch_size = model_cfg.patch_size

        # Training augmentations
        self.image_token_dropout = model_cfg.image_token_dropout

        # 3D Positional encoding (same as Dino3DGP)
        if model_cfg.use_fourier_pe:
            fourier_encoder = FourierPositionalEncoding(
                input_dim=3,
                num_frequencies=model_cfg.fourier_num_frequencies,
                include_input=model_cfg.fourier_include_input,
            )
            fourier_dim = fourier_encoder.output_dim
            self.pos_encoder = nn.Sequential(
                fourier_encoder,
                nn.Linear(fourier_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.pos_encoding_dim),
            )
        else:
            self.pos_encoder = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(),
                nn.Linear(128, self.pos_encoding_dim),
            )

        # Language encoder (same as Dino3DGP)
        self.use_text_embedding = model_cfg.use_text_embedding
        if self.use_text_embedding:
            self.text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            self.text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-base")
            self.text_encoder.requires_grad_(False)  # Freeze
            self.text_proj = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, self.hidden_dim),
            )

        # Gripper token encoder
        self.use_gripper_token = model_cfg.use_gripper_token
        if self.use_gripper_token:
            self.gripper_encoder = nn.Sequential(
                nn.Linear(10, 128),
                nn.ReLU(),
                nn.Linear(128, self.hidden_dim),
            )

        # Source token
        self.use_source_token = model_cfg.use_source_token
        if self.use_source_token:
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

        # Output head
        self.output_head = nn.Linear(self.hidden_dim, 13)

        # Register tokens
        self.num_registers = 4
        self.registers = nn.Parameter(
            torch.randn(1, self.num_registers, self.hidden_dim) * 0.02
        )

    def apply_image_token_dropout(self, tokens, patch_coords, num_cameras):
        """Apply image token dropout during training (same as Dino3DGP)."""
        if not self.training or not self.image_token_dropout:
            return tokens, patch_coords

        B, total_tokens, hidden_dim = tokens.shape
        tokens_per_camera = 196
        device = tokens.device

        dropout_type = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]

        if dropout_type == 0:
            return tokens, patch_coords
        elif dropout_type == 1:
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
            if num_cameras > 1:
                camera_to_drop = random.randint(0, num_cameras - 1)
                start_idx = camera_to_drop * tokens_per_camera
                end_idx = start_idx + tokens_per_camera

                mask = torch.ones(total_tokens, dtype=torch.bool, device=device)
                mask[start_idx:end_idx] = False

                tokens = tokens[:, mask, :]
                patch_coords = patch_coords[:, mask, :]
            return tokens, patch_coords

    def transform_to_world(self, points_cam, T_world_from_cam):
        """Transform points from camera frame to world frame."""
        B, N, _ = points_cam.shape
        ones = torch.ones(B, N, 1, device=points_cam.device)
        points_hom = torch.cat([points_cam, ones], dim=-1)
        points_world_hom = torch.einsum("bij,bnj->bni", T_world_from_cam, points_hom)
        points_world = points_world_hom[:, :, :3]
        return points_world

    def get_patch_centers(self, H, W, intrinsics, depth, extrinsics):
        """Compute 3D coordinates for patch centers using depth (multi-camera support)."""
        B, N, _, _ = depth.shape
        device = depth.device

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_patches = h_patches * w_patches

        y_centers = (
            torch.arange(h_patches, device=device) * self.patch_size
            + self.patch_size // 2
        )
        x_centers = (
            torch.arange(w_patches, device=device) * self.patch_size
            + self.patch_size // 2
        )
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")

        pixel_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        all_coords_world = []
        for cam_idx in range(N):
            pixel_coords_batch = pixel_coords.unsqueeze(0).expand(B, -1, -1)
            y_idx = pixel_coords_batch[:, :, 1].long()
            x_idx = pixel_coords_batch[:, :, 0].long()

            depth_cam = depth[:, cam_idx, :, :]
            depth_values = depth_cam[
                torch.arange(B, device=device).unsqueeze(1), y_idx, x_idx
            ]

            K = intrinsics[:, cam_idx, :, :]
            fx = K[:, 0, 0].unsqueeze(1)
            fy = K[:, 1, 1].unsqueeze(1)
            cx = K[:, 0, 2].unsqueeze(1)
            cy = K[:, 1, 2].unsqueeze(1)

            x_3d = (pixel_coords_batch[:, :, 0] - cx) * depth_values / fx
            y_3d = (pixel_coords_batch[:, :, 1] - cy) * depth_values / fy
            z_3d = depth_values

            patch_coords_cam = torch.stack([x_3d, y_3d, z_3d], dim=2).float()

            T_world_from_cam = extrinsics[:, cam_idx, :, :]
            patch_coords_world = self.transform_to_world(
                patch_coords_cam, T_world_from_cam
            )

            all_coords_world.append(patch_coords_world)

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
            tokens: (B, T, hidden_dim) token representations
        """
        B, N, C, H, W = image.shape

        # Extract ViT features for each camera (trainable)
        all_patch_features = []
        for cam_idx in range(N):
            cam_image = image[:, cam_idx, :, :, :]  # (B, 3, H, W)
            # Preprocess image (normalize like DINO does)
            inputs = self.image_processor(images=cam_image, return_tensors="pt")
            cam_image_processed = inputs["pixel_values"].to(cam_image.device)
            patch_features = self.backbone(
                cam_image_processed
            )  # (B, 196, vit_embed_dim)
            all_patch_features.append(patch_features)

        # Concatenate features from all cameras
        patch_features = torch.cat(
            all_patch_features, dim=1
        )  # (B, N*196, vit_embed_dim)

        # Get 3D positional encoding for patches (in world frame)
        patch_coords = self.get_patch_centers(H, W, intrinsics, depth, extrinsics)
        pos_encoding = self.pos_encoder(patch_coords)

        # Combine patch features with positional encoding
        tokens = torch.cat([patch_features, pos_encoding], dim=-1)

        # Apply image token dropout (training only)
        tokens, patch_coords = self.apply_image_token_dropout(tokens, patch_coords, N)

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

            lang_tokens = self.text_proj(text_embedding)
            tokens = torch.cat([tokens, lang_tokens], dim=1)
            mask = torch.cat([mask, text_tokens["attention_mask"] == 0], dim=1)

        # Add gripper token
        if self.use_gripper_token:
            grip_token = self.gripper_encoder(gripper_token).unsqueeze(1)
            tokens = torch.cat([tokens, grip_token], dim=1)
            mask = torch.cat(
                [mask, torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)], dim=1
            )

        # Add source token
        if self.use_source_token:
            source_indices = torch.tensor(
                [self.source_to_idx[s] for s in source], device=tokens.device
            )
            source_token = self.source_embeddings(source_indices).unsqueeze(1)
            tokens = torch.cat([tokens, source_token], dim=1)
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

        # Take only the patch tokens
        tokens = tokens[:, :num_patch_tokens]

        # Predict GMM parameters
        outputs = self.output_head(tokens)

        return outputs, patch_coords, tokens
