import random
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lfd3d.models.base_module import BaseModule
from lfd3d.models.pi0.pi0 import (
    FeatureType,
    PI0Policy,
    PolicyFeature,
    make_att_2d_masks,
)
from lfd3d.models.tax3d import calc_pcd_metrics
from lfd3d.utils.viz_utils import (
    get_action_anchor_pcd,
    get_img_and_track_pcd,
    invert_augmentation_and_normalization,
    project_pcd_on_image,
)
from pytorch3d.renderer import (
    AlphaCompositor,
    OrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    look_at_view_transform,
)
from torch import nn


class Pi0GoalNetwork(nn.Module):
    """
    Modified version of PointNet2_super to work with this codebase
    """

    def __init__(self, model_cfg):
        super(Pi0GoalNetwork, self).__init__()
        self.pi0 = PI0Policy.from_pretrained("lerobot/pi0")

        # Freeze the VLM
        for param in self.pi0.parameters():
            param.requires_grad = False

        # Missing in the default config for some reason
        # https://huggingface.co/lerobot/pi0/discussions/18
        self.pi0.config.input_features = {
            "observation.images.front": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.top": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.right": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.left": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        }
        vlm_dim = 2048  # From PaliGemma

        hidden_dim = model_cfg.hidden_dim
        num_layers = model_cfg.num_layers
        num_heads = model_cfg.num_heads

        # Project VLM tokens to hidden dim
        self.input_proj = nn.Linear(vlm_dim, hidden_dim)

        # Learnable gripper query tokens (4 points)
        self.gripper_queries = nn.Parameter(torch.randn(4, hidden_dim))

        # Cross-attention transformer layers
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, num_heads, batch_first=True),
            num_layers,
        )

        # Output projection to xyz (3 channels per point)
        self.output_proj = nn.Linear(hidden_dim, 3)

    def forward(self, batch):
        # Use PI0's preprocessing pipeline
        batch_size = batch["observation.images.front"].shape[0]
        batch = self.pi0.normalize_inputs(batch)
        images_processed, img_masks = self.pi0.prepare_images(batch)
        lang_tokens, lang_masks = self.pi0.prepare_language(batch)

        # Get prefix embeddings (vision+language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi0.model.embed_prefix(
            images_processed, img_masks, lang_tokens, lang_masks
        )

        # Process through Pi0's PaliGemma backbone
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        (prefix_outputs, suffix_outputs), past_key_values = (
            self.pi0.model.paligemma_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                inputs_embeds=[prefix_embs, None],  # Only vision+language
                past_key_values=None,
                use_cache=False,
                fill_kv_cache=False,
            )
        )

        # Pi0 forces bf16
        prefix_outputs = prefix_outputs.to(self.input_proj.weight.dtype)

        # Robot-finetuned VLM features
        vlm_tokens = self.input_proj(prefix_outputs)  # [batch, seq_len, hidden_dim]
        queries = self.gripper_queries.unsqueeze(0).expand(batch_size, -1, -1)
        output = self.decoder(queries, vlm_tokens)

        pred_points = self.output_proj(output)
        return pred_points


class Pi0GoalModule(BaseModule):
    """
    A goal generation module that handles model training, inference, evaluation and visualization.
    """

    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.prediction_type = self.model_cfg.type  # flow or point
        self.mode = cfg.mode  # train or eval
        self.val_outputs: defaultdict[str, List[Dict]] = defaultdict(list)
        self.train_outputs: List[Dict] = []
        self.predict_outputs: defaultdict[str, List[Dict]] = defaultdict(list)

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

        self.renderer = self.setup_renderer()
        self.cam_names = ["front", "top", "right", "left"]

    def setup_renderer(self):
        # Use pytorch3d plot_scene to visualize camera locations wrt pcd
        camera_configs = {
            "front": {"eye": (0, 0, -3), "up": (0, -1, 0)},
            "top": {"eye": (0, -3, 0), "up": (0, 0, -1)},
            "right": {"eye": (3, 0, 0), "up": (0, -1, 0)},
            "left": {"eye": (-3, 0, 0), "up": (0, -1, 0)},
        }
        all_R, all_T = [], []
        for name, config in camera_configs.items():
            R, T = look_at_view_transform(
                eye=[config["eye"]],
                at=[(0, 0, 0)],  # Look at origin
                up=[config["up"]],  # Up direction
            )
            all_R.append(R)
            all_T.append(T)

        batched_R = torch.cat(all_R, dim=0)
        batched_T = torch.cat(all_T, dim=0)
        batched_cameras = OrthographicCameras(R=batched_R, T=batched_T, focal_length=0.6)

        # 224 is a common shape for SiGLIP/DINO etc
        # Radius and points-per-pixel chosen arbitrarily to render decent images
        raster_settings = PointsRasterizationSettings(
            image_size=224,
            radius=0.02,  # Point size
            points_per_pixel=10,
            bin_size=64,
            max_points_per_bin=50000,
        )

        rasterizer = PointsRasterizer(
            cameras=batched_cameras, raster_settings=raster_settings
        )
        compositor = AlphaCompositor()
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)
        return renderer

    def render_multiview(self, point_cloud):
        # Only 3 channels and in valid range
        assert point_cloud.features_padded().shape[2] == 3
        assert point_cloud.features_padded().min() >= 0
        assert point_cloud.features_padded().max() <= 1

        # Sufficient density of points to render decent images
        # HACK: Need a better way to do this....
        assert point_cloud.points_padded().shape[1] >= 8192

        self.renderer.rasterizer.cameras.to(point_cloud.device).float()

        rendered_images = []
        for i in range(len(point_cloud)):
            single_pc = point_cloud[i : i + 1]  # Keep batch dim

            # 0mean 1std before rendering for consistent camera placements
            points = single_pc.points_padded()
            points = (points - points.mean(1)) / points.std()
            single_pc = single_pc.update_padded(points)

            single_pc_expanded = single_pc.extend(4)  # Expand to 4 copies
            with torch.no_grad():
                images = self.renderer(single_pc_expanded)  # [4, H, W, C]
            rendered_images.append(images)

        rendered_images = torch.stack(rendered_images).permute(
            1, 0, 2, 3, 4
        )  # [4, B, H, W, C]
        multiview_image_dict = {
            f"camera_{name}": img for name, img in zip(self.cam_names, rendered_images)
        }
        return multiview_image_dict

    def forward(self, batch):
        batch_size = batch["camera_front"].shape[0]
        device = batch["camera_front"].device

        pi0_batch = {
            "observation.state": torch.zeros(batch_size, 14, device=device),  # not used
            "observation.images.front": batch["camera_front"].permute(0, 3, 1, 2),
            "observation.images.top": batch["camera_top"].permute(0, 3, 1, 2),
            "observation.images.right": batch["camera_right"].permute(0, 3, 1, 2),
            "observation.images.left": batch["camera_left"].permute(0, 3, 1, 2),
            "task": batch["caption"],
        }
        pred_points = self.network(pi0_batch)

        init, gt = self.extract_gt_4_points(batch)
        loss = F.mse_loss(pred_points, gt)
        return None, loss

    @torch.no_grad()
    def predict(self, batch, progress=False):
        """
        Compute prediction for a given batch.
        NOTE: To maintain consistency with the codebase,
        this returns the displacement to the goal position,
        not the actual goal position itself.

        Args:
            batch: the input batch
            progress: whether to show progress bar
        """
        batch_size = batch["camera_front"].shape[0]
        device = batch["camera_front"].device

        pi0_batch = {
            "observation.state": torch.zeros(batch_size, 14, device=device),  # not used
            "observation.images.front": batch["camera_front"].permute(0, 3, 1, 2),
            "observation.images.top": batch["camera_top"].permute(0, 3, 1, 2),
            "observation.images.right": batch["camera_right"].permute(0, 3, 1, 2),
            "observation.images.left": batch["camera_left"].permute(0, 3, 1, 2),
            "task": batch["caption"],
        }
        pred_points = self.network(pi0_batch)

        init, gt = self.extract_gt_4_points(batch)

        pred_displacement = pred_points - init
        return {self.prediction_type: {"pred": pred_displacement}}

    def log_viz_to_wandb(self, batch, pred_dict, tag):
        """
        Log visualizations to wandb.

        Args:
            batch: the input batch
            pred_dict: the prediction dictionary
            tag: the tag to use for logging
        """
        batch_size = batch[self.label_key].points_padded().shape[0]
        # pick a random sample in the batch to visualize
        viz_idx = np.random.randint(0, batch_size)
        RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)
        max_depth = self.max_depth

        all_pred = pred_dict[self.prediction_type]["all_pred"][viz_idx].cpu().numpy()
        N = all_pred.shape[0]
        end2start = np.linalg.inv(batch["start2end"][viz_idx].cpu().numpy())

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
        rmse = pred_dict["rmse"][viz_idx]
        anchor_pcd = batch["anchor_pcd"].points_padded()[viz_idx].cpu().numpy()
        multiview_imgs = [
            batch["camera_front"][viz_idx].cpu().numpy(),
            batch["camera_top"][viz_idx].cpu().numpy(),
            batch["camera_right"][viz_idx].cpu().numpy(),
            batch["camera_left"][viz_idx].cpu().numpy(),
        ]

        pcd, gt = self.extract_gt_4_points(batch)
        pcd, gt = pcd.cpu().numpy()[viz_idx], gt.cpu().numpy()[viz_idx]
        all_pred_pcd = pcd + all_pred
        gt_pcd = gt
        padding_mask = torch.ones(gt.shape[0]).bool().numpy()

        # Move center back from action_pcd to the camera frame
        # and invert augmentation transforms before viz
        pcd_mean = batch["pcd_mean"][viz_idx].cpu().numpy()
        pcd_std = batch["pcd_std"][viz_idx].cpu().numpy()
        R = batch["augment_R"][viz_idx].cpu().numpy()
        t = batch["augment_t"][viz_idx].cpu().numpy()
        scene_centroid = batch["augment_C"][viz_idx].cpu().numpy()

        pcd = invert_augmentation_and_normalization(
            pcd, pcd_mean, pcd_std, R, t, scene_centroid
        )
        anchor_pcd = invert_augmentation_and_normalization(
            anchor_pcd, pcd_mean, pcd_std, R, t, scene_centroid
        )
        all_pred_pcd = invert_augmentation_and_normalization(
            all_pred_pcd, pcd_mean, pcd_std, R, t, scene_centroid
        )
        gt_pcd = invert_augmentation_and_normalization(
            gt_pcd, pcd_mean, pcd_std, R, t, scene_centroid
        )

        # All points cloud are in the start image's coordinate frame
        # We need to visualize the end image, therefore need to apply transform
        # Transform the point clouds to align with end image coordinate frame
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

        # Project tracks to image and save
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
        ###

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
            anchor_pcd.shape[0],
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

        # Multiview image
        img1 = cv2.hconcat([multiview_imgs[0], multiview_imgs[1]])
        img2 = cv2.hconcat([multiview_imgs[2], multiview_imgs[3]])
        multiview_img = cv2.vconcat([img1, img2])
        wandb_multiview_img = wandb.Image(
            multiview_img,
            caption="Multiview image rendered from pcd - [Front / Top / Right / Left]",
        )

        viz_dict = {
            f"{tag}/track_projected_to_rgb": wandb_proj_img,
            f"{tag}/image_and_tracks_pcd": wandb.Object3D(viz_pcd),
            f"{tag}/action_anchor_pcd": wandb.Object3D(action_anchor_pcd),
            f"{tag}/multiview_image": wandb_multiview_img,
            "trainer/global_step": self.global_step,
        }

        wandb.log(viz_dict)

    def training_step(self, batch, batch_idx):
        """
        Training step for the module. Logs training metrics and visualizations to wandb.
        """
        point_cloud = batch["anchor_pcd"]
        multiview_image_dict = self.render_multiview(point_cloud)
        batch.update(multiview_image_dict)

        self.train()
        batch_size = batch[self.label_key].points_padded().shape[0]

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
            self.eval()
            with torch.no_grad():
                all_pred_dict = [self.predict(batch)]
                # Use one sample for computing other metrics
                pred_dict = all_pred_dict[0]
                # Store all sample preds for viz
                pred_dict[self.prediction_type]["all_pred"] = [
                    i[self.prediction_type]["pred"] for i in all_pred_dict
                ]
                pred_dict[self.prediction_type]["all_pred"] = torch.stack(
                    pred_dict[self.prediction_type]["all_pred"]
                ).permute(1, 0, 2, 3)
            self.train()  # Switch back to training mode

            init, gt = self.extract_gt_4_points(batch)
            gt_displacement = gt - init

            padding_mask = torch.ones(gt.shape[0], gt.shape[1]).bool()
            pcd_std = batch["pcd_std"]
            ground_truth = batch[self.label_key].to(self.device)
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
                ####################################################
                # logging visualizations
                ####################################################
                self.log_viz_to_wandb(batch, pred_dict, "train")

        self.train_outputs.append(train_metrics)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step for the module. Logs validation metrics and visualizations to wandb.
        """
        point_cloud = batch["anchor_pcd"]
        multiview_image_dict = self.render_multiview(point_cloud)
        batch.update(multiview_image_dict)

        val_tag = self.trainer.datamodule.val_tags[dataloader_idx]
        self.eval()
        with torch.no_grad():
            all_pred_dict = [self.predict(batch)]
            pred_dict = all_pred_dict[0]

            # Store all sample preds for viz
            pred_dict[self.prediction_type]["all_pred"] = [
                i[self.prediction_type]["pred"] for i in all_pred_dict
            ]
            pred_dict[self.prediction_type]["all_pred"] = torch.stack(
                pred_dict[self.prediction_type]["all_pred"]
            ).permute(1, 0, 2, 3)

        init, gt = self.extract_gt_4_points(batch)
        gt_displacement = gt - init

        padding_mask = torch.ones(gt.shape[0], gt.shape[1]).bool()
        pcd_std = batch["pcd_std"]
        ground_truth = batch[self.label_key].to(self.device)
        pred_dict = calc_pcd_metrics(
            pred_dict,
            init,
            pred_dict[self.prediction_type]["all_pred"],
            gt_displacement,
            pcd_std,
            padding_mask,
        )
        self.val_outputs[val_tag].append(pred_dict)

        ####################################################
        # logging visualizations
        ####################################################
        if (
            batch_idx == self.random_val_viz_idx[val_tag]
            and self.trainer.is_global_zero
        ):
            self.log_viz_to_wandb(batch, pred_dict, f"val_{val_tag}")
        return pred_dict

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for model evaluation.
        """
        point_cloud = batch["anchor_pcd"]
        multiview_image_dict = self.render_multiview(point_cloud)
        batch.update(multiview_image_dict)

        eval_tag = self.trainer.datamodule.eval_tags[dataloader_idx]
        n_samples_wta = self.trainer.datamodule.n_samples_wta

        all_pred_dict = [self.predict(batch)]

        pred_dict = all_pred_dict[0]
        # Store all sample preds for viz
        pred_dict[self.prediction_type]["all_pred"] = [
            i[self.prediction_type]["pred"] for i in all_pred_dict
        ]
        pred_dict[self.prediction_type]["all_pred"] = torch.stack(
            pred_dict[self.prediction_type]["all_pred"]
        ).permute(1, 0, 2, 3)

        init, gt = self.extract_gt_4_points(batch)
        gt_displacement = gt - init

        padding_mask = torch.ones(gt.shape[0], gt.shape[1]).bool()
        pcd_std = batch["pcd_std"]
        ground_truth = batch[self.label_key].to(self.device)
        pred_dict = calc_pcd_metrics(
            pred_dict,
            init,
            pred_dict[self.prediction_type]["all_pred"],
            gt_displacement,
            pcd_std,
            padding_mask,
        )
        self.predict_outputs[eval_tag].append(pred_dict)

        return {
            "rmse": pred_dict["rmse"],
            "chamfer_dist": pred_dict["chamfer_dist"],
            "wta_rmse": pred_dict["wta_rmse"],
            "wta_chamfer_dist": pred_dict["wta_chamfer_dist"],
            "vid_name": batch["vid_name"],
            "caption": batch["caption"],
        }

    def on_predict_epoch_end(self):
        """
        Visualize random 5 batches in the test sets.
        """
        for dataloader_idx, eval_tag in enumerate(self.trainer.datamodule.eval_tags):
            if "test" not in eval_tag:
                continue

            pred_outputs = self.predict_outputs[eval_tag]
            rmse = torch.cat([x["rmse"] for x in pred_outputs])
            chamfer_dist = torch.cat([x["chamfer_dist"] for x in pred_outputs])
            cross_displacement, all_cross_displacement = [], []
            for i, pred in enumerate(pred_outputs):
                cross_displacement.extend(pred["cross_displacement"]["pred"])
                all_cross_displacement.extend(pred["cross_displacement"]["all_pred"])

            dataloader = self.trainer.predict_dataloaders[dataloader_idx]
            total_batches = len(dataloader)
            random_indices = random.sample(range(total_batches), min(5, total_batches))

            for i, batch in enumerate(dataloader):
                if i not in random_indices:
                    continue

                batch_len = len(batch["caption"])
                for idx in range(batch_len):
                    pred_dict = self.compose_pred_dict_for_viz(
                        rmse,
                        chamfer_dist,
                        cross_displacement,
                        all_cross_displacement,
                        idx,
                    )
                    viz_batch = self.compose_batch_for_viz(batch, idx)
                    self.log_viz_to_wandb(
                        viz_batch,
                        pred_dict,
                        eval_tag,
                    )
        self.predict_outputs.clear()
