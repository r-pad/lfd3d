import random
from collections import defaultdict
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import get_cosine_schedule_with_warmup
from lfd3d.models.pi0.pi0 import PI0Policy, make_att_2d_masks
from lfd3d.models.tax3d import calc_pcd_metrics
from torch import nn, optim


class Pi0GoalNetwork(nn.Module):
    """
    Modified version of PointNet2_super to work with this codebase
    """

    def __init__(self, model_cfg):
        super(Pi0GoalNetwork, self).__init__()
        self.pi0 = PI0Policy.from_pretrained("lerobot/pi0")
        self.pi0.eval()
        self.decoder = 0

    def forward(self, batch):
        breakpoint()
        # Use PI0's preprocessing pipeline
        batch = self.pi0.normalize_inputs(batch)
        images_processed, img_masks = self.pi0.prepare_images(batch)
        lang_tokens, lang_masks = self.pi0.prepare_language(batch)

        # Get prefix embeddings (vision+language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi0.model.embed_prefix(
            images_processed, img_masks, lang_tokens, lang_masks
        )

        # Process through robot-finetuned transformer
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

        tokens = prefix_outputs  # Robot-finetuned VLM features
        return tokens


class Pi0GoalModule(pl.LightningModule):
    """
    A goal generation module that handles model training, inference, evaluation and visualization.
    Based on CrossDisplacementModule but reworked to use the Articubot high-level model.
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

    def forward(self, batch):
        batch_size = batch["camera_front"].shape[0]
        device = batch["camera_front"].device
        pi0_batch = {
            "front": batch["camera_front"],
            "top": batch["camera_top"],
            "right": batch["camera_right"],
            "left": batch["camera_left"],
            "task": batch["caption"],
            "observation.state": torch.zeros(batch_size, 9, device=device),
        }
        pred_points = self.network(pi0_batch)
        init, gt = self.extract_gt_4_points(batch)
        loss = F.mse_loss(pred_points, gt)
        return None, loss

    @torch.no_grad()
    def predict(self, batch, progress=False):
        return None

    def log_viz_to_wandb(self, batch, pred_dict, weighted_displacement, tag):
        return None

    def training_step(self, batch, batch_idx):
        """
        Training step for the module. Logs training metrics and visualizations to wandb.
        """
        assert all(
            key in batch
            for key in ["camera_front", "camera_top", "camera_right", "camera_left"]
        ), "Enable multiview rendering to use Pi0Goal"

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
                pred_dict, weighted_displacement = all_pred_dict[0]
                # Store all sample preds for viz
                pred_dict[self.prediction_type]["all_pred"] = [
                    i[0][self.prediction_type]["pred"] for i in all_pred_dict
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
                self.log_viz_to_wandb(batch, pred_dict, weighted_displacement, "train")

        self.train_outputs.append(train_metrics)
        return loss

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
        return None

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

            # Per dataset metrics
            for metric, value in tag_metrics.items():
                log_dict[f"val_{val_tag}/{metric}"] = value

        # Avg over all datasets
        for metric, values in all_metrics.items():
            log_dict[f"val/{metric}"] = torch.stack(values).mean()

        # Minimize the linear combination of RMSE (reconstruction error) and -std (i.e. maximize diversity)
        # TODO: Find a better metric, and dynamically configure this....
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
        """
        Prediction step for model evaluation.
        """
        return None

    def on_predict_epoch_end(self):
        """
        Visualize random 5 batches in the test sets.
        """
        return None
