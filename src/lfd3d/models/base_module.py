import random

import pytorch_lightning as pl
import torch
from diffusers import get_cosine_schedule_with_warmup
from pytorch3d.structures import Pointclouds
from torch import optim


class BaseModule(pl.LightningModule):
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
        """
        Extract 3+1 hand-picked points for each embodiment to roughly form a "gripper"
        """
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

    def compose_pred_dict_for_viz(
        self, rmse, chamfer_dist, cross_displacement, all_cross_displacement, idx
    ):
        return {
            "rmse": [rmse[idx]],
            "chamfer_dist": [chamfer_dist[idx]],
            "cross_displacement": {
                "pred": cross_displacement[idx][None],
                "all_pred": all_cross_displacement[idx][None],
            },
        }

    def compose_batch_for_viz(self, batch, idx):
        viz_batch = {}
        for key in batch.keys():
            if type(batch[key]) == Pointclouds:
                pcd = batch[key].points_padded()[idx]
                viz_batch[key] = Pointclouds(points=pcd[None])
            elif key in ["rgbs", "depths"]:
                viz_batch[key] = batch[key][idx][None]
            else:
                viz_batch[key] = [batch[key][idx]]
        return viz_batch
