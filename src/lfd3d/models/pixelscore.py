import random
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from diffusers import get_cosine_schedule_with_warmup
from torch import nn

from lfd3d.utils.viz_utils import generate_heatmap_from_points


class GoalPixelScoreModule(pl.LightningModule):
    def __init__(self, network, cfg):
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.mode = cfg.mode
        self.prediction_type = "pixel_score"
        self.loss_fn = cfg.model.loss_fn
        self.val_outputs: defaultdict[str, list[dict]] = defaultdict(list)
        self.train_outputs: list[dict] = []
        self.predict_outputs: defaultdict[str, list[dict]] = defaultdict(list)

        assert self.loss_fn in [
            "mse",
            "ce",
            "bce",
            "kldiv",
        ], "Must be MSE or cross_entropy"

        if self.mode == "train":
            self.run_cfg = cfg.training

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

        self.batch_size = self.run_cfg.batch_size
        self.val_batch_size = self.run_cfg.val_batch_size
        self.max_depth = cfg.dataset.max_depth

        # TODO: Make config param
        self.weight_loss_weight = 10  # weight of the weighted displacement loss term

    def configure_optimizers(self):
        assert self.mode == "train", "Needed only in training"
        optimizer = torch.optim.AdamW(
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
                "interval": "step",
            },
        }

    @torch.no_grad()
    def predict(self, batch):
        text_embedding = batch["text_embed"]
        logits = self.network(
            batch["normalized_rgbs"][:, 0], text_embedding=text_embedding
        )
        pred_coord, pred_pixel_score = self.sample_pixel_scores(
            logits
        )  # B, 2 ; B H W 3

        gt_coord = batch["pixel_coord"]  # B, 2
        assert gt_coord.shape[0] == pred_coord.shape[0]
        assert gt_coord.shape[1] == pred_coord.shape[1] == 2
        gt_pred_dist = torch.norm(
            (pred_coord.to(torch.float32) - gt_coord.to(torch.float32)), dim=-1
        )  # B,
        return (
            {
                self.prediction_type: {
                    "pred": pred_pixel_score,
                    "gt_pred_dist": gt_pred_dist,
                }
            },
            gt_pred_dist,
            pred_pixel_score,
            pred_coord,
        )

    def sample_pixel(self, batch, deterministic=True, channel_idx=0, tau=0.2):
        """
        Input:
            Batch: predict_pixel_score B, C, H, W
            sample_min: argmax / argmin
            channel: sample one channel
        Output:
            h,w: coordinates

        Sample only on first channel
        """
        batch_size, C, H, W = batch.shape
        if deterministic:
            idx = torch.argmin(batch.view(batch_size, C, -1), dim=-1)  # B, C
        else:
            # flat = batch.reshape(batch_size, C, H * W)       # (B, C, HW)
            # idx = flat.argmax(dim=-1).long()              # (B, C) long; Indices in [0, HW)
            probs = F.softmax(batch.view(batch_size, C, -1) / tau, dim=-1)
            idx = torch.stack(
                [torch.multinomial(probs[:, i], num_samples=1) for i in range(C)], dim=1
            )  # B,C,1
            idx = idx.squeeze(-1)  # B, C

        h, w = torch.unravel_index(idx, (H, W))
        coord = torch.stack([h, w], dim=-1)  # B, 2
        assert (
            coord.shape[0] == batch_size and coord.shape[1] == C and coord.shape[2] == 2
        )
        return coord[:, channel_idx]  # B, 2

    def sample_pixel_scores(self, predicted_pixel_score):
        batch_size, C, H, W = predicted_pixel_score.shape

        if self.loss_fn in ("ce", "bce", "kldiv"):
            coord = self.sample_pixel(predicted_pixel_score, deterministic=False).to(
                torch.float32
            )  # B, 2

        else:  # mse
            coord = self.sample_pixel(predicted_pixel_score).to(torch.float32)  # B, 2

        pixel_score = [
            generate_heatmap_from_points(
                torch.stack([coord[i], coord[i], coord[i]], dim=0).cpu().numpy(),
                np.array([H, W]),
            )
            for i in range(batch_size)
        ]  # B, H, W, 3

        pixel_score = np.stack(pixel_score, axis=0)
        return coord, torch.from_numpy(pixel_score).to(self.device)  # B H W 3

    def training_step(self, batch, batch_idx):
        self.train()
        loss = self(batch)

        train_metrics = {"loss": loss}

        if (
            self.global_step % self.additional_train_logging_period == 0
            and self.global_step != 0
        ) or (batch_idx == 0):
            self.eval()
            with torch.no_grad():
                pred_dict = {}
                _, coord_dist, pred_pixel_score, pred_coord = self.predict(batch)
                pred_dict["pred"] = pred_pixel_score
                pred_dict["coord_dist"] = coord_dist

            self.train()

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

        self.log_dict(
            log_dictionary,
            add_dataloader_idx=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_outputs.clear()

    def forward(self, batch):
        text_embedding = batch["text_embed"]

        x = batch["normalized_rgbs"][:, 0]  # rgb_init B 3 H W
        predicted_pixel_score = self.network(x, text_embedding=text_embedding)

        gt_pixel_score = batch["normalized_pixel_scores"][
            :, 1
        ]  # normalized_pixel_score_end B 3 H W

        gt_pixel_score_label = batch["pixel_score_label"]  # B 3 H W

        if self.loss_fn == "kldiv":
            loss = self.kl_div(
                predicted_pixel_score, -gt_pixel_score
            )  # Inverse; smaller distance = higher prob

        elif self.loss_fn == "ce":
            loss = self.spatial_hard_ce(predicted_pixel_score, gt_pixel_score_label)

        elif self.loss_fn == "bce":
            loss = self.hard_bce(predicted_pixel_score, gt_pixel_score_label)

        else:  # mse
            loss = F.mse_loss(predicted_pixel_score, gt_pixel_score)

        return loss

    def kl_div(self, logits, labels):
        """
        Computer KL divergence : target_size * target_size classes
        logits: (B, C, H, W)  raw scores
        labels: (B, C, H, W) pixel probabilities
        """
        b, c, h, w = logits.shape

        log_probs = F.log_softmax(logits.reshape(b * c, h * w), dim=-1)
        targets = F.softmax(labels.reshape(b * c, h * w), dim=-1)

        loss = F.kl_div(log_probs, targets, reduction="batchmean")
        return loss

    def spatial_hard_ce(self, logits, labels):
        """
        Computer cross_entropy : target_size * target_size classes
        logits: (B, C, H, W)  raw scores
        labels: (B, C, H, W) one hot encoded
        """
        b, c, h, w = logits.shape

        idx = labels.view(b, c, -1).argmax(dim=-1)

        # flatten spatial -> classes = H*W; treat (B,C) as batch
        loss = F.cross_entropy(
            logits.view(b * c, h * w), idx.view(b * c), reduction="mean"
        )
        return loss

    def hard_bce(self, logits, labels):
        """
        Compute the binary cross-entropy loss.

        Args:
            logits: the predicted logits
            labels: the ground truth labels (one-hot encoded)

        Returns:
            the computed loss
        """
        b, c, h, w = logits.shape
        with torch.no_grad():
            pos_per_c = labels.sum(dim=(0, 2, 3))
            neg_per_c = (b * h * w) - pos_per_c
            pos_w = (neg_per_c / (pos_per_c + 1e-8)).clamp(max=1e6)
        pos_w = pos_w.to(logits.device, dtype=logits.dtype)
        pos_w_bc = pos_w.view(1, c, 1, 1)

        weight = 1.0 + labels * (pos_w_bc - 1.0)

        return F.binary_cross_entropy(logits, labels, weight=weight, reduction="mean")

    def log_viz_to_wandb(self, batch, pred_dict, tag):
        """
        Log visualizations to wandb.

        Args:
            batch: the input batch
            pred_dict: the prediction dictionary
            pixel_score: the output model
            tag: the tag to use for logging
        """

        gt_pixel_score_vis = batch["pixel_score_vis"].cpu().numpy()
        gt_heatmap_vis = np.transpose(
            (batch["normalized_pixel_scores"][:, 1] * 255)
            .cpu()
            .numpy()
            .astype(np.uint8),
            (0, 2, 3, 1),
        )  # B H W 3

        rgbs = batch["rgbs"][:, 0].cpu().numpy()  # rgb_end B H W 3

        pred_pixel_score = pred_dict["pred"].cpu().numpy()  # B H W 3

        batch_size, h, w, _ = pred_pixel_score.shape

        assert batch_size == gt_pixel_score_vis.shape[0] == rgbs.shape[0]

        average_gt_pred_dict = pred_dict["coord_dist"].mean().item()

        gt_pixel_score_img_list = [
            wandb.Image(
                gt_pixel_score_vis[i],
                caption="Ground Truth Pixel Score",
            )
            for i in range(batch_size)
        ]
        gt_pixel_heatmap_img_list = [
            wandb.Image(
                gt_heatmap_vis[i],
                caption="Ground Truth Pixel Score",
            )
            for i in range(batch_size)
        ]
        predict_pixel_score_img_list = [
            wandb.Image(
                pred_pixel_score[i],
                caption="Predicted Pixel Score",
            )
            for i in range(batch_size)
        ]

        rgba_pixel_score_img_list = [
            wandb.Image(
                np.concatenate(
                    [rgbs[i], pred_pixel_score[i][:, :, 0][..., None]], axis=-1
                ),
                caption="Predicted Pixel Score",
            )
            for i in range(batch_size)
        ]

        viz_dict = {
            f"{tag}/ground_truth_pixel_score": gt_pixel_score_img_list,
            f"{tag}/predicted_pixel_score": predict_pixel_score_img_list,
            f"{tag}/rgba_pixel_score": rgba_pixel_score_img_list,
            f"{tag}/ground_truth_heatmap": gt_pixel_heatmap_img_list,
            f"{tag}/average_coord_dist": average_gt_pred_dict,
            "trainer/global_step": self.global_step,
        }
        wandb.log(viz_dict)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        val_tag = self.trainer.datamodule.val_tags[dataloader_idx]
        with torch.no_grad():
            pred_dict = {}
            _, coord_dist, pred_pixel_score, pred_coord = self.predict(batch)
            pred_dict["pred"] = pred_pixel_score
            pred_dict["coord_dist"] = coord_dist

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

    def on_validation_epoch_start(self):
        # Choose a random batch index for each validation epoch
        self.random_val_viz_idx = {
            k: random.randint(0, len(v) - 1)
            for k, v in self.trainer.val_dataloaders.items()
        }

    def on_validation_epoch_end(self):
        log_dict = {}
        all_metrics = {
            "coord_dist": [],
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
        eval_tag = self.trainer.datamodule.eval_tags[dataloader_idx]
        with torch.no_grad():
            pred_dict = {}
            _, coord_dist, pred_pixel_score, pred_coord = self.predict(batch)
            pred_dict["pred"] = pred_pixel_score
            pred_dict["coord_dist"] = coord_dist
            pred_dict["pred_coord"] = pred_coord

        self.predict_outputs[eval_tag].append(pred_dict)
        self.log_viz_to_wandb(batch, pred_dict, f"predict_{eval_tag}")
        return {
            "pred": pred_dict["pred"],
            "coord_dist": pred_dict["coord_dist"],
            "caption": batch["caption"],
            "pred_coord": pred_dict["pred_coord"],
        }

    def on_predict_epoch_end(self):
        log_dict = {}

        for dataloader_idx, eval_tag in enumerate(self.trainer.datamodule.eval_tags):
            if "test" not in eval_tag:
                continue
            tag_metrics = {
                "coord_dist": [],
            }
            pred_dict = {"pred": [], "coord_dist": []}
            vis_batch = {
                "pixel_score_vis": [],
                "rgbs": [],
                "normalized_pixel_scores": [],
            }

            pred_outputs = self.predict_outputs[eval_tag]
            coord_dist = torch.cat([x["coord_dist"] for x in pred_outputs])
            dataloader = self.trainer.predict_dataloaders[dataloader_idx]
            total_batches = len(dataloader)
            random_indices = random.sample(range(total_batches), min(5, total_batches))

            tag_metrics["coord_dist"] = coord_dist.mean()
            log_dict[f"predict_{eval_tag}/coord_dist"] = tag_metrics["coord_dist"]

            for i, (pred_output, batch) in enumerate(zip(pred_outputs, dataloader)):
                if i not in random_indices:
                    continue

                pred_dict["pred"].append(pred_output["pred"])
                pred_dict["coord_dist"].append(pred_output["coord_dist"])
                vis_batch["rgbs"].append(batch["rgbs"])
                vis_batch["pixel_score_vis"].append(batch["pixel_score_vis"])
                vis_batch["normalized_pixel_scores"].append(
                    batch["normalized_pixel_scores"]
                )

            vis_batch["rgbs"] = torch.cat(vis_batch["rgbs"], dim=0)
            vis_batch["pixel_score_vis"] = torch.cat(
                vis_batch["pixel_score_vis"], dim=0
            )
            vis_batch["normalized_pixel_scores"] = torch.cat(
                vis_batch["normalized_pixel_scores"], dim=0
            )

            pred_dict["pred"] = torch.cat(pred_dict["pred"], dim=0)
            pred_dict["coord_dist"] = torch.cat(pred_dict["coord_dist"], dim=0)

            self.log_viz_to_wandb(
                vis_batch,
                pred_dict,
                eval_tag,
            )
        # self.logger.log_metrics(log_dict, step=int(self.global_step))
        self.predict_outputs.clear()


class PixelScoreNetwork(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.use_text_embedding = model_cfg.use_text_embedding

        self.encoded_text_dim = 128  # Output dimension after encoding
        if self.use_text_embedding:
            self.text_encoder = nn.Linear(
                1152, self.encoded_text_dim
            )  # SIGLIP input dim

            self.film_predictor = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),  # [B, 128] -> [B, 256]
                nn.ReLU(),
                nn.Linear(256, 512 * 2),  # [B, 256] -> [B, 1024]
            )
            # Init as gamma=0 and beta=1
            self.film_predictor[-1].weight.data.zero_()
            self.film_predictor[-1].bias.data.copy_(
                torch.cat([torch.ones(512), torch.zeros(512)])
            )

        self.down1 = UnetDownAbstraction(mlp_list=[3, 64])
        self.down2 = UnetDownAbstraction(mlp_list=[64, 128])
        self.down3 = UnetDownAbstraction(mlp_list=[128, 256])
        self.down4 = UnetDownAbstraction(mlp_list=[256, 512])

        self.up1 = UnetUpAbstraction(mlp_list=[512, 256])
        self.up2 = UnetUpAbstraction(mlp_list=[256, 128])
        self.up3 = UnetUpAbstraction(mlp_list=[128, 64])
        self.up4 = UnetUpAbstraction(mlp_list=[64, 3], activate_relu_final=False)

    def forward(self, x, text_embedding=None):
        encoded_text = self.text_encoder(text_embedding)  # B 128
        film_params = self.film_predictor(encoded_text)  # B, 1024
        gamma, beta = film_params.chunk(2, dim=1)  # [B, 512] each
        gamma = gamma.view(-1, 512, 1, 1)  # [B, 512, 1,1] for broadcasting
        beta = beta.view(-1, 512, 1, 1)  # [B, 512, 1,1] for broadcasting

        d1 = self.down1(x)  # B 64 H/2 W/2
        d2 = self.down2(d1)  # B 128 H/4 W/4
        d3 = self.down3(d2)  # B 256 H/8 W/8
        d4 = self.down4(d3)  # B 512 H/16 W/16

        if self.use_text_embedding:
            d4 = d4 * gamma + beta
        u1 = self.up1(d4, d3)  # B 256 H/8 W/8
        u2 = self.up2(u1, d2)  # B 128 H/4 W/4
        u3 = self.up3(u2, d1)  # B 64 H/2 W/2
        u4 = self.up4(u3, x)  # B 3 H W
        return u4


class UnetDownAbstraction(nn.Module):
    def __init__(self, mlp_list):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        for i in range(len(mlp_list) - 1):
            self.conv_blocks.append(
                nn.Conv2d(mlp_list[i], mlp_list[i + 1], kernel_size=3, padding=1)
            )
            self.bn_blocks.append(nn.BatchNorm2d(mlp_list[i + 1]))

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            y: [B, 2C, H/2, W/2]
        """
        for conv, bn in zip(self.conv_blocks, self.bn_blocks):
            x = F.relu(bn(conv(x)))

        return self.pool(x)


class UnetUpAbstraction(nn.Module):
    def __init__(self, mlp_list=None, activate_relu_final=False):
        if mlp_list is None:
            mlp_list = [3, 64]
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.activate_relu = activate_relu_final

        for i in range(len(mlp_list) - 1):
            self.conv_blocks.append(
                nn.Conv2d(mlp_list[i], mlp_list[i + 1], kernel_size=3, padding=1)
            )
            self.bn_blocks.append(nn.BatchNorm2d(mlp_list[i + 1]))

        self.up_conv = nn.ConvTranspose2d(
            mlp_list[0], mlp_list[0] - mlp_list[-1], kernel_size=2, stride=2
        )

    def forward(self, x, skip_x):
        """
        Input:
            x: [B, C, H/2, W/2]
            skip_x: [B, C/2, H, W]
        Output:
            y: [B, C, H, W]
        """
        x = self.up_conv(x)  # [B, C/2, H, W]
        x = torch.cat([x, skip_x], dim=1)
        for i, (conv, bn) in enumerate(zip(self.conv_blocks, self.bn_blocks)):
            if i == len(self.conv_blocks) - 1 and not self.activate_relu:
                x = bn(conv(x))
            else:
                x = F.relu(bn(conv(x)))
        return x
