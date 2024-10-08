from typing import Any

import lightning as L
import torch.nn.functional as F
from torch import optim


class ClassifierTrainingModule(L.LightningModule):
    def __init__(self, network, training_cfg) -> None:
        super().__init__()
        self.network = network
        self.lr = training_cfg.lr

    def forward(self, x):
        self.network(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.network(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        istrain = mode == "train"
        self.log("%s_loss" % mode, loss, prog_bar=istrain, add_dataloader_idx=False)
        self.log("%s_acc" % mode, acc, add_dataloader_idx=False)
        return {"loss": loss, "acc": acc, "preds": preds}

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            mode = "train_val"
        else:
            mode = "val"
        return self._calculate_loss(batch, mode=mode)

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="test")


class ClassifierInferenceModule(L.LightningModule):
    def __init__(self, network) -> None:
        super().__init__()
        self.network = network

    def forward(self, x):
        self.network(x)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        imgs, labels = batch
        preds = self.network(imgs)
        return {"preds": preds, "labels": labels}
