import os

import pytorch_lightning as pl
import torch
from torch.utils import data

from lfd3d.utils.data_utils import collate_pcd_fn


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg, seed):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.stage = None
        self.dataset_cfg = dataset_cfg
        self.seed = seed

        # setting root directory based on dataset type
        data_dir = os.path.expanduser(dataset_cfg.data_dir)
        self.root = data_dir

        # Subset of train to use for eval
        self.TRAIN_SUBSET_SIZE = 500

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "fit"):
        raise NotImplementedError("Not implemented for baseclass")

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            raise AttributeError(
                "train_dataset has not been set. Make sure to call setup() first."
            )
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if self.stage == "fit" else False,
            num_workers=self.num_workers,
            collate_fn=collate_pcd_fn,
        )

    def train_subset_dataloader(self):
        """A subset of train used for eval."""
        if not hasattr(self, "train_dataset"):
            raise AttributeError(
                "train_dataset has not been set. Make sure to call setup() first."
            )
        indices = torch.randint(
            0, len(self.train_dataset), (self.TRAIN_SUBSET_SIZE,)
        ).tolist()
        return data.DataLoader(
            data.Subset(self.train_dataset, indices),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pcd_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_datasets"):
            raise AttributeError(
                "val_datasets has not been set. Make sure to call setup() first."
            )
        return {
            tag: data.DataLoader(
                dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_pcd_fn,
            )
            for tag, dataset in self.val_datasets.items()
        }

    def test_dataloader(self):
        if not hasattr(self, "test_datasets"):
            raise AttributeError(
                "test_datasets has not been set. Make sure to call setup() first."
            )
        return {
            tag: data.DataLoader(
                dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_pcd_fn,
            )
            for tag, dataset in self.test_datasets.items()
        }
