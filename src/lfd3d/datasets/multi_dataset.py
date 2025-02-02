import random
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchdatasets as td

from lfd3d.datasets.hoi4d import HOI4DDataset
from lfd3d.datasets.rt1 import RT1Dataset
from lfd3d.utils.data_utils import collate_pcd_fn


class MultiDatasetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.stage = None
        self.dataset_cfg = dataset_cfg

        # Subset of train to use for eval
        self.TRAIN_SUBSET_SIZE = 500

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "fit"):
        self.stage = stage

        name_to_dataset_mapping = {"rt1": RT1Dataset, "hoi4d": HOI4DDataset}
        self.train_datasets, self.val_datasets, self.test_datasets = [], [], []

        for key, cfg in self.dataset_cfg.datasets.items():
            dataset_class = name_to_dataset_mapping[key]

            train_data = dataset_class(cfg.data_dir, cfg, "train")
            if train_data.cache_dir:
                train_data.cache(td.cachers.Pickle(Path(train_data.cache_dir)))
            self.train_datasets.append(train_data)

            self.val_datasets.append(dataset_class(cfg.data_dir, cfg, "val"))
            self.test_datasets.append(dataset_class(cfg.data_dir, cfg, "test"))

        self.train_dataset = data.ConcatDataset(self.train_datasets)
        self.val_dataset = data.ConcatDataset(self.val_datasets)
        self.test_dataset = data.ConcatDataset(self.test_datasets)

        train_dataset_lengths = [0] + [len(i) for i in self.train_datasets]
        val_dataset_lengths = [0] + [len(i) for i in self.val_datasets]
        test_dataset_lengths = [0] + [len(i) for i in self.test_datasets]
        train_dataset_indices, val_dataset_indices, test_dataset_indices = [], [], []

        for i in range(len(train_dataset_lengths) - 1):
            a_len, b_len = train_dataset_lengths[i], train_dataset_lengths[i + 1]
            train_dataset_indices.append(list(range(a_len, a_len + b_len)))

            a_len, b_len = val_dataset_lengths[i], val_dataset_lengths[i + 1]
            val_dataset_indices.append(list(range(a_len, a_len + b_len)))

            a_len, b_len = test_dataset_lengths[i], test_dataset_lengths[i + 1]
            val_dataset_indices.append(list(range(a_len, a_len + b_len)))

        self.train_batch_sampler = ChunkDatasetBatchSampler(
            train_dataset_indices,
            self.batch_size,
            shuffle=True if self.stage == "fit" else False,
        )
        self.val_batch_sampler = ChunkDatasetBatchSampler(
            val_dataset_indices, self.batch_size, shuffle=False
        )
        self.test_batch_sampler = ChunkDatasetBatchSampler(
            test_dataset_indices, self.batch_size, shuffle=False
        )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            collate_fn=collate_pcd_fn,
            batch_sampler=self.train_batch_sampler,
        )

    def train_subset_dataloader(self):
        """A subset of train used for eval."""
        raise NotImplementedError("Eval on each dataset separately for now.")

    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=collate_pcd_fn,
            batch_sampler=self.val_batch_sampler,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = data.DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            collate_fn=collate_pcd_fn,
            batch_sampler=self.test_batch_sampler,
        )
        return test_dataloader


class ChunkDatasetBatchSampler(torch.utils.data.Sampler):
    """This batch sampler ensures each batch contains data only
    from a single dataset."""

    def __init__(self, dataset_indices, batch_size, shuffle=True):
        self.dataset_indices = dataset_indices
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        # integer division to round up for incomplete batches
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.dataset_indices
        )

    def __iter__(self):
        if self.shuffle:
            [random.shuffle(i) for i in self.dataset_indices]
        all_batches = [
            torch.split(torch.tensor(i), self.batch_size) for i in self.dataset_indices
        ]
        all_batches = list(sum(all_batches, ()))
        all_batches = [batch.tolist() for batch in all_batches]
        if self.shuffle:
            random.shuffle(all_batches)

        return iter(all_batches)
