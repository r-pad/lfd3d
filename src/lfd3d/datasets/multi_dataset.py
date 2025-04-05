import math
import random
from pathlib import Path

import torch
import torchdatasets as td
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

from lfd3d.datasets.base_data import BaseDataModule
from lfd3d.datasets.hoi4d.hoi4d_dataset import HOI4DDataset
from lfd3d.datasets.rt1.rt1_dataset import RT1Dataset


class MultiDatasetDataModule(BaseDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):
        super().__init__(batch_size, val_batch_size, num_workers, dataset_cfg)
        self.val_tags = ["all"]

    def setup(self, stage: str = "fit"):
        self.stage = stage
        self.val_datasets = {}
        self.test_datasets = {}

        name_to_dataset_mapping = {"rt1": RT1Dataset, "hoi4d": HOI4DDataset}
        self.train_datasets_, self.val_datasets_, self.test_datasets_ = [], [], []

        for key, cfg in self.dataset_cfg.datasets.items():
            dataset_class = name_to_dataset_mapping[key]

            train_data = dataset_class(cfg.data_dir, cfg, "train")
            val_data = dataset_class(cfg.data_dir, cfg, "val")
            if train_data.cache_dir:
                train_data.cache(td.cachers.Pickle(Path(train_data.cache_dir)))
                val_data.cache(td.cachers.Pickle(Path(train_data.cache_dir) / "val"))
            self.train_datasets_.append(train_data)
            self.val_datasets_.append(val_data)
            self.test_datasets_.append(dataset_class(cfg.data_dir, cfg, "test"))

        self.train_dataset = data.ConcatDataset(self.train_datasets_)
        self.val_datasets[self.val_tags[0]] = data.ConcatDataset(self.val_datasets_)
        self.test_datasets[self.val_tags[0]] = data.ConcatDataset(self.test_datasets_)

        # For training with multiple datasets, we need a custom DistibutedSampler
        # and a custom BatchSampler. This is because the datasets items may have different
        # shapes, and can't be concatenated without resize/pad, which I would like to avoid.
        # The batch sampler ensures each batch contains only samples from the same dataset.
        #
        # However, this doesn't work out-of-the-box in distributed training.
        # Lightning tries to inject its own DistributedSampler which fails. So we need to
        # disable `use_distributed_sampler` in Trainer and add our own Distributed Sampler here.
        # The DistributedChunkSampler splits each dataset independently across GPUs
        #
        # Perhaps this is a little over-engineered ....

        # get the indices of elements in each dataset
        train_dataset_lengths = [0] + [len(i) for i in self.train_datasets_]
        val_dataset_lengths = [0] + [len(i) for i in self.val_datasets_]
        test_dataset_lengths = [0] + [len(i) for i in self.test_datasets_]
        train_dataset_indices, val_dataset_indices, test_dataset_indices = [], [], []

        for i in range(len(train_dataset_lengths) - 1):
            a_len, b_len = train_dataset_lengths[i], train_dataset_lengths[i + 1]
            train_dataset_indices.append(list(range(a_len, a_len + b_len)))

            a_len, b_len = val_dataset_lengths[i], val_dataset_lengths[i + 1]
            val_dataset_indices.append(list(range(a_len, a_len + b_len)))

            a_len, b_len = test_dataset_lengths[i], test_dataset_lengths[i + 1]
            test_dataset_indices.append(list(range(a_len, a_len + b_len)))

        # If distributed training, setup the distributed samplers
        # If not distributed training (or even with distributed training but
        # before the world has ben created), this will throw a RuntimeError.
        try:
            train_dist_sampler = DistributedChunkSampler(
                train_dataset_indices,
            )
            val_dist_sampler = DistributedChunkSampler(
                val_dataset_indices,
            )
            test_dist_sampler = DistributedChunkSampler(
                test_dataset_indices,
            )
        except RuntimeError:
            train_dist_sampler = train_dataset_indices
            val_dist_sampler = val_dataset_indices
            test_dist_sampler = test_dataset_indices

        # Use custom batch sampler to prevent batches mixing items from datasets
        self.train_batch_sampler = ChunkDatasetBatchSampler(
            train_dist_sampler,
            self.batch_size,
            shuffle=True if self.stage == "fit" else False,
        )
        self.val_batch_sampler = ChunkDatasetBatchSampler(
            val_dist_sampler,
            self.batch_size,
            shuffle=False,
        )
        self.test_batch_sampler = ChunkDatasetBatchSampler(
            test_dist_sampler,
            self.batch_size,
            shuffle=False,
        )

    def train_subset_dataloader(self):
        """A subset of train used for eval."""
        raise NotImplementedError("Eval on each dataset separately for now.")


class ChunkDatasetBatchSampler(torch.utils.data.BatchSampler):
    """This batch sampler ensures each batch contains data only
    from a single dataset, and for multiple datasets, it only uses as many samples
    from each as the smallest dataset.
    """

    def __init__(self, dataset_indices, batch_size, shuffle=True, drop_last=False):
        """
        Args:
            dataset_indices (List[List[int]]): List of index lists, one per dataset.
            batch_size (int): Batch size for each dataset.
            shuffle (bool): Whether to shuffle each dataset's indices.
            drop_last (bool): Whether to drop the last incomplete batch.
        """
        self.dataset_indices = dataset_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Compute the effective number of indices per dataset based on the smallest dataset.
        self.effective_length = min(len(indices) for indices in self.dataset_indices)

        # Precompute length (total number of batches)
        self._length = sum(
            math.floor(self.effective_length / self.batch_size)
            if drop_last
            else math.ceil(self.effective_length / self.batch_size)
            for _ in self.dataset_indices
        )

    def __len__(self):
        return self._length

    def __iter__(self):
        # We'll build batches for each dataset, using only effective_length samples.
        batches = []
        for indices in self.dataset_indices:
            # Work on a copy to avoid side effects.
            inds = indices.copy()
            if self.shuffle:
                random.shuffle(inds)
            # Subsample only as many as the smallest dataset.
            inds = inds[: self.effective_length]
            # Split into batches.
            # Note: torch.split returns tensors, so convert them back to lists.
            batched = torch.split(torch.tensor(inds), self.batch_size)
            # Optionally drop the last incomplete batch.
            for batch in batched:
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch.tolist())
        if self.shuffle:
            random.shuffle(batches)
        yield from batches


class DistributedChunkSampler(DistributedSampler):
    """This distributed sampler splits indices from different datasets independently,
    across GPUs."""

    def __init__(
        self,
        dataset_indices,
        num_replicas=None,
        rank=None,
        drop_last=False,
        shuffle=True,
        seed=0,
    ):
        self.dataset_indices = dataset_indices
        total_size = sum(len(indices) for indices in dataset_indices)
        # Initialize with total length (the params are not actually used though.)
        super().__init__(
            range(total_size), num_replicas, rank, shuffle, seed, drop_last
        )

    def __iter__(self):
        # split using rank / num replicas
        indices = [
            dataset_idx[self.rank : self.total_size : self.num_replicas]
            for dataset_idx in self.dataset_indices
        ]
        return iter(indices)
