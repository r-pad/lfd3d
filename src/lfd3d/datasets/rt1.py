import os

import pytorch_lightning as pl
import tensorflow_datasets as tfds
import torch.utils.data as data


class RT1Dataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.builder = tfds.builder_from_directory(builder_dir=root)
        self.dataset = self.builder.as_dataset(split="train")
        self.num_demos = len(self.dataset)
        self.dataset_cfg = dataset_cfg

        self.size = self.num_demos
        self.PAD_SIZE = 1000

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        breakpoint()
        data = next(iter(self.dataset.skip(index).take(1)))
        return 0


class RT1DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.stage = None
        self.dataset_cfg = dataset_cfg

        # setting root directory based on dataset type
        data_dir = os.path.expanduser(dataset_cfg.data_dir)
        self.root = data_dir

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "fit"):
        self.stage = stage

        self.train_dataset = RT1Dataset(self.root, self.dataset_cfg, "train")
        self.val_dataset = RT1Dataset(self.root, self.dataset_cfg, "val")
        self.test_dataset = RT1Dataset(self.root, self.dataset_cfg, "test")

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if self.stage == "train" else False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = data.DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_dataloader
