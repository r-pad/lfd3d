import lightning as L
import torch
import torch.utils.data as data
import torchvision as tv
from torchvision import transforms as T


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, root, batch_size, num_workers):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # Anything that needs to be done to download.
        tv.datasets.CIFAR10(self.root, train=True, download=True)
        tv.datasets.CIFAR10(self.root, train=False, download=True)

    def setup(self, stage: str):
        # Set up data augmentation.
        train_transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.ToTensor(),
                T.Normalize(
                    [0.49139968, 0.48215841, 0.44653091],
                    [0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )

        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    [0.49139968, 0.48215841, 0.44653091],
                    [0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )

        # We want to split the training set into train and val. But we don't want transforms on val.
        # So we create two datasets, and make sure that the split is consistent between them.
        train_dataset = tv.datasets.CIFAR10(
            self.root, train=True, transform=train_transform
        )
        val_dataset = tv.datasets.CIFAR10(
            self.root, train=True, transform=test_transform
        )
        generator = torch.Generator().manual_seed(42)
        self.train_set, _ = torch.utils.data.random_split(
            train_dataset, [45000, 5000], generator=generator
        )
        train_val_set, val_set = torch.utils.data.random_split(
            val_dataset, [45000, 5000], generator=generator
        )
        self.train_val_set = train_val_set
        self.val_set = val_set

        # Test set.
        self.test_set = tv.datasets.CIFAR10(
            self.root, train=False, transform=test_transform
        )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return [
            data.DataLoader(
                self.train_val_set,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            ),
            data.DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            ),
        ]

    def test_dataloader(self):
        return data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
