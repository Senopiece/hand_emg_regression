from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, ConcatDataset
from pathlib import Path
from typing import Sequence, Tuple

from data import e2pDataset


class EMGDataModule(LightningDataModule):
    def __init__(
        self,
        h5_slices: Sequence[Tuple[str | Path, int, int]],  # (path, start, end)
        emg_samples_per_frame: int = 32,
        frames_per_item: int = 6,
        batch_size: int = 64,
        val_split: float = 0.2,
        num_workers: int = 16,
        transform=None,
    ):
        super().__init__()
        self.h5_slices = h5_slices
        self.emg_samples_per_frame = emg_samples_per_frame
        self.frames_per_item = frames_per_item
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        full_dataset = ConcatDataset(
            [
                e2pDataset(
                    h5_path=path,
                    start=start,
                    end=end,
                    emg_samples_per_frame=self.emg_samples_per_frame,
                    frames_per_item=self.frames_per_item,
                    transform=self.transform,
                )
                for path, start, end in self.h5_slices
            ]
        )

        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        # optional: reuse val
        return self.val_dataloader()
