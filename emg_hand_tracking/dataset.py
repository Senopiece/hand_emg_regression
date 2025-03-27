import torch
import h5py
from torch.utils.data import Dataset
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, ConcatDataset
from pathlib import Path
from typing import Sequence, Tuple


def default_transform(x, y):
    x = {
        "emg_chunk": torch.tensor(x["emg_chunk"], dtype=torch.float32),
        "joint_chunk": torch.tensor(x["joint_chunk"], dtype=torch.float32),
    }
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


class emg2poseSessionSlice(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        start: int,  # emg pos
        end: int,  # emg pos
        emg_samples_per_frame: int = 32,
        frames_per_item: int = 6,
        transform=None,
    ):
        self.h5_path = Path(h5_path)
        self.emg_samples_per_frame = emg_samples_per_frame
        self.emg_window_size = emg_samples_per_frame * frames_per_item
        self.start = start
        self.transform = transform if transform is not None else default_transform

        assert end > start

        self.items_count = (end - start) // self.emg_window_size

        assert self.items_count > 0

    def __len__(self):
        return self.items_count

    def __getitem__(self, idx):
        assert idx < self.items_count

        emg_start = self.start + self.emg_window_size * idx
        emg_end = emg_start + self.emg_window_size

        joint_indices = range(emg_start, emg_end, self.emg_samples_per_frame)

        with h5py.File(self.h5_path, "r") as f:
            timeseries = f["emg2pose"]["timeseries"]  # type: ignore
            joint_angles = timeseries["joint_angles"]  # type: ignore
            x = {
                "emg_chunk": timeseries["emg"][emg_start:emg_end],  # type: ignore
                "joint_chunk": joint_angles[joint_indices],  # type: ignore
            }
            y = joint_angles[emg_end]  # type: ignore

        return self.transform(x, y)


class DataModule(LightningDataModule):
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
                emg2poseSessionSlice(
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


def emg2pose_slices(base_path: str):
    return list(
        map(
            lambda e: (
                f"{base_path}/2022-04-07-1649318400-8125c-cv-emg-pose-train@2-recording-{e[0]}_left.hdf5",
                e[1],
                e[2],
            ),
            [
                ("1", 23000, 38000),
                ("1", 45000, 60000),
                ("1", 100000, 120000),
                ("10", 0, 35000),
                ("10", 101000, 166715),
                ("11", 10000, 35000),
                ("11", 55000, 75000),
                ("12", 0, 9000),
                ("12", 11000, 30000),
                ("12", 35000, 39000),
            ],
        )
    )
