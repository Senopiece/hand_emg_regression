import torch
import h5py
from torch.utils.data import Dataset
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import warnings

# Having multiple workers will actually decrease the performance
warnings.filterwarnings(
    "ignore",
    message="The 'train_dataloader' does not have many workers which may be a bottleneck.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The 'val_dataloader' does not have many workers which may be a bottleneck.*",
    category=UserWarning,
)


def default_transform(d):
    return {
        "emg": torch.tensor(d["emg"], dtype=torch.float32),
        "poses": torch.tensor(d["poses"], dtype=torch.float32),
    }


class emg2poseInMemSessionSlice(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        start: int,  # in emg samples
        end: int,  # in emg samples
        step: int,  # in emg samples
        chennels: int = 16,
        emg_samples_per_frame: int = 32,
        frames_per_item: int = 6,
        transform=None,
    ):
        h5_path = Path(h5_path)
        transform = transform if transform is not None else default_transform

        emg_window_size = emg_samples_per_frame * frames_per_item

        with h5py.File(h5_path, "r") as f:
            timeseries = f["emg2pose"]["timeseries"]  # type: ignore
            joint_angles = timeseries["joint_angles"]  # type: ignore
            emg = timeseries["emg"]  # type: ignore

            def load_item(idx):
                emg_start = start + step * idx
                emg_end = emg_start + emg_window_size

                joint_indices = range(emg_start, emg_end + 1, emg_samples_per_frame)

                _emg = emg[emg_start:emg_end]  # type: ignore
                _poses = joint_angles[joint_indices]  # type: ignore

                assert len(_emg.shape) == 2  # type: ignore
                assert _emg.shape[1] == chennels  # type: ignore

                assert len(_poses.shape) == 2  # type: ignore
                assert _poses.shape[1] == 20  # 20 joint angles # type: ignore

                return transform(
                    {
                        "emg": _emg,
                        "poses": _poses,
                    }
                )

            try:
                count = (end - start - emg_window_size) // step
                assert count > 0
                self.items = [load_item(idx) for idx in range(count)]
            except Exception as e:
                raise Exception(f"Errored in {h5_path}") from e

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DataModule(LightningDataModule):
    def __init__(
        self,
        h5_slices: Mapping[
            str,
            Sequence[Tuple[str | Path, int, int, int, int]],
        ],  # train | val : (path, start, end, step, frames_per_item)
        emg_samples_per_frame: int = 32,
        batch_size: int = 64,
        transform=None,
    ):
        super().__init__()
        self.h5_slices = h5_slices
        self.emg_samples_per_frame = emg_samples_per_frame
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = [
            ConcatDataset(
                [
                    emg2poseInMemSessionSlice(
                        h5_path=path,
                        start=start,
                        end=end,
                        step=step,
                        emg_samples_per_frame=self.emg_samples_per_frame,
                        frames_per_item=frames_per_item,
                        transform=self.transform,
                    )
                    for path, start, end, step, frames_per_item in slices
                ]
            )
            for slices in [self.h5_slices["train"], self.h5_slices["val"]]
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self):
        return self.val_dataloader()


def emg2pose_slices(
    base_path: str,
    train_window: int,  # in frames
    val_window: int,  # in frames
    step: int,  # in emg
):
    d = {
        "train": [
            (1, 23000, 38000, step, train_window),
            (1, 45000, 60000, step, train_window),
            (1, 100000, 120000, step, train_window),
            (2, 1000, 16000, step, train_window),
            (2, 16500, 50000, step, train_window),
            (2, 64000, 118000, step, train_window),
            (2, 120000, 128000, step, train_window),
            (3, 1100, 16000, step, train_window),
            (3, 51000, 62966, step, train_window),
            (4, 1600, 11900, step, train_window),
            (4, 20000, 32400, step, train_window),
            (4, 101000, 123667, step, train_window),
            (5, 12000, 35000, step, train_window),
            (5, 39000, 50000, step, train_window),
            (5, 68000, 111000, step, train_window),
            (5, 120000, 130000, step, train_window),
            (6, 1000, 16000, step, train_window),
            (6, 31000, 105000, step, train_window),
            (6, 107000, 132000, step, train_window),
            (6, 135000, 156754, step, train_window),
            (7, 1000, 49000, step, train_window),
            (7, 50000, 85000, step, train_window),
            (7, 86000, 135706, step, train_window),
            (8, 0, 133000, step, train_window),
            # (8, 135000, 148923, step, train_window),
        ],
        "val": [
            (10, 0, 35000, step, val_window),
            (10, 101000, 166715, step, val_window),
            (11, 10000, 35000, step, val_window),
            (11, 55000, 75000, step, val_window),
            (12, 0, 9000, step, val_window),
            (12, 11000, 30000, step, val_window),
            (12, 35000, 39000, step, val_window),
        ],
    }
    res = {}
    for k, v in d.items():
        res[k] = list(
            map(
                lambda e: (
                    f"{base_path}/2022-04-07-1649318400-8125c-cv-emg-pose-train@2-recording-{e[0]}_left.hdf5",
                    *e[1:],
                ),
                v,
            )
        )
    return res
