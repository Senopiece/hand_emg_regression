import torch
import h5py
from torch.utils.data import Dataset
from pathlib import Path


def default_transform(x, y):
    x = {
        "emg_chunk": torch.tensor(x["emg_chunk"], dtype=torch.float32),
        "joint_chunk": torch.tensor(x["joint_chunk"], dtype=torch.float32),
    }
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


class e2pDataset(Dataset):
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
