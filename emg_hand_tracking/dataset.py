import math
import zipfile
import numpy as np
import torch
import yaml
import warnings
from torch.utils.data import Dataset, Sampler
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from typing import List, NamedTuple
from tqdm import tqdm
from scipy.interpolate import interp1d


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


class HandEmgTuple(NamedTuple):
    frame: np.ndarray  # (20,), float32 expected
    emg: np.ndarray  # (64, C), float32 expected


class HandEmgRecording(NamedTuple):
    couples: List[HandEmgTuple]
    sigma: np.ndarray  # (20,), float32 single final frame

    def to_torch(self):
        return HandEmgTorchRecording(
            couples=[
                HandEmgTorchTuple(
                    frame=torch.tensor(c.frame, dtype=torch.float32),
                    emg=torch.tensor(c.emg, dtype=torch.float32),
                )
                for c in self.couples
            ],
            sigma=torch.tensor(self.sigma, dtype=torch.float32),
        )


class HandEmgTorchTuple(NamedTuple):
    frame: torch.Tensor  # (20,), float32 expected
    emg: torch.Tensor  # (64, C), float32 expected


class HandEmgTorchRecording(NamedTuple):
    couples: List[HandEmgTorchTuple]
    sigma: torch.Tensor  # (20,), float32 single final frame


def load_recordings(filepath: str) -> List[HandEmgRecording]:
    """
    Load recordings from a ZIP archive written by RecordingsWriter.

    The number of samples (T) is deduced from the file length.

    Args:
        filepath: Path to the ZIP archive.
    """
    print("Loading recordings from", filepath)
    recordings = []
    with zipfile.ZipFile(filepath, mode="r") as archive:
        # Read the metadata file to get the number of EMG channels (C).
        metadata = yaml.safe_load(archive.read("_metadata.yml"))
        C = metadata["C"]

        # Get the list of file names from the archive and sort them numerically.
        file_names = sorted(
            archive.namelist(),
            key=lambda x: int(x.split(".")[0]) if x.endswith(".rec") else float("inf"),
        )
        for fname in tqdm(file_names):
            if fname == "_metadata.yml":
                continue  # Skip the metadata file

            data = archive.read(fname)
            pos = 0

            # Calculate the size per sample:
            # 20 float32 values for frame (20*4 bytes) and 64 * C float32 values for emg.
            sample_size = 20 * 4 + 64 * C * 4

            # Deduce the number of samples T.
            T = (
                len(data) - 20 * 4
            ) // sample_size  # Subtract the size of the final sigma frame

            rec = []
            for _ in range(T):
                frame_bytes = data[pos : pos + 20 * 4]
                frame = np.frombuffer(frame_bytes, dtype=np.float32)
                pos += 20 * 4

                emg_bytes = data[pos : pos + 64 * C * 4]
                emg = np.frombuffer(emg_bytes, dtype=np.float32).reshape((64, C))
                pos += 64 * C * 4

                rec.append(HandEmgTuple(frame=frame, emg=emg))

            # Read the final sigma frame
            sigma_bytes = data[pos : pos + 20 * 4]
            sigma = np.frombuffer(sigma_bytes, dtype=np.float32)

            # Append the complete recording to the list
            recordings.append(HandEmgRecording(couples=rec, sigma=sigma))

    return recordings


def resample(array: np.ndarray, native_fs: int, target_fs: int) -> np.ndarray:
    # Create a time array for the original series
    t_native = np.arange(array.shape[0]) / native_fs

    # Calculate the number of samples in the resampled series
    num_samples = int(array.shape[0] * target_fs / native_fs)

    # Create a time array for the resampled series
    t_target = np.linspace(0, t_native[-1], num_samples)

    # Interpolate the original series at the resampled time points
    f = interp1d(
        t_native, array, axis=0, kind="linear", fill_value=np.nan, bounds_error=False
    )
    return f(t_target)


class RecordingSlicing(Dataset):
    def __init__(
        self,
        recording: HandEmgRecording,
        frames_per_item: int,
    ):
        self.frames_per_item = frames_per_item
        self.recording = recording.to_torch()

    def __len__(self):
        res = len(self.recording.couples) - self.frames_per_item + 1
        return res if res > 0 else 0

    def __getitem__(self, idx):
        u = idx + self.frames_per_item

        s = self.recording.couples[idx:u]

        if u == len(self.recording.couples):
            f = self.recording.sigma
        else:
            f = self.recording.couples[u].frame

        return {
            "emg": torch.concat([e.emg for e in s]),
            "poses": torch.stack([e.frame for e in s] + [f]),
        }


class ConcatSamplerPerDataset(Sampler):
    """Samples elements from each dataset in a ConcatDataset separately"""

    def __init__(self, concat_dataset, sample_ratio=0.2, seed=None):
        self.datasets = concat_dataset.datasets
        self.sample_ratio = sample_ratio
        self.seed = seed

        # Calculate starting indices for each dataset in the concatenated dataset
        self.cumulative_sizes = concat_dataset.cumulative_sizes
        self.offsets = [0] + self.cumulative_sizes[:-1]

    def __iter__(self):
        # Use a separate generator for each epoch
        g = torch.Generator()
        if self.seed is not None:
            g.manual_seed(self.seed)

        indices = []
        # Sample from each dataset separately
        for dataset_idx, dataset in enumerate(self.datasets):
            size = len(dataset)
            n_samples = math.ceil(size * self.sample_ratio)
            # Generate random indices for this dataset
            dataset_indices = torch.randperm(size, generator=g)[:n_samples]
            # Add offset to map to concatenated dataset indices
            indices.extend(dataset_indices.add(self.offsets[dataset_idx]).tolist())

        # Shuffle the combined indices
        combined = torch.tensor(indices)[torch.randperm(len(indices), generator=g)]
        return iter(combined.tolist())

    def __len__(self):
        return sum(
            math.ceil(len(dataset) * self.sample_ratio) for dataset in self.datasets
        )


class DataModule(LightningDataModule):
    def __init__(
        self,
        path: str,  # path to the zip file
        frames_per_item: int,
        emg_samples_per_frame: int = 64,  # frames will be resampled if differs from 64
        batch_size: int = 64,
        sample_ratio: float = 0.8,
    ):
        super().__init__()
        self.path = path
        self.frames_per_item = frames_per_item
        self.emg_samples_per_frame = emg_samples_per_frame
        self.batch_size = batch_size
        self.sample_ratio = sample_ratio

        assert emg_samples_per_frame <= 64, "EMG samples per frame must be <= 64"
        assert (
            64 % emg_samples_per_frame == 0
        ), "EMG samples per frame must be a divisor of 64"

    def setup(self, stage=None):
        recordings = load_recordings(self.path)

        print("Resampling recordings...")
        for i in tqdm(range(len(recordings))):
            rec = recordings[i]

            frames = np.stack([sample.frame for sample in rec.couples] + [rec.sigma])
            frames = resample(frames, 1, 64 // self.emg_samples_per_frame)

            emg = np.concatenate([sample.emg for sample in rec.couples])

            new_rec = HandEmgRecording(
                couples=[
                    HandEmgTuple(
                        frame=frames[j],
                        emg=emg[
                            i
                            * self.emg_samples_per_frame : (i + 1)
                            * self.emg_samples_per_frame
                        ],
                    )
                    for j in range(frames.shape[0] - 1)
                ],
                sigma=rec.sigma,
            )

            recordings[i] = new_rec

        # split: train - the first 80% of each record, val - the last 20%
        train_recordings = []
        val_recordings = []
        for rec in recordings:
            split_idx = int(len(rec.couples) * 0.8)
            train_recordings.append(
                HandEmgRecording(
                    couples=rec.couples[:split_idx],
                    sigma=rec.couples[split_idx].frame,
                )
            )
            val_recordings.append(
                HandEmgRecording(
                    couples=rec.couples[split_idx:],
                    sigma=rec.sigma,
                )
            )

        self.train_dataset = ConcatDataset(
            [
                RecordingSlicing(
                    r,
                    frames_per_item=self.frames_per_item,
                )
                for r in train_recordings
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                RecordingSlicing(
                    r,
                    frames_per_item=self.frames_per_item,
                )
                for r in val_recordings
            ]
        )

    def train_dataloader(self):
        dataset = self.train_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=ConcatSamplerPerDataset(
                dataset,
                sample_ratio=self.sample_ratio,
            ),
            num_workers=0,
        )

    def val_dataloader(self):
        dataset = self.val_dataset

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=ConcatSamplerPerDataset(
                dataset,
                sample_ratio=self.sample_ratio,
                seed=42,
            ),
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self):
        return self.val_dataloader()
