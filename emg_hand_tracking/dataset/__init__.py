import math
import torch
import warnings
from torch.utils.data import Dataset, Sampler
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from typing import List, NamedTuple
from tqdm import tqdm

from .recordings import (
    HandEmgRecordingSegment,
    get_pose_format,
    inspect_channels,
    load_recordings,
)


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


class _ConcatSamplerPerDataset(Sampler):
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


# The dataset item
class EmgWithHand(NamedTuple):
    emg: torch.Tensor  # ({B}, T*W, C), float32
    poses: torch.Tensor  # ({B}, T+1, 20), float32


class _RecordingSlicing(Dataset):
    def __init__(
        self,
        segment: HandEmgRecordingSegment,
        frames_per_patch: int,
        no_emg: bool,
    ):
        self.frames_per_patch = frames_per_patch

        self.emg_per_frame = segment.couples[0].emg.shape[0]

        self.emg = torch.tensor(segment.emg, dtype=torch.float32)
        if no_emg:
            self.emg = torch.zeros_like(self.emg)

        self.frames = torch.tensor(segment.frames, dtype=torch.float32)

    def __len__(self):
        res = self.frames.shape[0] - self.frames_per_patch
        return res if res > 0 else 0

    def __getitem__(self, idx):
        u = idx + self.frames_per_patch

        return EmgWithHand(
            emg=self.emg[idx * self.emg_per_frame : u * self.emg_per_frame],
            poses=self.frames[idx : u + 1],
        )


class DataModule(LightningDataModule):
    def __init__(
        self,
        path: str,  # path to the zip file
        train_frames_per_patch: int,
        val_frames_per_patch: int,
        no_emg: bool,
        emg_samples_per_frame: (
            None | int
        ) = None,  # frames will be resampled if provided
        batch_size: int = 64,
        recordings_usage: int = 32,  # number of biggest recordings to use, -1 for all
        train_sample_ratio: float = 0.2,
        val_sample_ratio: float = 0.5,
        val_usage: float = 12,  # number of recordings tails of which will be used for validation, -1 for all
        val_window: int = 248,  # in frames
    ):
        super().__init__()
        self.path = path
        self.emg_samples_per_frame = emg_samples_per_frame
        self.train_frames_per_patch = train_frames_per_patch
        self.val_frames_per_patch = val_frames_per_patch
        self.no_emg = no_emg
        self.batch_size = batch_size
        self.train_sample_ratio = train_sample_ratio
        self.val_sample_ratio = val_sample_ratio
        self.val_window = val_window
        self.val_usage = val_usage
        self.recordings_usage = recordings_usage

    def _segments(self, stage=None):
        recordings = load_recordings(self.path, self.emg_samples_per_frame)

        # Sort recordings by their length in descending order
        recordings.sort(
            key=lambda rec: sum(len(segment.couples) for segment in rec), reverse=True
        )

        # Limit number of recordings to use
        recordings = recordings[: self.recordings_usage]
        assert len(recordings) > 0
        if self.val_usage > 0:
            self.val_usage = min(self.val_usage, len(recordings))
        else:
            self.val_usage = len(recordings) - self.val_usage + 1
            assert self.val_usage > 0

        # split: val - the last X frames from some recordings
        train_segments: List[HandEmgRecordingSegment] = []
        val_segments: List[HandEmgRecordingSegment] = []

        for i, rec in enumerate(recordings):
            # Use the tails of longest recordings for validation
            if i < self.val_usage:
                acc_window = 0
                for segment in reversed(rec):
                    acc_window += len(segment.couples)
                    if acc_window <= self.val_window:
                        val_segments.append(segment)
                    elif acc_window - len(segment.couples) >= self.val_window:
                        train_segments.append(segment)
                    else:
                        split_idx = acc_window - self.val_window
                        train_segments.append(
                            HandEmgRecordingSegment(
                                couples=segment.couples[:split_idx],
                                sigma=segment.couples[split_idx].frame,
                            )
                        )
                        val_segments.append(
                            HandEmgRecordingSegment(
                                couples=segment.couples[split_idx:],
                                sigma=segment.sigma,
                            )
                        )
            else:
                # Use the remaining recordings for training
                train_segments.extend(rec)

        return train_segments, val_segments

    @property
    def emg_channels(self):
        # Sneak peek into the dataset
        return inspect_channels(self.path)

    @property
    def pose_format(self):
        # Sneak peek into the dataset
        return get_pose_format(self.path)

    def setup(self, stage=None):
        train_segments, val_segments = self._segments()

        def to_dataset(
            name: str, segments: List[HandEmgRecordingSegment], frames_per_patch
        ):
            slices: List[_RecordingSlicing] = []
            for segment in tqdm(segments, desc=f"Transforming {name} dataset"):
                slices.append(
                    _RecordingSlicing(
                        segment,
                        frames_per_patch=frames_per_patch,
                        no_emg=self.no_emg,
                    )
                )
            return ConcatDataset(slices)

        self.train_dataset = to_dataset(
            "train",
            train_segments,
            self.train_frames_per_patch,
        )
        self.val_dataset = to_dataset(
            "val",
            val_segments,
            self.val_frames_per_patch,
        )

    def train_dataloader(self):
        dataset = self.train_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=_ConcatSamplerPerDataset(
                dataset,
                sample_ratio=self.train_sample_ratio,
            ),
            num_workers=0,
        )

    def val_dataloader(self):
        dataset = self.val_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=_ConcatSamplerPerDataset(
                dataset,
                sample_ratio=self.val_sample_ratio,
            ),
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self):
        return self.val_dataloader()
