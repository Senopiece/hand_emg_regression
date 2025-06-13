from itertools import chain
import math
from random import randint
import torch
import warnings
from torch.utils.data import Dataset, RandomSampler, Subset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from typing import List, NamedTuple
from tqdm import tqdm

from .sample import uniform_bounded_sum

from .recordings import (
    HandEmgRecordingSegment,
    calc_frame_duration,
    get_pose_format,
    inspect_channels,
    load_recordings,
    recording_size,
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


def collect_span(segments: List[HandEmgRecordingSegment], length: int):
    if length == 0:
        return [], segments

    pos = 0

    for i, seg in enumerate(segments):
        next_pos = pos + len(seg.couples)

        if next_pos < length:
            pos = next_pos
        elif next_pos == length:
            return segments[: i + 1], segments[i + 1 :]
        else:
            inverse_split_idx = length - next_pos
            return [
                *segments[:i],
                HandEmgRecordingSegment(
                    couples=seg.couples[:inverse_split_idx],
                    sigma=seg.couples[inverse_split_idx].frame,
                ),
            ], [
                HandEmgRecordingSegment(
                    couples=seg.couples[inverse_split_idx:],
                    sigma=seg.sigma,
                ),
                *segments[i + 1 :],
            ]

    # must not be reached because of dataset length check above
    raise ValueError("Ran out of segments")


def take_across(recordings, size, min_length, offset):
    # Trim too short recordings
    recordings = [rec for rec in recordings if recording_size(rec) >= min_length]

    # Collect intervals
    sizes = uniform_bounded_sum(
        size,
        [(min_length, recording_size(r)) for r in recordings],
    )
    collected = []
    remaining = []
    for rec, size in zip(recordings, sizes):
        couples_count = recording_size(rec)
        if offset:
            offset_range = couples_count - size
            offset = randint(0, offset_range - 1)
            assert offset >= 0, "Offset must be non-negative"
        else:
            offset = 0

        _, rec = collect_span(rec, offset)
        segments, rec = collect_span(rec, size)

        collected.extend(segments)
        remaining.append(rec)

    return collected, remaining


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

        if no_emg:
            # single channel zeroed emg
            self.emg = torch.zeros(
                (len(segment.couples) * segment.couples[0].emg.shape[0], 1),
                dtype=torch.float32,
            )
        else:
            self.emg = torch.tensor(segment.emg, dtype=torch.float32)

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
        no_emg: bool,
        emg_samples_per_frame: (
            None | int
        ) = None,  # frames will be resampled if provided
        batch_size: int = 64,
        train_length: float = 24.0,  # in minutes
        train_patches: int = 4,
        train_prediction_length: float = 2.0,  # in seconds
        train_sample_ratio: float = 0.3,
        context_span: int = 100,  # duration in ms of which poses to provide the model with as initial poses
        val_length: float = 24.0,  # in minutes
        val_patches: int = 4,
        val_prediction_length: float = 2.0,  # in seconds
        val_sample_ratio: float = 0.3,
        train_split_threshold: float = 0.2,  # minutes
        val_split_threshold: float = 1.0,  # seconds
        split_type: str = "relative_take",  # "relative_take" or "by_sequence"
    ):
        super().__init__()

        self.path = path

        self.emg_samples_per_frame = emg_samples_per_frame
        self.no_emg = no_emg
        self.batch_size = batch_size

        self.train_length = train_length
        self.train_patches = train_patches
        self.train_prediction_length = train_prediction_length
        self.train_sample_ratio = train_sample_ratio
        self.val_length = val_length
        self.val_patches = val_patches
        self.val_prediction_length = val_prediction_length
        self.val_sample_ratio = val_sample_ratio
        self.split_type = split_type

        self.couple_duration = calc_frame_duration(emg_samples_per_frame)
        self.couples_per_second = 1 / self.couple_duration
        self.couples_per_minute = 60 / self.couple_duration
        self.frames_per_ms = 1 / (1000 * self.couple_duration)

        self.train_split_threshold = int(
            train_split_threshold * self.couples_per_minute
        )
        self.val_split_threshold = int(val_split_threshold * self.couples_per_second)

        self.context_span_frames = int(context_span * self.frames_per_ms)
        self.train_size_in_frames = int(self.train_length * self.couples_per_minute)
        self.val_size_in_frames = int(self.val_length * self.couples_per_minute)

        self.train_frames_per_patch = (
            int(self.train_prediction_length / self.couple_duration)
            + self.context_span_frames
        )
        self.val_frames_per_patch = (
            int(self.val_prediction_length / self.couple_duration)
            + self.context_span_frames
        )

    def _segments_by_sequence(self, stage=None):
        recordings = load_recordings(self.path, self.emg_samples_per_frame)

        # Treat dataset as single recording with many segments
        segments = list(chain(*recordings))

        # Random offset
        couples_count = recording_size(segments)
        offset_range = (
            couples_count - self.val_size_in_frames - self.train_size_in_frames
        )

        if offset_range <= 0:
            raise ValueError("Insufficient dataset length")

        offset = randint(0, offset_range - 1)

        # Skip offset
        _, segments = collect_span(segments, offset)

        # Collect train
        train_segments, segments = collect_span(segments, self.train_size_in_frames)

        # Collect val
        val_segments, _ = collect_span(segments, self.val_size_in_frames)

        return (
            train_segments,
            val_segments,
        )

    def _segments_by_relative_take(self, stage=None):
        # Recordings contribute to val and train according to their length

        recordings = load_recordings(self.path, self.emg_samples_per_frame)

        train_segments, recordings = take_across(
            recordings,
            self.train_size_in_frames,
            self.train_split_threshold,
            offset=True,
        )
        val_segments, recordings = take_across(
            recordings, self.val_size_in_frames, self.val_split_threshold, offset=False
        )

        return (
            train_segments,
            val_segments,
        )

    @property
    def emg_channels(self):
        # Sneak peek into the dataset
        return 1 if self.no_emg else inspect_channels(self.path)

    @property
    def pose_format(self):
        # Sneak peek into the dataset
        return get_pose_format(self.path)

    def setup(self, stage=None):
        if self.split_type == "by_sequence":
            train_segments, val_segments = self._segments_by_sequence()
        elif self.split_type == "relative_take":
            train_segments, val_segments = self._segments_by_relative_take()
        else:
            raise ValueError(
                f"Unknown split type: {self.split_type}. "
                "Use 'by_sequence' or 'relative_take'."
            )

        def to_dataset(
            name: str,
            segments: List[HandEmgRecordingSegment],
            frames_per_patch: int,
            patches: int,
        ):
            slices: List[_RecordingSlicing] = []

            for segment in tqdm(segments, desc=f"Transforming {name} dataset"):
                if len(segment.couples) - frames_per_patch + 1 > 0:
                    slices.append(
                        _RecordingSlicing(
                            segment,
                            frames_per_patch=frames_per_patch,
                            no_emg=self.no_emg,
                        )
                    )

            d = ConcatDataset(slices)

            # subset it
            if len(d) < patches:
                raise ValueError(
                    f"Cannot create subset of length {patches} from dataset of length {len(d)}"
                )
            subset_indices = torch.randperm(len(d))[:patches].tolist()

            return Subset(d, subset_indices)

        self.train_dataset = to_dataset(
            "train",
            train_segments,
            self.train_frames_per_patch,
            self.train_patches,
        )
        self.val_dataset = to_dataset(
            "val",
            val_segments,
            self.val_frames_per_patch,
            self.val_patches,
        )

    def train_dataloader(self):
        dataset = self.train_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(
                dataset,
                num_samples=math.ceil(len(dataset) * self.train_sample_ratio),
            ),
            num_workers=0,
        )

    def val_dataloader(self):
        dataset = self.val_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(
                dataset,
                num_samples=math.ceil(len(dataset) * self.val_sample_ratio),
            ),
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self):
        return self.val_dataloader()
