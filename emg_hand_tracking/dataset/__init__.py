from itertools import chain
import math
from random import randint
import torch
import warnings
from torch.utils.data import Dataset, Sampler
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from typing import List, NamedTuple
from tqdm import tqdm

from .segmention import (
    print_bars,
    simulated_annealing,
    to_segmentation_array,
)

from .recordings import (
    HandEmgRecordingSegment,
    calc_frame_duration,
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


def collect_span(segments: List[HandEmgRecordingSegment], length: int):
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


def segmentate(label, segments, segmentation):
    remaining = segmentation - len(segments) - 1
    if remaining < 0:
        raise ValueError(
            f"{label} Oversegmentated, need {segmentation}, got {len(segments) - 1}"
        )

    arr = to_segmentation_array(segments)
    print_bars(f"{label} Init Segmentation:", arr)

    arr, _ = simulated_annealing(arr, remaining)
    arr = [int(e) for e in arr]

    sizes_arr = [b - a for a, b in zip(arr[:-1], arr[1:])]

    res = []
    for size in sizes_arr:
        segs, segments = collect_span(segments, size)
        res.extend(segs)

    print_bars(f"{label} Comp Segmentation:", to_segmentation_array(res))

    return res


class _ConcatSamplerPerDataset(Sampler):
    """Samples elements from each dataset in a ConcatDataset separately"""

    def __init__(self, concat_dataset, sample_ratio=0.2):
        self.datasets = concat_dataset.datasets
        self.sample_ratio = sample_ratio

        # Calculate starting indices for each dataset in the concatenated dataset
        self.cumulative_sizes = concat_dataset.cumulative_sizes
        self.offsets = [0] + self.cumulative_sizes[:-1]

    def __iter__(self):
        indices = []
        # Sample from each dataset separately
        for dataset_idx, dataset in enumerate(self.datasets):
            size = len(dataset)
            n_samples = math.ceil(size * self.sample_ratio)
            # Generate random indices for this dataset
            dataset_indices = torch.randperm(size)[:n_samples]
            # Add offset to map to concatenated dataset indices
            indices.extend(dataset_indices.add(self.offsets[dataset_idx]).tolist())

        # Shuffle the combined indices
        combined = torch.tensor(indices)[torch.randperm(len(indices))]
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
        train_split_length: float = 24.0,  # in minutes
        train_segmentation: int = 4,
        train_patch_length: float = 2.0,  # in seconds
        train_sample_ratio: float = 0.3,
        context_span: int = 100,  # duration in ms of which poses to provide the model with as initial poses
        val_split_length: float = 24.0,  # in minutes
        val_segmentation: int = 4,
        val_patch_length: float = 2.0,  # in seconds
        val_sample_ratio: float = 0.3,
    ):
        super().__init__()

        self.path = path

        self.emg_samples_per_frame = emg_samples_per_frame
        self.no_emg = no_emg
        self.batch_size = batch_size

        self.train_split_length = train_split_length
        self.train_segmentation = train_segmentation
        self.train_patch_length = train_patch_length
        self.train_sample_ratio = train_sample_ratio
        self.val_split_length = val_split_length
        self.val_segmentation = val_segmentation
        self.val_patch_length = val_patch_length
        self.val_sample_ratio = val_sample_ratio

        self.couple_duration = 1 / calc_frame_duration(emg_samples_per_frame)
        self.frames_per_ms = self.couple_duration * 0.001

        self.context_span_frames = int(context_span * self.frames_per_ms)

    def _segments(self, stage=None):
        recordings = load_recordings(self.path, self.emg_samples_per_frame)

        # Treat dataset as single recording with many segments
        segments = list(chain(*recordings))

        # Compute patch sizes
        train_frames_per_patch = int(self.train_patch_length / self.couple_duration)
        val_frames_per_patch = int(self.val_patch_length / self.couple_duration)

        # Tweak val patch len so the ms to predict in not decreased with increased context span
        val_frames_per_patch += self.context_span_frames

        # Treat dataset as single recording with many segments
        segments = list(chain(*recordings))

        couples_per_minute = 60 / self.couple_duration
        train_split_size_in_frames = int(self.train_split_length * couples_per_minute)
        val_split_size_in_frames = int(self.val_split_length * couples_per_minute)

        # Tweak val split len so that after adjusting the ms to predict in not decreased with increased context span, the number of samples is not decreased
        # NOTE:
        # it's all inspired by the equation : number_of_patches = total_len - number_of_segments*(patch_len - 1)
        # so that we have:
        # p1 = L1 - n(w - 1) : number of patches without accounting to context span = val_split_size_in_frames - segments*(val_frames_per_patch - 1)
        # w2 = w + c : but we need to increase the patch size so that the frames a model is needed to predict is not decreased with increased context window
        # p2 = L2 - n(w2 - 1) : number of patches by accounting to context span = val_split_size_in_frames2 - segments*(val_frames_per_patch + context_span - 1)
        # p1 = p2 : as we also need the number of samples to remain the same no matter what context span there is
        # solving, we get L2 = L1 + nc, so that after increasing the patch size to account for context span, we need to also increase the overall validation span to hold the number of patches the same
        val_split_size_in_frames += self.val_segmentation * self.context_span_frames

        # Random offset
        couples_count = sum(len(seg.couples) for seg in segments)
        offset_range = (
            couples_count - val_split_size_in_frames - train_split_size_in_frames
        )

        if offset_range < 0:
            raise ValueError("Insufficient dataset length")

        offset = randint(0, offset_range - 1)

        # Skip offset
        _, segments = collect_span(segments, offset)

        # Collect train
        train_segments, segments = collect_span(segments, train_split_size_in_frames)
        train_segments = segmentate("Train", train_segments, self.train_segmentation)

        # Collect val
        val_segments, _ = collect_span(segments, val_split_size_in_frames)
        val_segments = segmentate("Val", val_segments, self.val_segmentation)

        return (
            train_frames_per_patch,
            val_frames_per_patch,
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
        train_frames_per_patch, val_frames_per_patch, train_segments, val_segments = (
            self._segments()
        )

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
            d = ConcatDataset(slices)

            # check dataset length holds
            assert len(d) == sum(len(s.couples) for s in segments) - len(segments) * (
                frames_per_patch - 1
            )

            return d

        self.train_dataset = to_dataset(
            "train",
            train_segments,
            train_frames_per_patch,
        )
        self.val_dataset = to_dataset(
            "val",
            val_segments,
            val_frames_per_patch,
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
