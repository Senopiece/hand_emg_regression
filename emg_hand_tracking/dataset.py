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

W = 64


class HandEmgTuple(NamedTuple):
    frame: np.ndarray  # (20,), float32 expected
    emg: np.ndarray  # (W, C), float32 expected


class HandEmgRecordingSegment(NamedTuple):
    couples: List[HandEmgTuple]
    sigma: np.ndarray  # (20,), float32 single final frame

    def to_torch(self):
        return HandEmgTorchRecordingSegment(
            couples=[
                HandEmgTorchTuple(
                    frame=torch.tensor(c.frame, dtype=torch.float32),
                    emg=torch.tensor(c.emg, dtype=torch.float32),
                )
                for c in self.couples
            ],
            sigma=torch.tensor(self.sigma, dtype=torch.float32),
        )

    @property
    def emg(self):
        return np.concatenate([s.emg for s in self.couples])

    @property
    def frames(self):
        return np.stack([s.frame for s in self.couples] + [self.sigma])


HandEmgRecording = List[HandEmgRecordingSegment]


class HandEmgTorchTuple(NamedTuple):
    frame: torch.Tensor  # (20,), float32 expected
    emg: torch.Tensor  # (W, C), float32 expected


class HandEmgTorchRecordingSegment(NamedTuple):
    couples: List[HandEmgTorchTuple]
    sigma: torch.Tensor  # (20,), float32 single final frame

    @property
    def emg(self):
        return torch.concat([s.emg for s in self.couples])

    @property
    def frames(self):
        return torch.stack([s.frame for s in self.couples] + [self.sigma])


HandEmgTorchRecording = List[HandEmgTorchRecordingSegment]


def resample(array: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resample a signal array to a target size using linear interpolation

    Args:
        array (np.ndarray): Input signal array of shape (n_samples, ...)
        target_size (int): Desired number of samples in output

    Returns:
        np.ndarray: Resampled array of shape (target_size, ...)
    """
    # Generate x coordinates for original and target
    x = np.linspace(0, 1, array.shape[0])
    x_new = np.linspace(0, 1, target_size)

    # Preserve shape of additional dimensions
    orig_shape = array.shape
    n_channels = np.prod(orig_shape[1:]) if len(orig_shape) > 1 else 1

    # Reshape to 2D array (samples, channels)
    y = array.reshape(-1, n_channels)

    # Create interpolation function
    f = interp1d(x, y, axis=0, kind="linear")

    # Interpolate
    resampled = f(x_new)

    # Reshape back to original dimensions
    if len(orig_shape) > 1:
        resampled = resampled.reshape((target_size,) + orig_shape[1:])

    return resampled


def load_recordings(path: str, emg_samples_per_frame: int = W):
    """
    Load recordings from a zip file with the following structure:
    dataset.zip/
        metadata.yml
        recordings/
          1/
            segments/
              1
              2
          2/
            segments/
              1
              2

    Then resampling it if needed.
    """
    assert emg_samples_per_frame <= W, "EMG samples per frame must be <= W"
    assert (
        W % emg_samples_per_frame == 0
    ), "EMG samples per frame must be a divisor of W"

    recordings: List[HandEmgRecording] = []

    with zipfile.ZipFile(path, mode="r") as archive:
        # Read the metadata file to get the number of EMG channels (C)
        metadata = yaml.safe_load(archive.read("metadata.yml"))
        C = metadata["C"]

        # Get all recording directories
        all_files = archive.namelist()
        recording_dirs = set(
            path.split("/")[1]
            for path in all_files
            if path.startswith("recordings/") and len(path.split("/")) > 2
        )

        # Process each recording
        for rec_dir in tqdm(
            sorted(recording_dirs, key=lambda x: int(x)),
            desc="Loading dataset",
        ):
            recording = []

            # Get segment files for this recording
            segment_files = [
                f
                for f in all_files
                if f.startswith(f"recordings/{rec_dir}/segments/")
                and f.split("/")[-1].isdigit()
            ]

            # Process each segment
            for seg_file in sorted(segment_files, key=lambda x: int(x.split("/")[-1])):
                data = archive.read(seg_file)
                pos = 0

                # Calculate size per sample
                sample_size = 20 * 4 + W * C * 4  # frame (20*4) + emg (W*C*4)

                # Calculate number of samples
                T = (len(data) - 20 * 4) // sample_size  # Subtract final sigma frame

                # Read all frame-EMG pairs
                segment_data = []
                for _ in range(T):
                    # Read frame
                    frame_bytes = data[pos : pos + 20 * 4]
                    frame = np.frombuffer(frame_bytes, dtype=np.float32)
                    pos += 20 * 4

                    # Read EMG
                    emg_bytes = data[pos : pos + W * C * 4]
                    emg = np.frombuffer(emg_bytes, dtype=np.float32).reshape((W, C))
                    pos += W * C * 4

                    segment_data.append(HandEmgTuple(frame=frame, emg=emg))

                # Read final sigma frame
                sigma_bytes = data[pos : pos + 20 * 4]
                sigma = np.frombuffer(sigma_bytes, dtype=np.float32)

                # Add segment to recording
                recording.append(
                    HandEmgRecordingSegment(couples=segment_data, sigma=sigma)
                )

            recordings.append(recording)

    # Handle resampling if needed
    if emg_samples_per_frame != W:
        for i in tqdm(range(len(recordings)), desc="Upsampling segments"):
            for j in range(len(recordings[i])):
                rec = recordings[i][j]

                frames = resample(
                    rec.frames,
                    (W // emg_samples_per_frame) * len(rec.couples) + 1,
                )

                emg = rec.emg

                recordings[i][j] = HandEmgRecordingSegment(
                    couples=[
                        HandEmgTuple(
                            frame=frames[k],
                            emg=emg[
                                k
                                * emg_samples_per_frame : (k + 1)
                                * emg_samples_per_frame
                            ],
                        )
                        for k in range(frames.shape[0] - 1)
                    ],
                    sigma=rec.sigma,
                )

    return recordings


class RecordingSlicing(Dataset):
    def __init__(
        self,
        segment: HandEmgTorchRecordingSegment,  # type: ignore
        frames_per_item: int,
    ):
        self.frames_per_item = frames_per_item

        self.emg_per_frame = segment.couples[0].emg.shape[0]
        self.emg = segment.emg
        self.frames = segment.frames

    def __len__(self):
        res = (self.emg.shape[0] // self.emg_per_frame) - self.frames_per_item + 1
        return res if res > 0 else 0

    def __getitem__(self, idx):
        u = idx + self.frames_per_item

        return {
            "emg": self.emg[idx * self.emg_per_frame : u * self.emg_per_frame],
            "poses": self.frames[idx : u + 1],
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
        emg_samples_per_frame: int = W,  # frames will be resampled if differs from W
        batch_size: int = 64,
        train_sample_ratio: float = 0.2,
        val_sample_ratio: float = 0.5,
        val_percent: float = 1,  # percentage of recordings tails of which will be used for validation
        val_window: int = 248,  # in frames
    ):
        super().__init__()
        self.path = path
        self.emg_samples_per_frame = emg_samples_per_frame
        self.frames_per_item = frames_per_item
        self.batch_size = batch_size
        self.train_sample_ratio = train_sample_ratio
        self.val_sample_ratio = val_sample_ratio
        self.val_window = val_window
        self.val_percent = val_percent

    def _segments(self, stage=None):
        recordings = load_recordings(self.path, self.emg_samples_per_frame)

        # split: val - the last X frames from some recordings
        train_segments: List[HandEmgRecordingSegment] = []
        val_segments: List[HandEmgRecordingSegment] = []

        # filter bad recordings (recording is bad if val window is >25% of it)
        recordings = [
            rec
            for rec in recordings
            if 4 * self.val_window < sum(len(segment.couples) for segment in rec)
        ]

        # some of recordings need to be discarded from filling val to maintain val_percent
        train_idxs: List[int] = torch.randperm(len(recordings))[
            : int((1 - self.val_percent) * len(recordings))
        ].tolist()

        for i, rec in enumerate(recordings):
            # force use for train
            if i in train_idxs:
                train_segments.extend(rec)
                continue

            # use for validation
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

        return train_segments, val_segments

    @property
    def emg_channels(self):
        # Sneak peek into the dataset
        with zipfile.ZipFile(self.path, mode="r") as archive:
            metadata = yaml.safe_load(archive.read("metadata.yml"))
            return metadata["C"]

    def setup(self, stage=None):
        train_segments, val_segments = self._segments()

        def to_dataset(name, segments):
            slices = []
            for segment in tqdm(segments, desc=f"Transforming {name} dataset"):
                slices.append(
                    RecordingSlicing(
                        segment.to_torch(),
                        frames_per_item=self.frames_per_item,
                    )
                )
            return ConcatDataset(slices)

        self.train_dataset = to_dataset("train", train_segments)
        self.val_dataset = to_dataset("val", val_segments)

    def train_dataloader(self):
        dataset = self.train_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=ConcatSamplerPerDataset(
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
            sampler=ConcatSamplerPerDataset(
                dataset,
                sample_ratio=self.val_sample_ratio,
                seed=42,
            ),
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self):
        return self.val_dataloader()
