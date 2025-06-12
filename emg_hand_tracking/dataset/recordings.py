import zipfile
import numpy as np
import yaml
from typing import List, NamedTuple
from tqdm import tqdm
from scipy.interpolate import Akima1DInterpolator, interp1d

F = 2048  # assuming 2048 emg sampling frequency
W = 64  # W = EmgSampling/HandSampling


class HandEmgTuple(NamedTuple):
    frame: np.ndarray  # (20,), float32 expected
    emg: np.ndarray  # (W, C), float32 expected


class HandEmgRecordingSegment(NamedTuple):
    couples: List[HandEmgTuple]
    sigma: np.ndarray  # (20,), float32 single final frame

    @property
    def emg(self):
        return np.concatenate([s.emg for s in self.couples])

    @property
    def frames(self):
        return np.stack([s.frame for s in self.couples] + [self.sigma])


HandEmgRecording = List[HandEmgRecordingSegment]


def recording_size(recording: HandEmgRecording) -> int:
    return sum(len(segment.couples) for segment in recording)


def dataset_size(recordings: List[HandEmgRecording]) -> int:
    return sum(recording_size(rec) for rec in recordings)


def _resample_akima(array: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resample a signal array to a target size using Akima interpolation

    Args:
        array (np.ndarray): Input signal array of shape (n_samples, ...)
        target_size (int): Desired number of samples in output

    Returns:
        np.ndarray: Resampled array of shape (target_size, ...)
    """
    x = np.linspace(0, 1, array.shape[0])
    x_new = np.linspace(0, 1, target_size)

    orig_shape = array.shape
    n_channels = np.prod(orig_shape[1:]) if len(orig_shape) > 1 else 1

    y = array.reshape(-1, n_channels)

    resampled = np.zeros((target_size, n_channels))
    for ch in range(n_channels):
        akima = Akima1DInterpolator(x, y[:, ch])
        resampled[:, ch] = akima(x_new)

    if len(orig_shape) > 1:
        resampled = resampled.reshape((target_size,) + orig_shape[1:])

    return resampled


def _resample_linear(array: np.ndarray, target_size: int) -> np.ndarray:
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


RESAMPLE_BY_POSE_FORMAT = {
    "UmeTrack": _resample_linear,
    "AnatomicAngles": _resample_akima,
}


def load_recordings(path: str, emg_samples_per_frame: int | None = None):
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
    if emg_samples_per_frame is None:
        emg_samples_per_frame = W

    assert emg_samples_per_frame <= W, f"EMG samples per frame must be <= {W}"
    assert (
        W % emg_samples_per_frame == 0
    ), "EMG samples per frame must be a divisor of W"

    recordings: List[HandEmgRecording] = []

    with zipfile.ZipFile(path, mode="r") as archive:
        # Read the metadata file to get the number of EMG channels (C)
        metadata = yaml.safe_load(archive.read("metadata.yml"))
        C = metadata["C"]
        pose_format = metadata.get("pose_format", "AnatomicAngles")

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

    resample = RESAMPLE_BY_POSE_FORMAT[pose_format]

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


def calc_frame_duration(emg_samples_per_frame: float | None):
    if emg_samples_per_frame is None:
        emg_samples_per_frame = W
    return emg_samples_per_frame / F  # in seconds


def inspect_channels(path: str) -> int:
    with zipfile.ZipFile(path, mode="r") as archive:
        metadata = yaml.safe_load(archive.read("metadata.yml"))
        channels = metadata["C"]
        if not isinstance(channels, int):
            raise ValueError("Dataset is malformed")
        return channels


def get_pose_format(path: str):
    with zipfile.ZipFile(path, mode="r") as archive:
        return yaml.safe_load(archive.read("metadata.yml")).get(
            "pose_format", "AnatomicAngles"
        )
