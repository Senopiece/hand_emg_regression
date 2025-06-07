import os
import random
import sys
import time
from datetime import timezone, datetime

import torch
import typer
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from .dataset import DataModule
from .model import Model, SubfeatureSettings, SubfeaturesSettings

app = typer.Typer()


class ParameterCountLimit(Callback):
    def __init__(self, max_params: int = 2_000_000):
        super().__init__()
        self.max_params = max_params

    def on_fit_start(self, trainer, pl_module):
        total = sum(p.numel() for p in pl_module.parameters())
        if total > self.max_params:
            print(
                f"\n❌  Model has {total:,} parameters (limit is {self.max_params:,}). Exiting."
            )
            sys.exit(1)


class EpochTimeLimit(Callback):
    def __init__(self, max_epoch_time_seconds: float = 120.0):
        super().__init__()
        self.max_epoch_time = max_epoch_time_seconds
        self._start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._start_time is None:
            return

        elapsed = time.time() - self._start_time
        if elapsed > self.max_epoch_time:
            print(
                f"\n⚠️  Training time exceeded {self.max_epoch_time}s limit. Stopping training."
            )
            sys.exit(1)


def parse_slice_width_percentage_value(value: str, slice_width: int) -> int:
    try:
        # Try to convert the value to a float
        float_value = float(value)

        # Check if it's a percentage (0-1 range)
        if 0 <= float_value <= 1:
            return int(float_value * slice_width)
        else:
            # Treat as an absolute integer value
            return int(float_value)
    except ValueError:
        # Handle the case where the value is a string ending with '%'
        if value.endswith("%"):
            percentage = float(value[:-1])
            return int((percentage / 100) * slice_width)
        else:
            # Directly convert to integer if no special handling is needed
            return int(value)


@app.command()
def main(
    dataset_path: str = typer.Option(
        "dataset.zip",
        "--dataset_path",
        "-d",
        help="Path to the emg2pose directory",
    ),
    version: str = typer.Option(
        "model",
        "--version",
        "-v",
        help="Version prefix to add into run name",
    ),
    new: bool = typer.Option(
        False,
        "--new",
        "-n",
        help="Forget previous checkpoint and start from scratch",
    ),
    disable_progress_bar: bool = typer.Option(
        False,
        "--disable_progress_bar",
        "-p",
        help="Disable the progress bar (forcedly disabled for running multiple)",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        "-c",
        help="Run a quick test of the training loop",
    ),
    emg_featurizer: str = typer.Option(
        "sts",
        "--emg_featurizer",
        help="Use cnn/sts for feature extraction",
    ),
    no_emg: bool = typer.Option(
        False,
        "--no_emg",
        help="Zero out EMG signals (useful for validating hand motion predictability as itself)",
    ),
    emg_samples_per_frame: int = typer.Option(
        16,
        "--emg_samples_per_frame",
        help="Number of EMG samples per frame",
    ),
    slices: int = typer.Option(
        35,
        "--slices",
        help="Number of slices",
    ),
    patterns: int = typer.Option(
        64,
        "--patterns",
        help="Number of patterns",
    ),
    frames_per_window: int = typer.Option(
        10,
        "--frames_per_window",
        help="Number of frames per window",
    ),
    slice_width: int = typer.Option(
        88,
        "--slice_width",
        help="Width of each slice",
    ),
    mx_width_raw: str = typer.Option(
        "12",
        "--mx_width",
        help="Width for mx subfeature (can be an integer or a percentage of slice_width, e.g., '50%')",
    ),
    mx_stride_raw: str = typer.Option(
        "4",
        "--mx_stride",
        help="Stride for mx subfeature (can be an integer or a percentage of slice_width, e.g., '50%')",
    ),
    std_width_raw: str = typer.Option(
        "7",
        "--std_width",
        help="Width for std subfeature (can be an integer or a percentage of slice_width, e.g., '50%')",
    ),
    std_stride_raw: str = typer.Option(
        "1",
        "--std_stride",
        help="Stride for std subfeature (can be an integer or a percentage of slice_width, e.g., '50%')",
    ),
    synapse_features: int = typer.Option(
        520,
        "--synapse_features",
        help="Number of synapse features",
    ),
    muscle_features: int = typer.Option(
        128,
        "--muscle_features",
        help="Number of muscle features",
    ),
    predict_hidden_layer_size: int = typer.Option(
        256,
        "--predict_hidden_layer_size",
        help="Size of the hidden layer for prediction",
    ),
    train_frames_per_patch: int = typer.Option(
        100,
        "--train_frames_per_patch",
        help="Number of frames per item for training",
    ),
    val_frames_per_patch: int = typer.Option(
        100,
        "--val_frames_per_patch",
        help="Number of frames per item for validation",
    ),
    train_sample_ratio: float = typer.Option(
        0.1,
        "--train_sample_ratio",
        help="Ratio of training samples",
    ),
    val_sample_ratio: float = typer.Option(
        0.7,
        "--val_sample_ratio",
        help="Ratio of validation samples",
    ),
    recordings_usage: int = typer.Option(
        32,  # TODO: also allow to provide in %
        "--recordings_usage",
        help="Limit number of recordings to use (in favour of bigger recordings), -1 for all",
    ),
    val_usage: int = typer.Option(
        12,  # TODO: also allow to provide in % of recordings
        "--val_usage",
        help="Limit number of recordings of which tails use for val (in favour of bigger recordings), -1 for all",
    ),
    val_window: int = typer.Option(
        248,
        "--val_window",
        help="Window size for validation",
    ),
    batch_size: int = typer.Option(
        128,
        "--batch_size",
        help="Batch size",
    ),
    lr: float = typer.Option(
        1e-3,
        "--lr",
        help="Learning rate",
    ),
    l1: float = typer.Option(
        1e-4,
        "--l1",
        help="L1 factor",
    ),
    l2: float = typer.Option(
        1e-4,
        "--l2",
        help="Weight decay",
    ),
    epoch_time_limit: float = typer.Option(
        120.0,
        "--epoch_time_limit",
        help="Time limit for each epoch in seconds (default: 120). If exceeded, training will stop",
    ),
    patience: int = typer.Option(
        100,
        "--patience",
        help="Patience before termination",
    ),
):
    # Interpret boolean flags
    cont = not new
    enable_progress_bar = not disable_progress_bar
    fast_dev_run = check
    name = version  # match previous behavior

    # Parse the width and stride values
    mx_width = parse_slice_width_percentage_value(mx_width_raw, slice_width)
    mx_stride = parse_slice_width_percentage_value(mx_stride_raw, slice_width)
    std_width = parse_slice_width_percentage_value(std_width_raw, slice_width)
    std_stride = parse_slice_width_percentage_value(std_stride_raw, slice_width)

    # Set PyTorch precision
    torch.set_float32_matmul_precision("medium")

    # Resolve dataset
    randdatapath = "$rand:"
    if dataset_path.startswith(randdatapath):
        dataset_path = dataset_path[len(randdatapath) :]

        # List all items in the directory
        files = os.listdir(dataset_path)

        # Filter out directories, keep only files
        files = [f for f in files if os.path.isfile(os.path.join(dataset_path, f))]

        if files:
            dataset_path = os.path.join(dataset_path, random.choice(files))
            print(f"Peeking a random dataset: {dataset_path}")
        else:
            raise ValueError("No files found in the dataset directory.")

    # Initialize data module
    data_module = DataModule(
        path=dataset_path,
        emg_samples_per_frame=emg_samples_per_frame,
        train_frames_per_patch=train_frames_per_patch,
        val_frames_per_patch=val_frames_per_patch,
        no_emg=no_emg,
        train_sample_ratio=train_sample_ratio,
        val_sample_ratio=val_sample_ratio,
        recordings_usage=recordings_usage,
        val_usage=val_usage,
        val_window=val_window,
        batch_size=batch_size,
    )

    # Load or create model
    ckpt_path = f"./checkpoints/{name}.ckpt"
    if cont and os.path.exists(ckpt_path):
        print(f"Loading {ckpt_path}")
        model = Model.load_from_checkpoint(ckpt_path)
        # Override parameters
        model.lr = lr
        model.l1 = l1
        model.l2 = l2
    else:
        print(f"Making new {name}")
        model = Model(
            pose_format=data_module.pose_format,
            channels=data_module.emg_channels,
            emg_featurizer=emg_featurizer,
            emg_samples_per_frame=emg_samples_per_frame,
            slices=slices,
            patterns=patterns,
            frames_per_window=frames_per_window,
            slice_width=slice_width,
            subfeatures=SubfeaturesSettings(
                mx=SubfeatureSettings(
                    width=mx_width,
                    stride=mx_stride,
                ),
                std=SubfeatureSettings(
                    width=std_width,
                    stride=std_stride,
                ),
            ),
            synapse_features=synapse_features,
            muscle_features=muscle_features,
            predict_hidden_layer_size=predict_hidden_layer_size,
            lr=lr,
            l1=l1,
            l2=l2,
        )

    # Setup logger if not in fast_dev_run
    logger = None
    if not fast_dev_run:
        logger = WandbLogger(
            project="emg-hand-regression",
            version=name
            + f"-{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
        )

    # Initialize trainer with callbacks
    trainer = Trainer(
        min_epochs=70,
        max_epochs=1000,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_progress_bar=enable_progress_bar,
        logger=logger,
        callbacks=[
            ParameterCountLimit(max_params=2_000_000),
            ModelCheckpoint(
                dirpath="checkpoints",
                save_weights_only=True,
                filename=name,
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                enable_version_counter=False,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                mode="min",
            ),
            EpochTimeLimit(epoch_time_limit),
        ],
        fast_dev_run=fast_dev_run,
    )

    # Start training
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    app()
