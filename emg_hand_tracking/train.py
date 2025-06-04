import os
import argparse
import sys
import time
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datetime import timezone, datetime
import torch

from .dataset import DataModule
from .model import Model, SubfeatureSettings, SubfeaturesSettings


class ParameterCountLimit(Callback):
    def __init__(self, max_params: int = 5_000_000):
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

    def on_train_epoch_start(self, trainer, pl_module):
        self._start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self._start_time
        if elapsed > self.max_epoch_time:
            print(
                f"\n⚠️  Epoch took {elapsed:.1f}s (limit is {self.max_epoch_time}s). Exiting."
            )
            sys.exit(1)


def main(
    name: str,
    enable_progress_bar: bool,
    dataset_path: str,
    cont: bool,
    fast_dev_run: bool,
    no_emg: bool,
    emg_samples_per_frame: int,
    slices: int,
    patterns: int,
    frames_per_window: int,
    slice_width: int,
    mx_width: int,
    mx_stride: int,
    std_width: int,
    std_stride: int,
    synapse_features: int,
    muscle_features: int,
    predict_hidden_layer_size: int,
    train_frames_per_patch: int,
    val_frames_per_patch: int,
    train_sample_ratio: float,
    val_sample_ratio: float,
    recordings_usage: int,
    val_usage: int,
    val_window: int,
    batch_size: int,
    lr: float,
    l2: float,
    epoch_time_limit: float,
):
    torch.set_float32_matmul_precision("medium")

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

    ckpt_path = f"./checkpoints/{name}.ckpt"
    if cont and os.path.exists(ckpt_path):
        print(f"Loading {ckpt_path}")
        model = Model.load_from_checkpoint(ckpt_path)

        # Override parameters
        model.lr = lr
        model.l2 = l2
    else:
        print(f"Making new {name}")
        model = Model(
            pose_format=data_module.pose_format,
            channels=data_module.emg_channels,
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
            l2=l2,
        )

    logger = None
    if not fast_dev_run:
        logger = WandbLogger(
            project="emg-hand-regression",
            version=name
            + f"-{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
        )

    trainer = Trainer(
        max_epochs=1000,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_progress_bar=enable_progress_bar,
        logger=logger,
        callbacks=[
            ParameterCountLimit(max_params=5_000_000),
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
                patience=100,
                mode="min",
            ),
            EpochTimeLimit(epoch_time_limit),
        ],
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(
        model,
        datamodule=data_module,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EMG-to-Pose model")
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=str,
        default="dataset.zip",
        help="Path to the emg2pose directory (can also be set via the DATASET_PATH environment variable)",
    )
    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default="model",
        help="Version prefix to add into run name",
    )
    parser.add_argument(
        "--new",
        "-n",
        action="store_true",
        help="Forget previous checkpoint and start from scratch",
    )
    parser.add_argument(
        "--disable_progress_bar",
        "-p",
        action="store_true",
        help="Disable the progress bar, is forcedly disabled for running multiple",
    )
    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        help="Run a quick test of the training loop",
    )
    parser.add_argument(
        "--no_emg",
        action="store_true",
        help="Zero out EMG signals (useful for validating hand motion predictability as itself)",
    )
    parser.add_argument(
        "--emg_samples_per_frame",
        type=int,
        default=16,
        help="Number of EMG samples per frame",
    )
    parser.add_argument(
        "--slices",
        type=int,
        default=35,
        help="Number of slices",
    )
    parser.add_argument(
        "--patterns",
        type=int,
        default=64,
        help="Number of patterns",
    )
    parser.add_argument(
        "--frames_per_window",
        type=int,
        default=10,
        help="Number of frames per window",
    )
    parser.add_argument(
        "--slice_width",
        type=int,
        default=88,
        help="Width of each slice",
    )
    parser.add_argument(
        "--mx_width",
        type=int,
        default=12,
        help="Width for mx subfeature",
    )
    parser.add_argument(
        "--mx_stride",
        type=int,
        default=4,
        help="Stride for mx subfeature",
    )
    parser.add_argument(
        "--std_width",
        type=int,
        default=7,
        help="Width for std subfeature",
    )
    parser.add_argument(
        "--std_stride",
        type=int,
        default=1,
        help="Stride for std subfeature",
    )
    parser.add_argument(
        "--synapse_features",
        type=int,
        default=520,
        help="Number of synapse features",
    )
    parser.add_argument(
        "--muscle_features",
        type=int,
        default=128,
        help="Number of muscle features",
    )
    parser.add_argument(
        "--predict_hidden_layer_size",
        type=int,
        default=256,
        help="Size of the hidden layer for prediction",
    )
    parser.add_argument(
        "--train_frames_per_patch",
        type=int,
        default=100,
        help="Number of frames per item",
    )
    parser.add_argument(
        "--val_frames_per_patch",
        type=int,
        default=100,
        help="Number of frames per item",
    )
    parser.add_argument(
        "--train_sample_ratio",
        type=float,
        default=0.1,
        help="Ratio of training samples",
    )
    parser.add_argument(
        "--val_sample_ratio",
        type=float,
        default=0.7,
        help="Ratio of validation samples",
    )
    parser.add_argument(
        "--recordings_usage",
        type=int,
        default=32,
        help="Limit number of recordings to use (in favour of bigger recordings)",
    )
    parser.add_argument(
        "--val_usage",
        type=int,
        default=12,
        help="Limit number of recordings of which tails use for val (in favour of bigger recordings)",
    )
    parser.add_argument(
        "--val_window",
        type=int,
        default=248,
        help="Window size for validation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--epoch_time_limit",
        type=float,
        default=120.0,
        help="Time limit for each epoch in seconds (default: 120) if exceeded, training will stop",
    )

    args = parser.parse_args()

    main(
        name=args.version,
        enable_progress_bar=not args.disable_progress_bar,
        dataset_path=args.dataset_path,
        cont=not args.new,
        fast_dev_run=args.check,
        no_emg=args.no_emg,
        emg_samples_per_frame=args.emg_samples_per_frame,
        slices=args.slices,
        patterns=args.patterns,
        frames_per_window=args.frames_per_window,
        slice_width=args.slice_width,
        mx_width=args.mx_width,
        mx_stride=args.mx_stride,
        std_width=args.std_width,
        std_stride=args.std_stride,
        synapse_features=args.synapse_features,
        muscle_features=args.muscle_features,
        predict_hidden_layer_size=args.predict_hidden_layer_size,
        train_frames_per_patch=args.train_frames_per_patch,
        val_frames_per_patch=args.val_frames_per_patch,
        train_sample_ratio=args.train_sample_ratio,
        val_sample_ratio=args.val_sample_ratio,
        recordings_usage=args.recordings_usage,
        val_usage=args.val_usage,
        val_window=args.val_window,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        epoch_time_limit=args.epoch_time_limit,
    )
