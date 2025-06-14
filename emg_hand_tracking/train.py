import os
import random
import sys
import time
from datetime import timezone, datetime
from typing import Any

import torch
import typer
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from emg_hand_tracking.dataset.recordings import F

from .dataset import DataModule, calc_frame_duration
from .model import Model, SubfeatureSettings, SubfeaturesSettings


app = typer.Typer(pretty_exceptions_show_locals=False)


class EpochTimeLimit(Callback):
    def __init__(self, max_epoch_time_minutes: float = 10.0):
        super().__init__()
        self.max_epoch_time = max_epoch_time_minutes
        self._start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._start_time is None:
            return

        elapsed = (time.time() - self._start_time) / 60.0
        if elapsed > self.max_epoch_time:
            print(
                f"\n⚠️  Training time exceeded {self.max_epoch_time}s limit. Stopping training."
            )
            sys.exit(1)


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
    raw_seed: str = typer.Option(
        "rand",
        "--seed",
        "-s",
        help="Random seed for reproducibility, set to 'rand' for random seed",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        "-c",
        help="Run a quick test of the training loop",
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
    context_span: int = typer.Option(
        32,
        "--context_span",
        help="Size of the context window (in ms)",
    ),
    poses_in_context: int = typer.Option(
        3,
        "--poses_in_context",
        help="Number of frames to be utilized for prediction",
    ),
    slice_width: int = typer.Option(
        10,
        "--slice_width",
        help="Width of each slice (in ms)",
    ),
    mx_width_ms: float = typer.Option(
        4,
        "--mx_width",
        help="Width for mx subfeature in ms",
    ),
    mx_stride_ms: float = typer.Option(
        1,
        "--mx_stride",
        help="Stride for mx subfeature in ms",
    ),
    std_width_ms: float = typer.Option(
        4,
        "--std_width",
        help="Width for std subfeature in ms",
    ),
    std_stride_ms: float = typer.Option(
        1,
        "--std_stride",
        help="Stride for std subfeature in ms",
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
    train_length: float = typer.Option(
        8.0,
        "--train_length",
        help="Size of the train subset (in minutes)",
    ),
    train_patches: int = typer.Option(
        64,
        "--train_patches",
        help="Number of patches to sample of the train subset",
    ),
    train_prediction_length: float = typer.Option(
        2.0,
        "--train_prediction_length",
        help="Size of the patch part to to predict in train (in seconds)",
    ),
    train_sample_ratio: float = typer.Option(
        0.1,
        "--train_sample_ratio",
        help="Ratio of train patches to use in one epoch",
    ),
    val_length: float = typer.Option(
        0.4,
        "--val_length",
        help="Size of the val subset (in minutes)",
    ),
    val_patches: int = typer.Option(
        12,
        "--val_patches",
        help="Number of patches to sample of the val subset",
    ),
    val_prediction_length: float = typer.Option(
        2.0,
        "--val_prediction_length",
        help="Size of the patch part to to predict in val (in seconds)",
    ),
    val_sample_ratio: float = typer.Option(
        0.1,
        "--val_sample_ratio",
        help="Ratio of validation patches to use in one epoch",
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
    slmerr_k: float = typer.Option(
        1.0,
        "--slmerr_k",
        help="Scale for landmark MSE component of loss",
    ),
    vel_k: float = typer.Option(
        1.0,
        "--vel_k",
        help="Scale for velocity component of loss",
    ),
    accel_k: float = typer.Option(
        1.0,
        "--accel_k",
        help="Scale for acceleration component of loss",
    ),
    epoch_time_limit: float = typer.Option(
        10.0,
        "--epoch_time_limit",
        help="Time limit for each epoch in minutes. If exceeded, training will stop",
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

    # Set PyTorch precision
    torch.set_float32_matmul_precision("medium")

    # Setup logger if not in fast_dev_run
    logger = None
    if not fast_dev_run:
        logger = WandbLogger(
            project="emg-hand-regression",
            version=name
            + f"-{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
            allow_val_change=True,
        )
        logger.experiment  # init run
        wandb.config.__dict__["_locked"] = {}  # `delete lock on sweep parameters

    def update_wandb_config(key: str, value: Any):
        if logger is not None:
            wandb.config.update(
                {key: value},
                allow_val_change=True,
            )

    # Seed all
    if raw_seed.isdecimal():
        seed = int(raw_seed)
    elif raw_seed == "rand":
        seed = random.randint(0, 2**31 - 1)
    else:
        raise ValueError(
            f"Invalid seed value: {raw_seed}. It should be a number or 'rand'."
        )

    seed_everything(seed, workers=True)
    update_wandb_config("seed", seed)

    # Resolve dataset
    randdatapath = "/rand:"
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

        update_wandb_config("dataset_path", dataset_path)

    # Initialize data module
    data_module = DataModule(
        path=dataset_path,
        emg_samples_per_frame=emg_samples_per_frame,
        no_emg=no_emg,
        batch_size=batch_size,
        train_length=train_length,
        train_patches=train_patches,
        train_prediction_length=train_prediction_length,
        train_sample_ratio=train_sample_ratio,
        context_span=context_span,
        val_length=val_length,
        val_patches=val_patches,
        val_prediction_length=val_prediction_length,
        val_sample_ratio=val_sample_ratio,
    )

    # Load or create model
    ckpt_path = f"./checkpoints/{name}.ckpt"
    if cont and os.path.exists(ckpt_path):
        print(f"Loading {ckpt_path}")
        model = Model.load_from_checkpoint(ckpt_path)

        # Override learning hyperparameters
        model.set_pose_format(
            data_module.pose_format,
        )
        model.lr = lr
        model.l1 = l1
        model.l2 = l2
        model.slmerr_k = slmerr_k
        model.vel_k = vel_k
        model.accel_k = accel_k

    else:
        print(f"Making new {name}")

        emg_per_sec = F
        emg_per_ms = emg_per_sec * 0.001

        # convert ms to emg
        slice_emg_width = int(slice_width * emg_per_ms)
        mx_width = int(mx_width_ms * emg_per_ms)
        mx_stride = int(mx_stride_ms * emg_per_ms)
        std_width = int(std_width_ms * emg_per_ms)
        std_stride = int(std_stride_ms * emg_per_ms)

        model = Model(
            # Architecture hyperparameters
            channels=data_module.emg_channels,
            emg_samples_per_frame=emg_samples_per_frame,
            slices=slices,
            patterns=patterns,
            poses_in_context=poses_in_context,
            frames_per_window=data_module.context_span_frames,
            slice_emg_width=slice_emg_width,
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
            #
            # Learning hyperparameters
            pose_format=data_module.pose_format,
            lr=lr,
            l1=l1,
            l2=l2,
            slmerr_k=slmerr_k,
            vel_k=vel_k,
            accel_k=accel_k,
        )

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())

    max_params = 3_000_000
    if total_params > max_params:
        print(
            f"\n❌  Model has {total_params:,} parameters (limit is {max_params}). Exiting."
        )
        return

    if logger is not None:
        wandb.log(
            {
                "model/parameters": total_params,
                "model/size_mb": total_size / 1024 / 1024,
            }
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
        deterministic=seed is not None,
    )

    # Start training
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    app()
