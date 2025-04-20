import os
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datetime import timezone, datetime
import torch

from .dataset import DataModule
from .model import Model, SubfeatureSettings, SubfeaturesSettings


def main(
    name: str,
    enable_progress_bar: bool,
    dataset_path: str,
    cont: bool,
    fast_dev_run: bool,
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
    frames_per_item: int,
    train_sample_ratio: float,
    val_sample_ratio: float,
    val_usage: int,
    val_window: int,
    batch_size: int,
    lr: float,
):
    torch.set_float32_matmul_precision("medium")

    data_module = DataModule(
        path=dataset_path,
        emg_samples_per_frame=emg_samples_per_frame,
        frames_per_item=frames_per_item,
        train_sample_ratio=train_sample_ratio,
        val_sample_ratio=val_sample_ratio,
        val_usage=val_usage,
        val_window=val_window,
        batch_size=batch_size,
    )

    ckpt_path = f"./checkpoints/{name}.ckpt"
    if cont and os.path.exists(ckpt_path):
        print(f"Loading {ckpt_path}")
        model = Model.load_from_checkpoint(ckpt_path)
    else:
        print(f"Making new {name}")
        model = Model(
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
        )

    trainer = Trainer(
        max_epochs=200,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_progress_bar=enable_progress_bar,
        logger=WandbLogger(
            project="emg-hand-regression",
            version=name
            + f"-{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
        ),
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
                patience=10,
                mode="min",
            ),
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
        "--emg_samples_per_frame",
        type=int,
        default=16,
        help="Number of EMG samples per frame",
    )
    parser.add_argument(
        "--slices",
        type=int,
        default=32,
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
        default=20,
        help="Number of frames per window",
    )
    parser.add_argument(
        "--slice_width",
        type=int,
        default=34,
        help="Width of each slice",
    )
    parser.add_argument(
        "--mx_width",
        type=int,
        default=7,
        help="Width for mx subfeature",
    )
    parser.add_argument(
        "--mx_stride",
        type=int,
        default=3,
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
        default=3,
        help="Stride for std subfeature",
    )
    parser.add_argument(
        "--synapse_features",
        type=int,
        default=256,
        help="Number of synapse features",
    )
    parser.add_argument(
        "--muscle_features",
        type=int,
        default=64,
        help="Number of muscle features",
    )
    parser.add_argument(
        "--predict_hidden_layer_size",
        type=int,
        default=128,
        help="Size of the hidden layer for prediction",
    )
    parser.add_argument(
        "--frames_per_item",
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
        default=0.5,
        help="Ratio of validation samples",
    )
    parser.add_argument(
        "--val_usage",
        type=int,
        default=10,
        help="Usage of validation data",
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

    args = parser.parse_args()

    main(
        name=args.version,
        enable_progress_bar=not args.disable_progress_bar,
        dataset_path=args.dataset_path,
        cont=not args.new,
        fast_dev_run=args.check,
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
        frames_per_item=args.frames_per_item,
        train_sample_ratio=args.train_sample_ratio,
        val_sample_ratio=args.val_sample_ratio,
        val_usage=args.val_usage,
        val_window=args.val_window,
        batch_size=args.batch_size,
        lr=args.lr,
    )
