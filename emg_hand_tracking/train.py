import os
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datetime import timezone, datetime
import torch

from .dataset import DataModule
from .model import Model


def main(
    name: str,
    enable_progress_bar: bool,
    dataset_path: str,
    cont: bool,
):
    torch.set_float32_matmul_precision("medium")

    emg_samples_per_frame = 16  # 120 preds/sec

    data_module = DataModule(
        path=dataset_path,
        emg_samples_per_frame=emg_samples_per_frame,
        frames_per_item=100,
        train_sample_ratio=0.1,
        val_sample_ratio=0.5,
        val_window=248,
        batch_size=128,
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
    args = parser.parse_args()

    main(
        name=args.version,
        enable_progress_bar=not args.disable_progress_bar,
        dataset_path=args.dataset_path,
        cont=not args.new,
    )
