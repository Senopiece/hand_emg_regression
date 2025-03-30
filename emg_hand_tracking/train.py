import os
import argparse
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch

from .dataset import DataModule, emg2pose_slices
from .model import Model


def main(dataset_path: str, checkpoint: str | None = None):
    torch.set_float32_matmul_precision("medium")

    print("Loading model...")
    model = Model(
        emg_samples_per_frame=32,
        frames_per_window=6,
        channels=16,
    )

    print("Loading dataset...")
    data_module = DataModule(
        h5_slices=emg2pose_slices(
            dataset_path,
            train_window=12,
            val_window=10,
            step=model.emg_window_length,
        ),
        emg_samples_per_frame=model.emg_samples_per_frame,
        batch_size=64,
    )

    print("Preparing trainer...")
    trainer = Trainer(
        max_epochs=1000,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=WandbLogger(
            project="emg-hand-regression",
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                save_top_k=0,
                save_last=True,
                monitor="val_loss",
                mode="min",
            ),
        ],
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=checkpoint,
    )


if __name__ == "__main__":
    load_dotenv()

    env_dataset_path = os.getenv("DATASET_PATH")

    parser = argparse.ArgumentParser(description="Train EMG-to-Pose model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=env_dataset_path,
        help="Path to the emg2pose directory (can also be set via the DATASET_PATH environment variable)",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Path to a checkpoint file to load model weights from, can be last or hpc",
    )
    args = parser.parse_args()

    if args.dataset_path is None:
        raise ValueError(
            "Please provide a dataset path via the --dataset_path argument or set the DATASET_PATH environment variable."
        )

    main(dataset_path=args.dataset_path, checkpoint=args.checkpoint)
