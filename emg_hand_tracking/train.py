import os
import argparse
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from .dataset import DataModule, emg2pose_slices
from .model import Model


def main(dataset_path: str):
    torch.set_float32_matmul_precision("medium")

    emg_samples_per_frame = 32
    frames_per_item = 6

    data_module = DataModule(
        h5_slices=emg2pose_slices(dataset_path),
        emg_samples_per_frame=emg_samples_per_frame,
        frames_per_item=frames_per_item,
        batch_size=64,
    )

    model = Model(
        emg_samples_per_frame=emg_samples_per_frame,
        frames_per_item=frames_per_item,
        channels=16,
    )

    trainer = Trainer(
        max_epochs=1000,
        logger=TensorBoardLogger("logs", name="emg_model"),
    )

    trainer.fit(model, datamodule=data_module)


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
    args = parser.parse_args()

    if args.dataset_path is None:
        raise ValueError(
            "Please provide a dataset path via the --dataset_path argument or set the DATASET_PATH environment variable."
        )

    main(dataset_path=args.dataset_path)
