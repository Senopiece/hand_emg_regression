import os
import argparse
import signal
import subprocess
import sys
from typing import List
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datetime import timezone, datetime
import torch

from .dataset import DataModule, emg2pose_slices
from .model import Model


def run_single(
    model_name: str,
    dataset_path: str,
    cont: bool,
):
    torch.set_float32_matmul_precision("medium")

    ckpt_path = f"./checkpoints/{model_name}.ckpt"
    if cont and os.path.exists(ckpt_path):
        print(f"Loading model {model_name} from {ckpt_path}...")
        model = Model.construct_from_checkpoint(model_name, ckpt_path)
    else:
        print(f"Initializing {model_name}...")
        model = Model.construct(model_name)

    print("Loading dataset...")
    data_module = DataModule(
        h5_slices=emg2pose_slices(
            dataset_path,
            train_window=16,
            val_window=16,
            step=model.emg_window_length,
        ),
        emg_samples_per_frame=model.emg_samples_per_frame,
        batch_size=64,
    )

    print("Preparing trainer...")
    trainer = Trainer(
        max_epochs=5000,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_progress_bar=False,
        logger=WandbLogger(
            project="emg-hand-regression",
            version=model_name
            + "-train16val16"
            + f"-{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                save_weights_only=True,
                filename=model_name,
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


def run_many(
    models: List[str],
    dataset_path: str,
    cont: bool,
):
    processes = []

    def terminate_processes(signum, frame):
        print("\nTerminating all processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        exit(0)

    signal.signal(signal.SIGINT, terminate_processes)

    for model in models:
        print(f"Starting process for model: {model}")
        process = subprocess.Popen(
            [
                "python",
                "-m",
                "emg_hand_tracking.train",
                "--model",
                model,
                "--dataset_path",
                dataset_path,
                *(["-c"] if cont else []),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append(process)

    try:
        while processes:
            for p in processes:
                output = p.stdout.readline()
                if output:
                    print(f"{p.args[4]} | {output.strip()}")
                if p.poll() is not None:  # Process has ended
                    processes.remove(p)

                    # Terminate all if any exited with non-zero exit code
                    if p.returncode != 0:
                        print(
                            f"Process for model {p.args[4]} exited with error code {p.returncode}."
                        )
                        terminate_processes(None, None)
    except KeyboardInterrupt:
        terminate_processes(None, None)


def main(
    model_names: List[str],
    dataset_path: str,
    cont: bool,
):
    if len(model_names) == 0:
        print("Empty model names", file=sys.stderr)
        return 1

    if "all" in model_names:
        model_names = list(Model.impls())

    if len(model_names) == 1:
        run_single(model_names[0], dataset_path, cont)
    else:
        run_many(model_names, dataset_path, cont)

    return 0


if __name__ == "__main__":
    load_dotenv()

    env_dataset_path = os.getenv("DATASET_PATH")

    parser = argparse.ArgumentParser(description="Train EMG-to-Pose model")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="all",
        help="Models to train, separated by comma. Available: all, "
        + ", ".join(Model.impls()),
    )
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=str,
        default=env_dataset_path,
        help="Path to the emg2pose directory (can also be set via the DATASET_PATH environment variable)",
    )
    parser.add_argument(
        "--cont",
        "-c",
        action="store_true",
        help="Continue from the last checkpoint",
    )
    args = parser.parse_args()

    if args.dataset_path is None:
        raise ValueError(
            "Please provide a dataset path via the --dataset_path argument or set the DATASET_PATH environment variable."
        )

    sys.exit(
        main(
            model_names=args.model.split(","),
            dataset_path=args.dataset_path,
            cont=args.cont,
        )
    )
