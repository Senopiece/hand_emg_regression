import os
import argparse
import signal
import subprocess
import sys
from typing import List
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datetime import timezone, datetime
import torch

from .dataset import DataModule
from .model import EMG_SAMPLES_PER_FRAME, emg_channels, Model


def run_single(
    run_prefix: str,
    model_name: str,
    enable_progress_bar: bool,
    dataset_path: str,
    cont: bool,
):
    global emg_channels
    torch.set_float32_matmul_precision("medium")

    data_module = DataModule(
        path=dataset_path,
        emg_samples_per_frame=EMG_SAMPLES_PER_FRAME,
        frames_per_item=100,
        sample_ratio=0.05,
        batch_size=64,
    )

    emg_channels = data_module.emg_channels

    ckpt_path = f"./checkpoints/{model_name}.ckpt"
    if cont and os.path.exists(ckpt_path):
        print(f"Loading model {model_name} from {ckpt_path}...")
        model = Model.construct_from_checkpoint(model_name, ckpt_path)
    else:
        print(f"Initializing {model_name}...")
        model = Model.construct(model_name)

    version_name = run_prefix + model_name
    trainer = Trainer(
        max_epochs=200,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_progress_bar=enable_progress_bar,
        logger=WandbLogger(
            project="emg-hand-regression",
            version=version_name
            + f"-{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                save_weights_only=True,
                filename=version_name,
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                enable_version_counter=False,
            ),
            # EarlyStopping(
            #     monitor="val_loss",
            #     patience=30,
            #     mode="min",
            # ),
        ],
    )

    trainer.fit(
        model,
        datamodule=data_module,
    )


def run_many(
    run_prefix: str,
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
                *(["-v", run_prefix] if run_prefix != "" else []),
                "-p",
                *(["-n"] if not cont else []),
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
    run_prefix: str,
    model_names: List[str],
    enable_progress_bar: bool,
    dataset_path: str,
    cont: bool,
):
    if len(model_names) == 0:
        print("Empty model names", file=sys.stderr)
        return 1

    if "all" in model_names:
        model_names = list(Model.impls())

    if len(model_names) == 1:
        run_single(run_prefix, model_names[0], enable_progress_bar, dataset_path, cont)
    else:
        run_many(run_prefix, model_names, dataset_path, cont)

    return 0


if __name__ == "__main__":
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
        default="dataset.zip",
        help="Path to the emg2pose directory (can also be set via the DATASET_PATH environment variable)",
    )
    parser.add_argument(
        "--datasets_path",
        "-ds",
        type=str,
        help="Path to the directory containing multiple dataset paths",
    )
    parser.add_argument(
        "--version",
        "-v",
        type=str,
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

    if args.datasets_path:
        if not os.path.isdir(args.datasets_path):
            raise ValueError(
                f"The provided datasets_path '{args.datasets_path}' is not a valid directory."
            )

        dataset_paths = [
            os.path.join(args.datasets_path, f)
            for f in os.listdir(args.datasets_path)
            if os.path.isfile(os.path.join(args.datasets_path, f))
        ]
        processes = []

        def terminate_processes(signum, frame):
            print("\nTerminating all processes...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.wait()
            exit(0)

        signal.signal(signal.SIGINT, terminate_processes)

        for dataset_path in dataset_paths:
            filename = os.path.basename(dataset_path)
            filename_without_extension = os.path.splitext(filename)[0]
            version_postfix = filename_without_extension + "_"
            process = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "emg_hand_tracking.train",
                    "--model",
                    args.model,
                    "--dataset_path",
                    f'"{dataset_path}"',
                    *(
                        ["-v", args.version + version_postfix]
                        if args.version
                        else [version_postfix]
                    ),
                    "-p",
                    *(["-n"] if args.new else []),
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
                        print(f"{p.args[6]} | {output.strip()}")
                    if p.poll() is not None:  # Process has ended
                        processes.remove(p)

                        # Terminate all if any exited with non-zero exit code
                        if p.returncode != 0:
                            print(
                                f"Process for dataset {p.args[6]} exited with error code {p.returncode}."
                            )
                            terminate_processes(None, None)
        except KeyboardInterrupt:
            terminate_processes(None, None)
    else:
        sys.exit(
            main(
                run_prefix=args.version if args.version else "",
                model_names=args.model.split(","),
                enable_progress_bar=not args.disable_progress_bar,
                dataset_path=args.dataset_path,
                cont=not args.new,
            )
        )
