from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning import EMGDataModule

from model import EMGToPoseModel
from dataset_slices import dataset_slices


if __name__ == "__main__":
    emg_samples_per_frame = 32
    frames_per_item = 6
    joint_output_size = frames_per_item * 10

    data_module = EMGDataModule(
        h5_slices=dataset_slices,
        emg_samples_per_frame=emg_samples_per_frame,
        frames_per_item=frames_per_item,
        batch_size=64,
    )

    model = EMGToPoseModel(
        emg_samples_per_frame=emg_samples_per_frame,
        frames_per_item=frames_per_item,
        channels=16,
    )

    trainer = Trainer(
        max_epochs=100,
        logger=TensorBoardLogger("logs", name="emg_model"),
    )

    trainer.fit(model, datamodule=data_module)
