from emg2pose.kinematics import forward_kinematics, load_default_hand_model
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .util import handmodel2device
from .modules import WindowedApply, WeightedMean


class Model(pl.LightningModule):
    def __init__(
        self,
        channels: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.emg_samples_per_frame = 32
        self.frames_per_window = 6
        self.channels = channels
        self.hm = load_default_hand_model()

        self.emg_window_length = self.emg_samples_per_frame * self.frames_per_window

        # T = total_seq_length + S*emg_samples_per_frame
        # C = channels

        self.conv = nn.Sequential(  # <- (B, C, T)
            nn.Conv1d(
                in_channels=channels,
                out_channels=1024,
                kernel_size=101,
                padding=50,
                bias=False,
            ),  # -> (B, 1024, T)
            WindowedApply(  # separate windows for calculating multiple predictions
                window_len=self.emg_window_length,
                step=self.emg_samples_per_frame,
                f=nn.Sequential(  # <- (B, 1024, total_seq_length)
                    nn.Flatten(),
                    nn.Linear(1024 * self.emg_window_length, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 64),
                ),
            ),  # -> (B, W, 64), S=W
        )

        self.predict = nn.Sequential(
            nn.Linear(self.frames_per_window * 20 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 20),
        )

        self.filter = WeightedMean(self.frames_per_window + 1)

    def _forward(self, emg, initial_poses):
        emg = emg.permute(0, 2, 1)  # (B, T, C)
        windows = self.conv(emg).permute(1, 0, 2)  # (W, B, 12)

        outputs = []
        for emg_features in windows:
            # (B, frames_per_window * 20)
            initial_poses_flat = initial_poses.flatten(start_dim=1)

            # Concatenate the EMG and joint context features.
            # (B, frames_per_window * 20 + 64)
            combined = torch.cat([initial_poses_flat, emg_features], dim=1)
            pos_pred = self.predict(combined)  # (B, 20)

            # Update prediction with filter
            # (B, 20)
            all_poses = torch.cat([initial_poses, pos_pred.unsqueeze(1)], dim=1)
            pos_pred_update = self.filter(all_poses)

            outputs.append(pos_pred_update)

            # Update joint context: remove the oldest frame (along the last dimension)
            # and append the new prediction.
            # maintain initial_poses shape (B, frames_per_window, 20)
            initial_poses = torch.cat(
                [initial_poses[:, 1:, :], pos_pred_update.unsqueeze(1)],
                dim=1,
            )

        # Stack all predictions along a new time dimension.
        # Final output shape: (B, S, 20)
        return torch.stack(outputs, dim=1)

    def _step(self, name: str, batch):
        emg = batch["emg"]
        poses = batch["poses"]

        self.hm = handmodel2device(self.hm, self.device)

        # (B, S=frames_per_window, 20)
        initial_poses = poses[:, : self.frames_per_window, :]

        # ground truth (B, 20, S=full-frames_per_window)
        y = poses[:, self.frames_per_window :, :].permute(0, 2, 1)

        x = {"emg": emg, "initial_poses": initial_poses}
        y_hat = self(x).permute(0, 2, 1)  # (B, 20, S=full-frames_per_window)

        # Compute forward kinematics for predicted and ground truth poses.
        landmarks_pred = forward_kinematics(y_hat, self.hm)  # (B, S, L, 3)
        landmarks_gt = forward_kinematics(y, self.hm)  # (B, S, L, 3)

        sq_delta = (landmarks_pred - landmarks_gt) ** 2  # (B, S, L, 3)
        loss_per_lmk = sq_delta.mean(dim=-1)  # (B, S, L)
        loss_per_prediction = loss_per_lmk.mean(dim=-1)  # (B, S)
        loss_per_sequence = loss_per_prediction.mean(dim=1)  # (B,)
        loss = loss_per_sequence.mean()  # scalar

        self.log(f"{name}_loss", loss)
        self.log(f"{name}_lm_err_mm", loss.sqrt())
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch)

    def validation_step(self, batch, batch_idx):
        self._step("val", batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        """
        x["emg"]: (B, T, C) - requires T >= frames_per_window*emg_samples_per_frame and be devisable by emg_samples_per_frame
        x["initial_poses"]: (B, frames_per_window, 20) - the initial known hand pose
        returns S = T / emg_samples_per_frame - frames_per_window next predicted joints in shape (B, S, 20)
        """
        return self._forward(x["emg"], x["initial_poses"])
