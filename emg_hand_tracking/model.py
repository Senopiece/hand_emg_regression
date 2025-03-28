from emg2pose.kinematics import forward_kinematics, load_default_hand_model
import torch
import torch.nn as nn
import pytorch_lightning as pl

from emg2pose.kinematics import HandModel

from .util import WindowedApply


def _handmodel2device(hm: HandModel, device):
    if hm.joint_rotation_axes.device == device:
        return hm
    return HandModel(
        joint_rotation_axes=hm.joint_rotation_axes.to(device),
        joint_rest_positions=hm.joint_rest_positions.to(device),
        joint_frame_index=hm.joint_frame_index.to(device),
        joint_parent=hm.joint_parent.to(device),
        joint_first_child=hm.joint_first_child.to(device),
        joint_next_sibling=hm.joint_next_sibling.to(device),
        landmark_rest_positions=hm.landmark_rest_positions.to(device),
        landmark_rest_bone_weights=hm.landmark_rest_bone_weights.to(device),
        landmark_rest_bone_indices=hm.landmark_rest_bone_indices.to(device),
        hand_scale=None if hm.hand_scale is None else hm.hand_scale.to(device),
        mesh_vertices=(
            None if hm.mesh_vertices is None else hm.mesh_vertices.to(device)
        ),
        mesh_triangles=(
            None if hm.mesh_triangles is None else hm.mesh_triangles.to(device)
        ),
        dense_bone_weights=(
            None if hm.dense_bone_weights is None else hm.dense_bone_weights.to(device)
        ),
        joint_limits=(None if hm.joint_limits is None else hm.joint_limits.to(device)),
    )


class Model(pl.LightningModule):
    def __init__(
        self,
        emg_samples_per_frame: int,
        frames_per_window: int,
        channels: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.emg_samples_per_frame = emg_samples_per_frame
        self.frames_per_window = frames_per_window
        self.channels = channels
        self.hm = load_default_hand_model()

        self.emg_window_length = emg_samples_per_frame * frames_per_window

        # T = total_seq_length + S*emg_samples_per_frame
        # C = channels

        self.conv = nn.Sequential(  # <- (B, C, T)
            nn.Conv1d(
                in_channels=channels,
                out_channels=1024,
                kernel_size=101,
                padding=50,
            ),  # -> (B, 1024, T)
            WindowedApply(
                window_len=self.emg_window_length,
                step=emg_samples_per_frame,
                f=nn.Sequential(  # <- (B, 1024, total_seq_length)
                    nn.Flatten(),
                    nn.Linear(1024 * self.emg_window_length, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 12),
                ),
            ),  # -> (B, W, 12), S=W
        )

        self.frames_features = nn.Sequential(  # <- (B, frames_per_window, 20)
            nn.Flatten(),
            nn.Linear(frames_per_window * 20, 80),
            nn.ReLU(),
            nn.Linear(80, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
        )

        self.predict = nn.Sequential(  # <- (B, 12+12)
            nn.Linear(24, 256),
            nn.ReLU(),
            nn.Linear(256, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
        )

    def _forward(self, emg, initial_poses):
        emg = emg.permute(0, 2, 1)  # (B, T, C)
        windows = self.conv(emg).permute(1, 0, 2)  # (W, B, 12)

        outputs = []
        for emg_features in windows:
            joint_features = self.frames_features(initial_poses)  # (B, 12)

            # Concatenate the EMG and joint context features.
            combined = torch.cat([emg_features, joint_features], dim=1)  # (B, 24)
            pos_pred = self.predict(combined)  # (B, 20)

            outputs.append(pos_pred)

            # Update joint context: remove the oldest frame (along the last dimension)
            # and append the new prediction.
            # maintain initial_poses shape (B, frames_per_window, 20)
            initial_poses = torch.cat(
                [initial_poses[:, 1:, :], pos_pred.unsqueeze(1)], dim=1
            )

        # Stack all predictions along a new time dimension.
        # Final output shape: (B, S, 20)
        return torch.stack(outputs, dim=1)

    def _step(self, name: str, batch):
        emg = batch["emg"]
        poses = batch["poses"]

        self.hm = _handmodel2device(self.hm, self.device)

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
