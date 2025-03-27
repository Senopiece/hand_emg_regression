from emg2pose.kinematics import forward_kinematics, load_default_hand_model
import torch
import torch.nn as nn
import pytorch_lightning as pl

from emg2pose.kinematics import HandModel


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
        frames_per_item: int,
        channels: int,
    ):
        """
        Parameters:
            emg_samples_per_frame: Number of samples per frame.
            frames_per_item: Number of frames per data item.
            channels: Number of channels in the EMG signal.
        """
        super().__init__()
        self.save_hyperparameters()

        self.hm = load_default_hand_model()

        # Total sequence length (time dimension)
        total_seq_length = emg_samples_per_frame * frames_per_item

        # Conv1d expects (B, channels, sequence_length)
        self.emg_squeezer = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=1024, kernel_size=101, padding=50
            ),
            nn.Flatten(),  # Flattens from (B, 1024, total_seq_length) to (B, 1024 * total_seq_length)
            nn.Linear(1024 * total_seq_length, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.ReLU(),
        )

        self.prev_frames_squeezer = nn.Sequential(
            nn.Linear(frames_per_item * 20, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
        )

        self.composer = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
        )

    def forward(self, x):
        """
        x["emg_chunk"]: (B, T, C) where T = emg_samples_per_frame * frames_per_item
        x["joint_chunk"]: (B, frames_per_item, 20)
        """
        emg = x["emg_chunk"]  # (B, T, C)
        joint_ctx = x["joint_chunk"]  # (B, frames_per_item, 20)

        self.hm = _handmodel2device(self.hm, self.device)

        # Prepare EMG for conv
        emg = emg.permute(0, 2, 1)  # (B, C, T)
        emg_features = self.emg_squeezer(emg)  # (B, 10)

        # Flatten joint context and pass through squeezer
        joint_features = joint_ctx.view(
            joint_ctx.size(0), -1
        )  # (B, frames_per_item * 20)
        joint_features = self.prev_frames_squeezer(joint_features)  # (B, 10)

        # Combine both and pass through final composer
        combined = torch.cat([emg_features, joint_features], dim=1)  # (B, 20)
        output = self.composer(combined)  # (B, 20)

        return output

    def _step(self, name: str, batch):
        x, y = batch

        # Predicted joint angles: shape (B, 20)
        y_hat = self(x)

        # Convert predicted angles to BCT format: (B, 20, T) with T=1.
        y_hat_bct = y_hat.unsqueeze(-1)  # shape: (B, 20, 1)

        # Compute 3D landmarks for predictions.
        # shape: (B, 1, num_landmarks, 3)
        landmarks_pred = forward_kinematics(y_hat_bct, self.hm)
        # shape: (B, num_landmarks, 3)
        landmarks_pred = landmarks_pred.squeeze(1)

        # Do the same for ground-truth angles.
        y_bct = y.unsqueeze(-1)  # (B, 20, 1)
        landmarks_gt = forward_kinematics(y_bct, self.hm)
        landmarks_gt = landmarks_gt.squeeze(1)  # shape: (B, num_landmarks, 3)

        # Calculate sum of squared errors per sample and divide by number of landmarks.
        loss_per_sample = torch.mean(
            torch.sum((landmarks_pred - landmarks_gt) ** 2, dim=2), dim=1
        )

        # Average loss over the batch.
        loss = loss_per_sample.mean()
        self.log(f"{name}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch)

    def validation_step(self, batch, batch_idx):
        self._step("val", batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
