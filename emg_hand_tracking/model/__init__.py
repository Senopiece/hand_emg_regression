from typing import Dict, NamedTuple
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn

from emg_hand_tracking.dataset import EmgWithHand

from .fk import FK_BY_POSE_FORMAT
from .modules import (
    ExtractLearnableSlices,
    LearnablePatternSimilarity,
    Max,
    Parallel,
    Permute,
    StdDev,
    WindowedApply,
    WeightedMean,
)


# Input to the model
class InitialStateAndEmg(NamedTuple):
    emg: torch.Tensor  # ({B}, T*W, C), float32, T >= I
    initial_poses: torch.Tensor  # ({B}, I, 20), float32


class SubfeatureSettings(NamedTuple):
    width: int = 7  # in emg samples
    stride: int = 3  # in emg samples

    def windows(self, input_width: int):
        if self.width > input_width:
            return 0
        return (input_width - self.width) // self.stride + 1


class SubfeaturesSettings(NamedTuple):
    mx: SubfeatureSettings = SubfeatureSettings()
    std: SubfeatureSettings = SubfeatureSettings()


class Model(LightningModule):
    def __init__(
        self,
        pose_format: str,
        channels: int,
        emg_samples_per_frame: int,
        slices: int,
        patterns: int,
        poses_in_context: int,
        frames_per_window: int,
        slice_emg_width: int,  # in emg samples
        subfeatures: SubfeaturesSettings,
        synapse_features: int,
        muscle_features: int,
        predict_hidden_layer_size: int,
        lr: float = 1e-3,
        l1: float = 1e-4,
        l2: float = 1e-5,
        slmerr_k: float = 1.0,
        vel_k: float = 1.0,
        accel_k: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.set_pose_format(pose_format)

        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self.slmerr_k = slmerr_k
        self.vel_k = vel_k
        self.accel_k = accel_k

        self.channels = channels
        self.emg_samples_per_frame = emg_samples_per_frame
        self.poses_in_context = poses_in_context
        self.frames_per_window = frames_per_window
        self.pos_vel_acc_datasize = (
            self.frames_per_window * 20
            + (self.frames_per_window - 1) * 20
            + (self.frames_per_window - 2) * 20
        )

        self.emg_window_length = self.emg_samples_per_frame * self.frames_per_window

        # T = total_seq_length + S*emg_samples_per_frame
        # C = channels

        # separate windows per prediction if feed a sequence for more than one prediction
        self.emg_feature_extract = WindowedApply(  # <- (B, C, T)
            window_len=self.emg_window_length,
            step=self.emg_samples_per_frame,
            f=nn.Sequential(  # <- (B, C, total_seq_length)
                nn.ZeroPad1d(slice_emg_width // 2),
                ExtractLearnableSlices(
                    n=slices, width=slice_emg_width
                ),  # -> (B, slices, slice_emg_width)
                Parallel(
                    nn.Sequential(
                        LearnablePatternSimilarity(
                            n=patterns, width=slice_emg_width
                        ),  # -> (B, slices, patterns)
                        nn.Flatten(),  # -> (B, slices*patterns)
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        Permute(0, 2, 1),
                        WeightedMean(slice_emg_width),
                    ),  # -> (B, slices)
                    Max(),  # -> (B, slices)
                    StdDev(),  # -> (B, slices)
                    nn.Sequential(
                        WindowedApply(
                            window_len=subfeatures.mx.width,
                            step=subfeatures.mx.stride,
                            # -> (B, slices, pattern_subfeature_width)
                            f=StdDev(),  # -> (B, slices)
                        ),  # -> (B, W, slices)
                        WeightedMean(
                            subfeatures.mx.windows(slice_emg_width)
                        ),  # -> (B, slices)
                    ),
                    nn.Sequential(
                        WindowedApply(
                            window_len=subfeatures.std.width,
                            step=subfeatures.std.stride,
                            # -> (B, slices, pattern_subfeature_width)
                            f=Max(),  # -> (B, slices)
                        ),  # -> (B, W, slices)
                        WeightedMean(
                            subfeatures.std.windows(slice_emg_width)
                        ),  # -> (B, slices)
                    ),
                ),
            ),
        )  # -> (B, W, F), S=W

        parallel_layer = self.emg_feature_extract.f[2]  # type: ignore
        subfeatures_count = len(parallel_layer.modules_list) - 1

        self.synapse_feature_extract = nn.Sequential(
            nn.Linear(
                slices * patterns
                + subfeatures_count * slices
                + self.pos_vel_acc_datasize,
                synapse_features,
            ),
            nn.ReLU(),
        )

        self.muscle_feature_extract = nn.Linear(
            synapse_features + self.pos_vel_acc_datasize,
            muscle_features,
        )

        self.predict = nn.Sequential(
            nn.Linear(
                muscle_features + self.pos_vel_acc_datasize,
                predict_hidden_layer_size,
            ),
            nn.ReLU(),
            nn.Linear(predict_hidden_layer_size, 20),
        )

        self.filter = WeightedMean(self.frames_per_window + 1)

    def set_pose_format(self, pose_format: str):
        """
        Set the pose format for the forward kinematics.
        This is useful if the model needs to adapt to different pose formats.
        """
        if pose_format not in FK_BY_POSE_FORMAT:
            raise ValueError(f"Unsupported pose format: {pose_format}")
        self.forward_kinematics = FK_BY_POSE_FORMAT[pose_format]

    def _forward(self, emg: torch.Tensor, initial_poses: torch.Tensor):
        emg = emg.permute(0, 2, 1)  # (B, T, C)
        windows = self.emg_feature_extract(emg).permute(1, 0, 2)  # (W, B, E)

        initial_poses = initial_poses[:, -self.poses_in_context :, :]

        outputs = []
        for emg_features in windows:
            # (B, frames_per_window-1, 20)
            initial_poses_vels = initial_poses.diff(dim=1)
            # (B, frames_per_window-2, 20)
            initial_poses_accels = initial_poses_vels.diff(dim=1)

            # (B, frames_per_window * 20)
            initial_poses_flat = initial_poses.flatten(start_dim=1)
            # (B, (frames_per_window-1) * 20)
            initial_poses_vels_flat = initial_poses_vels.flatten(start_dim=1)
            # (B, (frames_per_window-2) * 20)
            initial_poses_accels_flat = initial_poses_accels.flatten(start_dim=1)

            pos_vel_acc = torch.cat(
                [
                    initial_poses_flat,
                    initial_poses_vels_flat,
                    initial_poses_accels_flat,
                ],
                dim=1,
            )

            # Run through chain of predictions
            synapse_f = self.synapse_feature_extract(
                torch.cat(
                    [
                        emg_features,
                        pos_vel_acc,
                    ],
                    dim=1,
                )
            )
            muscle_f = self.muscle_feature_extract(
                torch.cat(
                    [
                        synapse_f,
                        pos_vel_acc,
                    ],
                    dim=1,
                )
            )
            pos_pred = self.predict(
                torch.cat(
                    [
                        muscle_f,
                        pos_vel_acc,
                    ],
                    dim=1,
                )
            )
            pos_pred_update = self.filter(
                torch.cat([initial_poses, pos_pred.unsqueeze(1)], dim=1)
            )

            outputs.append(pos_pred_update)

            # Update joint context: remove the oldest frame (along the last dimension)
            # and append the new prediction.
            # maintain initial_poses shape (B, poses_in_context, 20)
            initial_poses = torch.cat(
                [initial_poses[:, 1:, :], pos_pred_update.unsqueeze(1)],
                dim=1,
            )

        # Stack all predictions along a new time dimension.
        # Final output shape: (B, S, 20)
        return torch.stack(outputs, dim=1)

    def forward(self, *args, **kwargs):
        """
        some initial emg and poses are provided with some further emg,
        models estimates hand poses corresponding to further emg
        and returns these as ({B}, T + 1 - I, 20)
        """
        # Extract x from three different types of inputs:
        # - forward(emg=..., initial_poses=...)
        # - forward(InitialStateAndEmg(emg=..., initial_poses=...))
        # - forward({"emg": ..., "initial_poses":...})
        if len(args) != 0:
            x: InitialStateAndEmg | Dict[str, torch.Tensor] = args[0]
            if not isinstance(x, InitialStateAndEmg):
                x = InitialStateAndEmg(**x)
        else:
            x = InitialStateAndEmg(**kwargs)

        if len(x.emg.shape) == 2:
            # handle not batched input, return also not batched
            return self._forward(
                x.emg.unsqueeze(0),
                x.initial_poses.unsqueeze(0),
            ).squeeze(0)
        else:
            return self._forward(x.emg, x.initial_poses)

    def training_step(self, batch, batch_idx):
        return self._step("train", batch)

    def validation_step(self, batch, batch_idx):
        self._step("val", batch)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.l2,
        )

    # @torch.compile(backend="cudagraphs")
    def _metrics(self, emg, poses):
        # (B, S=frames_per_window, 20)
        initial_poses = poses[:, : self.frames_per_window, :]

        # Ground truth (B, 20, S=full-frames_per_window)
        y = poses[:, self.frames_per_window :, :]

        y_hat = self.forward(
            emg=emg,
            initial_poses=initial_poses,
        )  # (B, S=full-frames_per_window, 20)

        # Compute forward kinematics for predicted and ground truth poses.
        landmarks_pred = self.forward_kinematics(y_hat)  # (B, S, L, 3)
        landmarks_gt = self.forward_kinematics(y)  # (B, S, L, 3)

        # First term is the right squared mean landmark error
        sq_delta = (landmarks_pred - landmarks_gt) ** 2  # (B, S, L, 3)
        slmerr = sq_delta.sum(dim=-1).mean()

        loss = slmerr * self.slmerr_k
        lmerr = slmerr.sqrt()

        # A term for differential follow (reduces jitter and helps to learn faster)
        for k in [self.vel_k, self.accel_k]:
            # Differentiate
            landmarks_pred = landmarks_pred.diff(dim=1)  # (B, S, L, 3)
            landmarks_gt = landmarks_gt.diff(dim=1)  # (B, S, L, 3)

            sq_delta = (landmarks_pred - landmarks_gt) ** 2  # (B, S, L, 3)
            loss += k * sq_delta.sum(dim=-1).mean()

        # Add L1 regularization loss manually
        l1_loss = sum(p.abs().sum() for p in self.parameters() if p.requires_grad)
        loss += self.l1 * l1_loss

        return lmerr, loss

    def _step(self, name: str, batch: EmgWithHand | tuple):
        if isinstance(batch, EmgWithHand):
            emg = batch.emg
            poses = batch.poses
        elif isinstance(batch, tuple):
            emg, poses = batch
        else:
            raise ValueError(
                f"Expected batch to be EmgWithHand or tuple, got {type(batch)}"
            )

        lmerr, loss = self._metrics(emg, poses)

        self.log(f"{name}_lmerr", lmerr)
        self.log(f"{name}_loss", loss)

        return loss
