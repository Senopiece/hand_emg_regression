from typing import Any, Dict
from emg2pose.kinematics import forward_kinematics, load_default_hand_model
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .util import handmodel2device
from .modules import (
    ExtractLearnableSlices,
    LearnablePatternSimilarity,
    Max,
    Parallel,
    StdDev,
    WindowedApply,
    WeightedMean,
)


class Model(pl.LightningModule):
    _impls: Dict[str, type["Model"]] = {}

    @staticmethod
    def _resolve_immediate_name(c: type):
        if c.__doc__:
            docstring = c.__doc__.strip()
            assert (
                docstring.isalnum()
            ), f"Class docstring must be alphanumeric: {docstring}"
            return docstring
        else:
            if c.__name__.startswith("_"):
                return ""
            return c.__name__

    @staticmethod
    def _resolve_name(c: type):
        prefix = "_".join(
            Model._resolve_immediate_name(base)
            for base in c.__bases__
            if base.__name__ != "Model" and base.__name__ != "object"
        )
        suffix = Model._resolve_immediate_name(c)
        return f"{prefix}_{suffix}" if prefix != "" else suffix

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if not cls.__name__.startswith("_"):
            cls._impls[cls.name()] = cls

    @classmethod
    def name(cls):
        return Model._resolve_name(cls)

    @classmethod
    def construct(cls, name):
        return cls._impls[name]()

    @classmethod
    def construct_from_checkpoint(cls, name, ckpt):
        return cls._impls[name].load_from_checkpoint(ckpt)

    @classmethod
    def impls(cls):
        return cls._impls.keys()

    def _forward(self, emg, initial_poses):
        raise NotImplementedError()

    def forward(self, x):
        """
        x["emg"]: (B, T, C) - requires T >= frames_per_window*emg_samples_per_frame and be devisable by emg_samples_per_frame
        x["initial_poses"]: (B, frames_per_window, 20) - the initial known hand pose
        returns S = T / emg_samples_per_frame - frames_per_window next predicted joints in shape (B, S, 20)
        """
        return self._forward(x["emg"], x["initial_poses"])

    def training_step(self, batch, batch_idx):
        return self._step("train", batch)

    def validation_step(self, batch, batch_idx):
        self._step("val", batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class V43(Model):
    def __init__(self):
        super().__init__()

        slices = 256
        patterns = 128

        pattern_subfeature_windows = 10  # TODO: mb separate for subfeatures
        pattern_subfeature_width = 7
        pattern_subfeature_stride = 3

        slice_width = (
            pattern_subfeature_width
            + (pattern_subfeature_windows - 1) * pattern_subfeature_stride
        )

        self.channels = 16
        self.emg_samples_per_frame = 32  # 60 predictions/sec
        self.frames_per_window = 8
        self.pos_vel_acc_datasize = (
            self.frames_per_window * 20
            + (self.frames_per_window - 1) * 20
            + (self.frames_per_window - 2) * 20
        )

        self.hm = load_default_hand_model()

        self.emg_window_length = self.emg_samples_per_frame * self.frames_per_window

        # T = total_seq_length + S*emg_samples_per_frame
        # C = channels

        # separate windows per prediction if feed a sequence for more than one prediction
        self.emg_feature_extract = WindowedApply(  # <- (B, C, T)
            window_len=self.emg_window_length,
            step=self.emg_samples_per_frame,
            f=nn.Sequential(  # <- (B, C, total_seq_length)
                nn.ZeroPad1d(
                    slice_width // 2
                ),  # happens to be not needed, mb remove later
                ExtractLearnableSlices(
                    n=slices, width=slice_width
                ),  # -> (B, slices, slice_width)
                Parallel(
                    nn.Sequential(
                        LearnablePatternSimilarity(
                            n=patterns, width=slice_width
                        ),  # -> (B, slices, patterns)
                        nn.Flatten(),  # -> (B, slices*patterns)
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        WindowedApply(
                            window_len=pattern_subfeature_width,
                            step=pattern_subfeature_stride,
                            # -> (B, slices, pattern_subfeature_width)
                            f=StdDev(),  # -> (B, slices)
                        ),  # -> (B, W, slices)
                        WeightedMean(pattern_subfeature_windows),  # -> (B, slices)
                    ),
                    nn.Sequential(
                        WindowedApply(
                            window_len=pattern_subfeature_width,
                            step=pattern_subfeature_stride,
                            # -> (B, slices, pattern_subfeature_width)
                            f=Max(),  # -> (B, slices)
                        ),  # -> (B, W, slices)
                        WeightedMean(pattern_subfeature_windows),  # -> (B, slices)
                    ),
                ),
            ),
        )  # -> (B, W, slices * patterns + 2 * slices), S=W

        self.synapse_feature_extract = nn.Sequential(
            nn.Linear(
                slices * patterns + 2 * slices + self.pos_vel_acc_datasize,
                2048,
            ),
            nn.ReLU(),
        )

        self.muscle_feature_extract = nn.Linear(
            2048 + self.pos_vel_acc_datasize,
            64,
        )

        self.predict = nn.Sequential(
            nn.Linear(
                64 + self.pos_vel_acc_datasize,
                1024,
            ),
            nn.ReLU(),
            nn.Linear(1024, 20),
        )

        self.filter = WeightedMean(self.frames_per_window + 1)

    def _forward(self, emg, initial_poses):
        emg = emg.permute(0, 2, 1)  # (B, T, C)
        windows = self.emg_feature_extract(emg).permute(1, 0, 2)  # (W, B, E)

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

        # Ground truth (B, 20, S=full-frames_per_window)
        y = poses[:, self.frames_per_window :, :].permute(0, 2, 1)

        x = {"emg": emg, "initial_poses": initial_poses}
        y_hat = self(x).permute(0, 2, 1)  # (B, 20, S=full-frames_per_window)

        # Compute forward kinematics for predicted and ground truth poses.
        landmarks_pred = forward_kinematics(y_hat, self.hm)  # (B, S, L, 3)
        landmarks_gt = forward_kinematics(y, self.hm)  # (B, S, L, 3)

        # First term is the right error
        sq_delta = (landmarks_pred - landmarks_gt) ** 2  # (B, S, L, 3)
        loss_per_lmk = sq_delta.sum(dim=-1)  # (B, S, L)
        loss_per_prediction = loss_per_lmk.mean(dim=-1)  # (B, S)
        loss_per_sequence = loss_per_prediction.mean(dim=1)  # (B,)
        loss = loss_per_sequence.mean()  # scalar

        # Add term to follow the movement
        for _ in range(2):
            # Differentiate
            landmarks_pred = landmarks_pred.diff(dim=1)  # (B, S, L, 3)
            landmarks_gt = landmarks_gt.diff(dim=1)  # (B, S, L, 3)

            sq_delta = (landmarks_pred - landmarks_gt) ** 2  # (B, S, L, 3)
            loss_per_lmk = sq_delta.sum(dim=-1)  # (B, S, L)
            loss_per_prediction = loss_per_lmk.mean(dim=-1)  # (B, S)
            loss_per_sequence = loss_per_prediction.mean(dim=1)  # (B,)
            loss += loss_per_sequence.mean()  # scalar

        # Add L1 regularization
        l1_lambda = 1e-5
        l1_loss = sum(param.abs().sum() for param in self.parameters())
        loss += l1_lambda * l1_loss

        self.log(f"{name}_loss", loss)
        return loss
