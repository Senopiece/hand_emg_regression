from typing import Any, Dict
from emg2pose.kinematics import forward_kinematics, load_default_hand_model
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .util import handmodel2device
from .modules import (
    ExtractLearnableSlices,
    LearnablePatternCosSimilarity,
    LearnablePatternDot,
    LearnablePatternSimilarity,
    LearnablePatternUnnormSimilarity,
    Max,
    Mean,
    Parallel,
    StdDev,
    Variance,
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


class _Base(Model):
    def _forward(self, emg, initial_poses):
        emg = emg.permute(0, 2, 1)  # (B, T, C)
        windows = self.emg_feature_extract(emg).permute(1, 0, 2)  # (W, B, E)

        outputs = []
        for emg_features in windows:
            # (B, frames_per_window * 20)
            initial_poses_flat = initial_poses.flatten(start_dim=1)

            # Concatenate the EMG and joint context features.
            # (B, frames_per_window * 20 + E)
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

        self.log(f"{name}_lm_err_mm", loss.sqrt())

        # Add L1 regularization
        l1_lambda = 1e-5
        l1_loss = sum(param.abs().sum() for param in self.parameters())
        loss += l1_lambda * l1_loss

        self.log(f"{name}_loss", loss)
        return loss


class _BaseMultiF(_Base):
    """DynamicSlice21BigMultifeatured"""

    def __init__(self, sim):
        super().__init__()

        slices = 256
        patterns = 128
        slice_width = 33
        E = 64  # emg feature size

        self.channels = 16
        self.emg_samples_per_frame = 32
        self.frames_per_window = 8

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
                        sim(n=patterns, width=slice_width),  # -> (B, slices, patterns)
                        nn.Flatten(),  # -> (B, slices*patterns)
                    ),
                    Variance(),  # -> (B, slices)
                    Mean(),  # -> (B, slices)
                    Max(),  # -> (B, slices)
                    StdDev(),  # -> (B, slices)
                ),
                nn.Linear(slices * patterns + 4 * slices, 1024, bias=False),
                nn.ReLU(),
                nn.Linear(1024, E),
            ),
        )  # -> (B, W, E), S=W

        self.predict = nn.Sequential(
            nn.Linear(self.frames_per_window * 20 + E, 512),
            nn.ReLU(),
            nn.Linear(512, 20),
        )

        self.filter = WeightedMean(self.frames_per_window + 1)


class dot(_BaseMultiF):
    def __init__(self):
        super().__init__(LearnablePatternDot)


class cos(_BaseMultiF):
    def __init__(self):
        super().__init__(LearnablePatternCosSimilarity)


class corr(_BaseMultiF):
    def __init__(self):
        super().__init__(LearnablePatternSimilarity)


class unnorm(_BaseMultiF):
    def __init__(self):
        super().__init__(LearnablePatternUnnormSimilarity)


class _DynamicSlice19_big(_Base):
    def __init__(self):
        super().__init__()

        slices = 256
        patterns = 128
        slice_width = 65
        E = 64  # emg feature size

        self.channels = 16
        self.emg_samples_per_frame = 32
        self.frames_per_window = 8

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
                LearnablePatternCosSimilarity(
                    n=patterns, width=slice_width
                ),  # -> (B, slices, patterns)
                nn.Flatten(),
                nn.Linear(slices * patterns, 512, bias=False),
                nn.ReLU(),
                nn.Linear(512, E),
            ),
        )  # -> (B, W, E), S=W

        self.predict = nn.Sequential(
            nn.Linear(self.frames_per_window * 20 + E, 512),
            nn.ReLU(),
            nn.Linear(512, 20),
        )

        self.filter = WeightedMean(self.frames_per_window + 1)
