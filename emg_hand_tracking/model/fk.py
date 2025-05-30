import torch
from emg2pose.kinematics import forward_kinematics, load_default_hand_model
from .util import handmodel2device
from .aa_fk import hand_landmarks_by_angles

UTHM = load_default_hand_model()


def umetrack_fk(
    y: torch.Tensor,  # (B, T, 20)
) -> torch.Tensor:  # (B, T, L, 3)
    y = y.permute(0, 2, 1)  # (B, 20, T)
    hm = handmodel2device(UTHM, y.device)
    return forward_kinematics(y, hm)


def aa_fk(
    y: torch.Tensor,  # (B, T, 20)
) -> torch.Tensor:  # (B, T, L, 3)
    B, T, D = y.shape  # D should be 20
    y_flat = y.reshape(B * T, D)  # (B*T, 20)
    landmarks = hand_landmarks_by_angles(y_flat)  # (B*T, L, 3)
    return landmarks.reshape(B, T, *landmarks.shape[1:])  # (B, T, L, 3)


FK_BY_POSE_FORMAT = {
    "UmeTrack": umetrack_fk,
    "AnatomicAngles": aa_fk,
}
