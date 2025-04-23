import torch
from emg2pose.kinematics import HandModel, forward_kinematics


def handmodel2device(hm: HandModel, device):
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


POINTS_SELECT = [5, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]
L = len(POINTS_SELECT)
L_3 = L * 3


# x: B, S, 20
def forward_hand_kinematics(x: torch.Tensor, hand_model):
    hands = forward_kinematics(x.permute(0, 2, 1), hand_model).squeeze(1)
    return hands[:, :, POINTS_SELECT, :]  # B, S, L, 3
