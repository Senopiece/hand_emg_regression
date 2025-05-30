import torch


def rot(
    v: torch.Tensor,
    p: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
):
    """
    Batched rotation function.

    v, p: (B, 3) tensors representing vector and the first axis of rotation
    alpha, beta: (B,) tensors representing angles to use for rotation
    """

    sinA = torch.sin(alpha).unsqueeze(-1)  # (B, 1)
    cosA = torch.cos(alpha).unsqueeze(-1)  # (B, 1)
    sinB = torch.sin(beta).unsqueeze(-1)  # (B, 1)
    cosB = torch.cos(beta).unsqueeze(-1)  # (B, 1)

    q = torch.cross(v, p, dim=1)  # (B, 3)

    p_hat = p.mul(cosB).sub(v.mul(sinB))
    v_hat = q.mul(sinA).add(v.mul(cosA.mul(cosB))).add(p.mul(cosA.mul(sinB)))

    return v_hat, p_hat


_V = [0, 1, 0]
_P = [0, 0, 1]

MORPHOLOGY = [
    [
        0.3502,
        0.3765,
        0.2691,
        -0.2615,
        0.2654,
        0.3382,
        0.7531,
        0.0717,
        0.9498,
        1.2573,
    ],
    [
        0.4468,
        0.2765,
        0.2192,
        -0.0019,
        0.9693,
        0.2282,
        -0.3828,
        0.3692,
        -0.2930,
        -0.1947,
    ],
    [
        0.5582,
        0.3288,
        0.2200,
        -0.0020,
        0.9970,
        -0.0124,
        -0.5517,
        0.1407,
        -0.2910,
        -0.1613,
    ],
    [
        0.5533,
        0.3325,
        0.2214,
        -0.0453,
        0.9214,
        -0.2073,
        -0.1670,
        0.7382,
        -0.3557,
        -1.0503,
    ],
    [
        0.4466,
        0.2851,
        0.2230,
        -0.1164,
        0.7822,
        -0.3913,
        -0.1512,
        0.5448,
        -0.3952,
        -0.9645,
    ],
]


def hand_landmarks_by_angles(
    angles: torch.Tensor,
):
    """
    angles: (B, 20)
    Returns: (B, 20, 3)
    """

    B = angles.shape[0]

    morphology = torch.tensor(MORPHOLOGY, dtype=angles.dtype, device=angles.device)

    landmarks = torch.zeros(B, 21, 3, dtype=angles.dtype, device=angles.device)
    angles = angles.unflatten(dim=-1, sizes=(5, 4))

    # Write morphological const points
    # landmarks[:, 0, :] is already zero
    for i in range(0, 5):
        landmarks[:, 4 * i + 1, :] = morphology[i, 3:6]

    V = torch.tensor(_V, dtype=angles.dtype, device=angles.device).expand(B, 3)
    P = torch.tensor(_P, dtype=angles.dtype, device=angles.device).expand(B, 3)

    for finger_index, (morph, local_angles) in enumerate(
        zip(morphology, angles.transpose(0, 1))
    ):
        base_idx = finger_index * 4 + 1
        bone_lengths = morph[0:3]
        joint = morph[3:6].expand(B, 3).clone()
        alpha, beta, gamma = morph[6:9].clone()

        a0, b0, a2, a3 = local_angles.unbind(dim=-1)
        chain_angles = torch.stack(
            [
                torch.stack([a0, b0], dim=-1),
                torch.stack([a2, torch.zeros_like(a2)], dim=-1),
                torch.stack([a3, torch.zeros_like(a3)], dim=-1),
            ],
            dim=1,
        )  # (B, 3, 2)

        v, p = rot(V, P, alpha, beta)

        # Rotate p around v according to parameter gamma
        sinG, cosG = torch.sin(gamma), torch.cos(gamma)
        p = p.mul(cosG).add(torch.cross(p, v, dim=-1).mul(sinG))

        # Iterate over the chain and write the 3d points
        for j in range(3):
            l = bone_lengths[j]
            alpha, beta = chain_angles[:, j, 0], chain_angles[:, j, 1]
            v, p = rot(v, p, alpha, beta)
            joint = joint.add(v.mul(l))
            landmarks[:, base_idx + j + 1, :] = joint

    landmarks = torch.cat(
        [landmarks[:, :1, :], landmarks[:, 2:, :]], dim=1
    )  # cut thumb base

    landmarks *= (
        90.0  # scale to typical hand size (now landmarks are in mm coordinates)
    )

    return landmarks
