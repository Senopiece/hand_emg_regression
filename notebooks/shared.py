import numpy as np
import torch
import numpy as np
import plotly.graph_objs as go


def rootme():
    import os
    import sys

    # Go one level up from 'notebooks/' to project root
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

    # Set PYTHONPATH environment variable
    os.environ["PYTHONPATH"] = project_root

    # Also update sys.path so Python knows to look there for imports
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Optional: verify
    print("PYTHONPATH =", os.environ["PYTHONPATH"])


def plot_3d_hands(
    landmarks_sequence: torch.Tensor | np.ndarray,
    ps: torch.Tensor | np.ndarray | None = None,
    ps2: torch.Tensor | np.ndarray | None = None,
):
    if isinstance(landmarks_sequence, torch.Tensor):
        landmarks_sequence = landmarks_sequence.detach().cpu().numpy()

    if ps is not None and isinstance(ps, torch.Tensor):
        ps = ps.detach().cpu().numpy()
    if ps2 is not None and isinstance(ps2, torch.Tensor):
        ps2 = ps2.detach().cpu().numpy()

    n_frames = landmarks_sequence.shape[0]
    n_pts = landmarks_sequence.shape[1]
    labels = [str(i) for i in range(n_pts)]

    fig = go.Figure()

    # Add traces for the first frame
    fig.add_trace(
        go.Scatter3d(
            x=landmarks_sequence[0, :, 0],
            y=landmarks_sequence[0, :, 1],
            z=landmarks_sequence[0, :, 2],
            mode="markers+text",
            marker=dict(size=4, color="blue"),
            text=labels,
            textposition="top center",
            textfont=dict(size=8, color="black"),
            name="Landmarks",
        )
    )

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),  # Thumb
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 7),  # Index
        (0, 8),
        (8, 9),
        (9, 10),
        (10, 11),  # Middle
        (0, 12),
        (12, 13),
        (13, 14),
        (14, 15),  # Ring
        (0, 16),
        (16, 17),
        (17, 18),
        (18, 19),  # Pinky
        (1, 4),
        (4, 8),
        (8, 12),
        (12, 16),  # Palm
    ]

    # Add traces for connections for the first frame
    for i, j in connections:
        fig.add_trace(
            go.Scatter3d(
                x=[landmarks_sequence[0, i, 0], landmarks_sequence[0, j, 0]],
                y=[landmarks_sequence[0, i, 1], landmarks_sequence[0, j, 1]],
                z=[landmarks_sequence[0, i, 2], landmarks_sequence[0, j, 2]],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
        )

    def add_beams(landmarks, vectors, color):
        for i in range(n_pts):
            start = landmarks[i]
            end = start + vectors[i] * 0.1
            fig.add_trace(
                go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                )
            )

    # Add ps (red) and ps2 (green) beams if provided for first frame
    if ps is not None:
        add_beams(landmarks_sequence[0], ps[0], "red")
    if ps2 is not None:
        add_beams(landmarks_sequence[0], ps2[0], "green")

    # Create frames for the animation
    frames = []
    for k in range(n_frames):
        frame_data = [
            go.Scatter3d(
                x=landmarks_sequence[k, :, 0],
                y=landmarks_sequence[k, :, 1],
                z=landmarks_sequence[k, :, 2],
                mode="markers+text",
                marker=dict(size=4, color="blue"),
                text=labels,
                textposition="top center",
                textfont=dict(size=8, color="black"),
                name="Landmarks",
            )
        ] + [
            go.Scatter3d(
                x=[landmarks_sequence[k, i, 0], landmarks_sequence[k, j, 0]],
                y=[landmarks_sequence[k, i, 1], landmarks_sequence[k, j, 1]],
                z=[landmarks_sequence[k, i, 2], landmarks_sequence[k, j, 2]],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
            for i, j in connections
        ]

        if ps is not None:
            for i in range(n_pts):
                start = landmarks_sequence[k, i]
                end = start + ps[k, i] * 0.1
                frame_data.append(
                    go.Scatter3d(
                        x=[start[0], end[0]],
                        y=[start[1], end[1]],
                        z=[start[2], end[2]],
                        mode="lines",
                        line=dict(color="red", width=2),
                        showlegend=False,
                    )
                )

        if ps2 is not None:
            for i in range(n_pts):
                start = landmarks_sequence[k, i]
                end = start + ps2[k, i] * 0.1
                frame_data.append(
                    go.Scatter3d(
                        x=[start[0], end[0]],
                        y=[start[1], end[1]],
                        z=[start[2], end[2]],
                        mode="lines",
                        line=dict(color="green", width=2),
                        showlegend=False,
                    )
                )

        frames.append(go.Frame(data=frame_data, name=str(k)))

    fig.frames = frames

    # Add slider and play/pause button
    sliders = [
        {
            "pad": {"b": 10, "t": 10},
            "len": 0.9,
            "x": 0.1,
            "xanchor": "left",
            "y": 0,
            "yanchor": "top",
            "steps": [
                {
                    "args": [
                        [f.name],
                        {
                            "frame": {"duration": 33.33, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(frames)
            ],
        }
    ]

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 33.33, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 30},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=sliders,
        scene=dict(
            xaxis=dict(
                range=[2, -2],
                title="X",
            ),
            yaxis=dict(
                range=[-1, 3],
                title="Y",
            ),
            zaxis=dict(
                range=[-2, 2],
                title="Z",
            ),
            camera=dict(eye=dict(x=0.2, y=0.2, z=0.2)),
        ),
        scene_aspectmode="cube",
        title="3D Hand Pose with Point Labels",
        margin=dict(l=0, r=0, b=0, t=30),
        height=400,
    )

    fig.show()


## Forward kinematics

import torch

DEFAULT_MORPHOLOGY = [
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


def hand_landmarks_by_angles(
    angles: torch.Tensor,
):
    """
    angles: (B, 20)
    Returns: (B, 20, 3)
    """

    B = angles.shape[0]

    morphology = torch.tensor(
        DEFAULT_MORPHOLOGY,
        dtype=angles.dtype,
        device=angles.device,
    )

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

    return torch.cat(
        [landmarks[:, :1, :], landmarks[:, 2:, :]], dim=1
    )  # cut thumb base
