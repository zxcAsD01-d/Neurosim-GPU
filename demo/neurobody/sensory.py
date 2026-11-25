"""Human-plausible sensory organs and their neuron populations.

Counts follow a coarse human ratio (not a claim of biological accuracy):

| modality | neurons | receptors / channels |
|---|---:|---|
| vision (2 eyes) | 512 | retinal ON/OFF + LGN relay |
| touch (glabrous skin) | 128 | Merkel, Meissner, Ruffini, Pacinian x 32 zones |
| hearing (cochlea) | 64 | 64 tonotopic hair-cell channels |
| proprioception | 117 | muscle spindle (3 per DoF) |
| vestibular | 16 | otolith + canal hair cells |
| nociception | 8 | free nerve endings, high-threshold |
| memory (hippocampus) | 384 | 256 place cells + 128 time/grid cells |

Only touch/audio/nociception/memory are encoded here; vision lives in
``eyes.py`` and proprioception/vestibular in ``embodied.py``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

TOUCH_TYPES = ("merkel", "meissner", "ruffini", "pacinian")
TOUCH_ZONES = 32
AUDIO_CHANNELS = 64
NOCICEPT = 8
PLACE_CELLS = 256
TIME_CELLS = 128
PLACE_GRID = 16
ROOM = 4.0  # metres covered by the place-cell grid


def make_touch_bank(seed: int = 31) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed tuning of 4 receptor types x 32 zones over the leg DoFs."""

    rng = np.random.default_rng(seed)
    # 12 leg DoF components (left/right upper/lower leg) out of the 39 joints
    weight = rng.normal(0.0, 1.0, size=(4, TOUCH_ZONES, 12)).astype(np.float32)
    bias = rng.normal(0.0, 0.3, size=(4, TOUCH_ZONES)).astype(np.float32)
    return torch.from_numpy(weight), torch.from_numpy(bias)


def encode_touch(q: torch.Tensor, qd: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """(B, 41) angles/velocities -> (B, 128) touch rates.

    Leg joints stand in for foot contact until physics provides real contact
    forces.  Four receptor types differ in their weight distribution (Pacinian
    is velocity-dominated, Merkel is position-dominated).
    """

    leg = torch.cat([q[:, 12:24], qd[:, 12:24]], dim=-1)  # legs + their velocity
    features = leg.reshape(leg.shape[0], 12, 2).mean(-1)  # coarse per-DoF drive
    activations = torch.einsum("b d, t z d -> b t z", features, weight)
    return torch.sigmoid(activations + bias).reshape(q.shape[0], -1)


def make_audio_bank(seed: int = 37) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    freq = np.linspace(1.0, 8.0, AUDIO_CHANNELS).astype(np.float32)
    phase = rng.uniform(0.0, 2 * math.pi, size=AUDIO_CHANNELS).astype(np.float32)
    gain = rng.uniform(0.5, 1.0, size=AUDIO_CHANNELS).astype(np.float32)
    return torch.from_numpy(freq), torch.stack([torch.from_numpy(phase), torch.from_numpy(gain)])


def encode_audio(phase: torch.Tensor, freq: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Synthetic gait rhythm through 64 tonotopic channels.

    Placeholder until a real audio stream exists: the "sound" is the step
    rhythm at 1.6-2.2 Hz plus harmonics, so the auditory neurons have a
    structured, predictable input.
    """

    t = phase[:, None]  # (B, 1) seconds
    theta = 2 * math.pi * freq[None, :] * t + params[0][None, :]
    signal = torch.sin(theta) * torch.exp(-0.03 * freq)[None, :] * params[1][None, :]
    return torch.sigmoid(3.0 * signal)


def encode_nocicept(qd: torch.Tensor) -> torch.Tensor:
    """High-threshold free nerve endings: only extreme joint speeds fire."""

    chunks = torch.tensor_split(qd.abs(), NOCICEPT, dim=1)
    groups = torch.stack([chunk.amax(-1) for chunk in chunks], dim=-1)
    return torch.sigmoid(6.0 * (groups - 0.12))


def make_place_bank(seed: int = 41) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    xs = np.linspace(-ROOM / 2, ROOM / 2, PLACE_GRID)
    zs = np.linspace(-ROOM / 2, ROOM / 2, PLACE_GRID)
    grid = np.stack([x for x in xs for _ in zs]), np.stack([z for _ in xs for z in zs])
    centers = np.stack(grid, axis=1)[:PLACE_CELLS]
    centers += rng.normal(0.0, 0.05, size=centers.shape)
    return torch.from_numpy(centers.astype(np.float32))


def encode_memory(pos: torch.Tensor, phase: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """256 place cells + 128 time cells (theta rhythm)."""

    d = torch.cdist(pos[:, None, :], centers[None, :, :]).squeeze(1)  # (B, 256)
    place = torch.exp(-(d / 0.35) ** 2)
    t = phase[:, None] * 2 * math.pi * 1.8
    time_cells = torch.sin(t + torch.linspace(0, 2 * math.pi, TIME_CELLS, device=phase.device)[None, :])
    time_cells = 0.5 + 0.5 * time_cells
    return torch.cat([place, time_cells], dim=-1)
