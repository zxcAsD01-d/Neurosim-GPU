"""Adaptive internal-time halting for Cortex-LM.

The model may spend a variable number of internal updates on one token. A small
halting head produces a stop probability at each internal step. Training uses
the soft halting distribution (so the halting head receives gradients); eval can
use a hard cumulative threshold to report actual ticks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HaltingConfig:
    threshold: float = 0.5
    ponder_weight: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if self.ponder_weight < 0:
            raise ValueError("ponder_weight must be >= 0")


class AdaptiveHaltingHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.halt = nn.Linear(d_model, 1)

    def logits(self, readouts: torch.Tensor) -> torch.Tensor:
        """``readouts`` is [batch, max_ticks, d_model] -> [batch, max_ticks]."""

        return self.halt(readouts).squeeze(-1)


def soft_halting(
    readouts: torch.Tensor,
    logits: torch.Tensor,
    config: HaltingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted readout, halting weights, expected ticks."""

    if readouts.ndim != 3 or logits.shape != readouts.shape[:2]:
        raise ValueError("readouts/logits shape mismatch")
    probabilities = torch.sigmoid(logits)
    weights = []
    remaining = torch.ones_like(probabilities[:, :1])
    for step in range(probabilities.shape[1]):
        weight = remaining * probabilities[:, step : step + 1]
        weights.append(weight)
        remaining = remaining * (1.0 - probabilities[:, step : step + 1])
    weights_tensor = torch.cat(weights, dim=1)
    # Residual mass (never halted) is assigned to the last internal step.
    weights_tensor[:, -1:] = weights_tensor[:, -1:] + remaining
    weights_tensor = weights_tensor / weights_tensor.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    weighted = (weights_tensor.unsqueeze(-1) * readouts).sum(dim=1)
    positions = torch.arange(
        1, probabilities.shape[1] + 1, device=readouts.device
    ).float()
    expected = (weights_tensor * positions).sum(dim=1)
    return weighted, weights_tensor, expected


@torch.no_grad()
def hard_ticks(
    logits: torch.Tensor, threshold: float
) -> torch.Tensor:
    """First step where cumulative halt probability reaches threshold."""

    cumulative = torch.cumsum(torch.sigmoid(logits), dim=1)
    reached = cumulative >= threshold
    first = reached.float().argmax(dim=1) + 1
    no_halt = ~reached.any(dim=1)
    return torch.where(
        no_halt, torch.full_like(first, logits.shape[1]), first
    )


def ponder_cost(expected_ticks: torch.Tensor, max_ticks: int) -> torch.Tensor:
    return expected_ticks.mean() / max(max_ticks, 1)
