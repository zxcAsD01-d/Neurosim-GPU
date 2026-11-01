"""Minimal cognitive-core modules for the consciousness research route.

These are prototypes, not claims of consciousness:

* GlobalWorkspace: capacity-limited competition, broadcast and an ignition score.
* SelfMonitor: confidence / agency / error heads and expected calibration error.
* HippocampalMemory: sparse pattern separation, one-shot write, pattern-completion
  read and replay.
* should_ask: an uncertainty-gated active-inquiry policy.

They are deliberately small and testable so the architecture can be ablated
before it is wired into the language model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WorkspaceConfig:
    slots: int = 32
    top_k: int = 8


class GlobalWorkspace(nn.Module):
    """A tiny global workspace with top-k competition and broadcast."""

    def __init__(self, d_model: int, config: WorkspaceConfig | None = None) -> None:
        super().__init__()
        self.config = config or WorkspaceConfig()
        if not 0 < self.config.top_k <= self.config.slots:
            raise ValueError("top_k must be in (0, slots]")
        self.keys = nn.Parameter(
            torch.randn(self.config.slots, d_model) * 0.02
        )
        self.values = nn.Parameter(torch.zeros(self.config.slots, d_model))
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.write_gate = nn.Linear(d_model, 1)

    def forward(
        self, hidden: torch.Tensor, *, write: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden.ndim != 2:
            raise ValueError("hidden must be [batch, d_model]")
        query = F.normalize(self.query(hidden), dim=-1)
        keys = F.normalize(self.keys, dim=-1)
        scores = query @ keys.T
        top_scores, top_indices = torch.topk(
            scores, self.config.top_k, dim=-1
        )
        weights = torch.softmax(top_scores * 4.0, dim=-1)
        values = self.values[top_indices]
        broadcast = (weights.unsqueeze(-1) * values).sum(dim=1)
        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(-1)
        ignition = 1.0 - entropy / math.log(self.config.top_k)
        with torch.no_grad():
            if write:
                gate = torch.sigmoid(self.write_gate(hidden)).squeeze(-1)
                for row in range(hidden.shape[0]):
                    self.values[top_indices[row, 0]] = (
                        (1.0 - gate[row]) * self.values[top_indices[row, 0]]
                        + gate[row] * hidden[row]
                    )
        return broadcast, {
            "ignition": ignition,
            "weights": weights,
            "indices": top_indices,
            "slot_usage": (self.values.abs().sum(-1) > 0).float().mean(),
        }


class SelfMonitor(nn.Module):
    """Confidence, agency and error heads plus ECE."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.confidence = nn.Linear(d_model, 1)
        self.agency = nn.Linear(d_model, 1)
        self.error = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "confidence": torch.sigmoid(self.confidence(hidden)).squeeze(-1),
            "agency": torch.sigmoid(self.agency(hidden)).squeeze(-1),
            "error": torch.sigmoid(self.error(hidden)).squeeze(-1),
        }


def expected_calibration_error(
    confidence: torch.Tensor,
    correct: torch.Tensor,
    bins: int = 10,
) -> float:
    confidence = confidence.detach().float().flatten()
    correct = correct.detach().float().flatten()
    total = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (confidence >= low) & (
            confidence < high if index < bins - 1 else confidence <= high
        )
        if not bool(mask.any()):
            continue
        total += float(mask.float().mean()) * abs(
            float(confidence[mask].mean()) - float(correct[mask].mean())
        )
    return total


class HippocampalMemory(nn.Module):
    """Sparse one-shot episodic memory with pattern separation/completion."""

    def __init__(
        self,
        d_model: int,
        slots: int = 64,
        sparse_k: int = 8,
    ) -> None:
        super().__init__()
        if not 0 < sparse_k < slots:
            raise ValueError("sparse_k must be in (0, slots)")
        self.d_model = d_model
        self.slots = slots
        self.sparse_k = sparse_k
        projection = torch.randn(d_model, slots)
        self.register_buffer("projection", projection)
        self.register_buffer("keys", torch.zeros(slots, slots))
        self.register_buffer("values", torch.zeros(slots, d_model))
        self.register_buffer("used", torch.zeros(slots, dtype=torch.bool))
        self.register_buffer("usage", torch.zeros(slots))

    def separate(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = hidden @ self.projection
        threshold = torch.topk(projected, self.sparse_k, dim=-1).values[:, -1:]
        return projected * torch.sigmoid((projected - threshold) / 0.05)

    @torch.no_grad()
    def write(self, hidden: torch.Tensor) -> None:
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        separated = self.separate(hidden)
        for row in range(hidden.shape[0]):
            candidates = (~self.used).nonzero(as_tuple=False).flatten()
            if candidates.numel() == 0:
                slot = int(torch.argmin(self.usage))
            else:
                slot = int(candidates[0])
            self.keys[slot] = separated[row]
            self.values[slot] = hidden[row]
            self.used[slot] = True
            self.usage[slot] += 1

    def read(
        self, hidden: torch.Tensor, *, top_k: int = 4, complete: bool = True
    ) -> torch.Tensor:
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        separated = self.separate(hidden)
        keys = F.normalize(self.keys, dim=-1)
        query = F.normalize(separated, dim=-1)
        scores = query @ keys.T
        scores = scores.masked_fill(~self.used.unsqueeze(0), -1e4)
        top_scores, top_indices = torch.topk(scores, min(top_k, self.slots), dim=-1)
        weights = torch.softmax(top_scores * 5.0, dim=-1)
        retrieved = (
            weights.unsqueeze(-1) * self.values[top_indices]
        ).sum(dim=1)
        if complete:
            second = self.read(retrieved, top_k=top_k, complete=False)
            retrieved = F.normalize(retrieved + 0.5 * second, dim=-1) * (
                retrieved.norm(dim=-1, keepdim=True) + 1e-6
            )
        return retrieved

    def replay(self, count: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.used.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            return self.keys[:0], self.values[:0]
        selected = indices[
            torch.randint(
                0,
                indices.numel(),
                (min(count, indices.numel()),),
                device=indices.device,
            )
        ]
        return self.keys[selected], self.values[selected]


def should_ask(confidence: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Active-inquiry policy: ask when confidence is below threshold."""

    return confidence < threshold
