"""Joint objective for sequential, time-aware Cortex-LM training.

The language loss is only one term. The joint function combines:

* next-token cross entropy;
* a time-perception head (predict normalized position in the sequence);
* a question-tendency head (predict whether the next token is a question mark);
* a compute cost (internal ticks / budget), used by the time curriculum.

This module does not change the checkpoint layout; the two small heads are
trained alongside the existing Cortex-LM parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class JointConfig:
    time_weight: float = 0.2
    question_weight: float = 0.2
    compute_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("time_weight", "question_weight", "compute_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


class JointObjective(nn.Module):
    """Small heads plus the weighted joint loss."""

    def __init__(
        self,
        d_model: int,
        question_token_ids: tuple[int, ...] = (),
        config: JointConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or JointConfig()
        self.time_head = nn.Linear(d_model, 1)
        self.question_head = nn.Linear(d_model, 1)
        self.register_buffer(
            "question_ids",
            torch.tensor(question_token_ids, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        readout: torch.Tensor,
        lm_loss: torch.Tensor,
        labels: torch.Tensor,
        position: float,
        ticks: int,
        max_ticks: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if readout.ndim == 3:
            readout = readout[:, -1]
        time_prediction = torch.sigmoid(self.time_head(readout)).squeeze(-1)
        time_target = torch.full_like(time_prediction, float(position))
        time_loss = F.mse_loss(time_prediction, time_target)

        question_logits = self.question_head(readout).squeeze(-1)
        if self.question_ids.numel():
            is_question = torch.isin(labels, self.question_ids)
            question_target = is_question.to(question_logits.dtype)
            question_loss = F.binary_cross_entropy_with_logits(
                question_logits, question_target
            )
            question_accuracy = (
                (question_logits > 0) == is_question
            ).float().mean()
        else:
            question_loss = torch.zeros((), device=readout.device)
            question_accuracy = torch.zeros((), device=readout.device)

        compute = torch.tensor(
            ticks / max(max_ticks, 1),
            device=readout.device,
            dtype=readout.dtype,
        )
        total = (
            lm_loss
            + self.config.time_weight * time_loss
            + self.config.question_weight * question_loss
            + self.config.compute_weight * compute
        )
        return total, {
            "lm": lm_loss.detach(),
            "time": time_loss.detach(),
            "question": question_loss.detach(),
            "question_accuracy": question_accuracy.detach(),
            "compute": compute.detach(),
            "time_mae": (time_prediction - time_target).abs().mean().detach(),
        }
