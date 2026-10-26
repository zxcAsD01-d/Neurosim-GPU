from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Dynamics:
    """Per-neuron parameters for the synchronous discrete-time model."""

    alpha: torch.Tensor
    gain: torch.Tensor

    def __post_init__(self) -> None:
        if self.alpha.ndim != 1 or self.gain.ndim != 1:
            raise ValueError("alpha and gain must be rank-1 tensors")
        if self.alpha.shape != self.gain.shape:
            raise ValueError("alpha and gain must have the same shape")
        if torch.any((self.alpha < 0) | (self.alpha >= 1)):
            raise ValueError("alpha must satisfy 0 <= alpha < 1")
        if torch.any(self.gain < 0):
            raise ValueError("gain must be non-negative")

    @property
    def n_neurons(self) -> int:
        return self.alpha.numel()

    @property
    def beta(self) -> torch.Tensor:
        return 1.0 - self.alpha

    def advance(
        self,
        state: torch.Tensor,
        synaptic_current: torch.Tensor,
        drive: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        return self.advance_modulated(state, synaptic_current, drive)

    def advance_modulated(
        self,
        state: torch.Tensor,
        synaptic_current: torch.Tensor,
        drive: torch.Tensor | float = 0.0,
        *,
        drive_shift: torch.Tensor | float = 0.0,
        gain_scale: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Advance with externally supplied receptor or chemical modulation."""

        equilibrium = (
            self.gain
            * gain_scale
            * torch.tanh(synaptic_current + drive + drive_shift)
        )
        return torch.clamp(
            self.alpha * state + self.beta * equilibrium,
            min=0.0,
            max=1.0,
        )
