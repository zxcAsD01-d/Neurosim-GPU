"""Optional dendritic filtering and activity adaptation for the rate core.

This is a dimensionless, research-inspired surrogate, not a conductance or
spiking model. Parameters are per neuron; the two extra states are per sample.
See docs/neuron-research-2026.md for equations, evidence and limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .dynamics import Dynamics
from .runtime import HybridPELSRuntime, MatrixRuntime, PropagationMode, RuntimeStats
from .topology import SparseTopology


@dataclass(frozen=True)
class DendriticDynamics:
    """One signed dendritic compartment and a slow negative-feedback trace.

    ``soma`` retains the original point-neuron dynamics. ``coupling`` in [0, 1]
    interpolates direct recurrent current and filtered dendritic activity.
    Zero coupling AND zero adaptation strength recover the original model.
    Decays are retention factors per tick, not time constants in milliseconds.
    """

    soma: Dynamics
    dendrite_decay: torch.Tensor
    adaptation_decay: torch.Tensor
    coupling: torch.Tensor
    adaptation_strength: torch.Tensor

    def __post_init__(self) -> None:
        reference = self.soma.alpha
        for name, value in (
            ("soma.alpha", reference), ("soma.gain", self.soma.gain),
            ("dendrite_decay", self.dendrite_decay),
            ("adaptation_decay", self.adaptation_decay),
            ("coupling", self.coupling),
            ("adaptation_strength", self.adaptation_strength),
        ):
            if value.shape != reference.shape or value.ndim != 1:
                raise ValueError(f"{name} must have one value per neuron")
            if not value.is_floating_point():
                raise ValueError(f"{name} must be floating point")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match soma device and dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        for name in ("dendrite_decay", "adaptation_decay"):
            value = getattr(self, name)
            if bool(((value < 0) | (value >= 1)).any()):
                raise ValueError(f"{name} must be in [0, 1)")
        if bool(((self.coupling < 0) | (self.coupling > 1)).any()):
            raise ValueError("coupling must be in [0, 1]")
        if bool((self.adaptation_strength < 0).any()):
            raise ValueError("adaptation_strength must be non-negative")

    @classmethod
    def from_time_constants(
        cls,
        soma: Dynamics,
        *,
        dt: float,
        tau_dendrite: torch.Tensor | float,
        tau_adaptation: torch.Tensor | float,
        coupling: torch.Tensor | float = 0.5,
        adaptation_strength: torch.Tensor | float = 0.2,
    ) -> DendriticDynamics:
        """Use exp(-dt/tau), with dt and taus in the same chosen time unit.

        Scalars broadcast to N; vectors allow heterogeneous time constants.
        Defaults are illustrative, not fitted to experimental recordings.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")

        def vector(value: torch.Tensor | float, name: str) -> torch.Tensor:
            result = torch.as_tensor(
                value, dtype=soma.alpha.dtype, device=soma.alpha.device
            )
            if result.ndim == 0:
                result = result.expand_as(soma.alpha)
            if result.shape != soma.alpha.shape:
                raise ValueError(f"{name} must be scalar or have one value per neuron")
            return result

        def decay(value: torch.Tensor | float, name: str) -> torch.Tensor:
            tau = vector(value, name)
            if not bool((torch.isfinite(tau) & (tau > 0)).all()):
                raise ValueError(f"{name} must be finite and positive")
            return torch.exp(-dt / tau)

        return cls(
            soma,
            decay(tau_dendrite, "tau_dendrite"),
            decay(tau_adaptation, "tau_adaptation"),
            vector(coupling, "coupling"),
            vector(adaptation_strength, "adaptation_strength"),
        )

    def advance(
        self,
        state: torch.Tensor,
        dendritic_state: torch.Tensor,
        adaptation_state: torch.Tensor,
        synaptic_current: torch.Tensor,
        drive: torch.Tensor | float = 0.0,
        *,
        dendritic_drive: torch.Tensor | float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure differentiable update; no in-place writes or host synchronization.

        External ``drive`` reaches the soma directly; ``dendritic_drive`` joins
        recurrent input before local saturation. Feedback uses a_t, then tracks
        the newly computed s_(t+1). Inputs must be finite and broadcast to [B,N].
        """
        dendrite = (
            self.dendrite_decay * dendritic_state
            + (1.0 - self.dendrite_decay)
            * torch.tanh(synaptic_current + dendritic_drive)
        )
        current = (
            (1.0 - self.coupling) * synaptic_current
            + self.coupling * dendrite
            - self.adaptation_strength * adaptation_state
        )
        state = self.soma.advance(state, current, drive)
        adaptation = (
            self.adaptation_decay * adaptation_state
            + (1.0 - self.adaptation_decay) * state
        )
        return state, dendrite, adaptation


class _CompartmentState:
    """State ownership shared by the portable matrix and persistent runtimes."""

    state: torch.Tensor
    compartment_dynamics: DendriticDynamics
    dendritic_state: torch.Tensor
    adaptation_state: torch.Tensor

    def _init_compartments(
        self, topology: SparseTopology, dynamics: DendriticDynamics,
        initial_state: torch.Tensor,
    ) -> None:
        if initial_state.ndim != 2 or initial_state.shape[1] != topology.n_neurons:
            raise ValueError("initial_state must have shape [batch, n_neurons]")
        if dynamics.soma.n_neurons != topology.n_neurons:
            raise ValueError("dynamics and topology neuron counts must match")
        for value in (dynamics.soma.alpha, topology.weight):
            if value.device != initial_state.device or value.dtype != initial_state.dtype:
                raise ValueError("state, topology and dynamics must match device and dtype")
        if not bool((torch.isfinite(initial_state)
                     & (initial_state >= 0) & (initial_state <= 1)).all()):
            raise ValueError("initial_state must be finite and in [0, 1]")
        self.compartment_dynamics = dynamics
        self.dendritic_state = torch.zeros_like(initial_state)
        self.adaptation_state = torch.zeros_like(initial_state)

    def _advance_compartments(
        self, current: torch.Tensor, drive: torch.Tensor | float,
        dendritic_drive: torch.Tensor | float,
    ) -> torch.Tensor:
        for name, value in (("drive", drive), ("dendritic_drive", dendritic_drive)):
            if isinstance(value, torch.Tensor):
                try:
                    shape = torch.broadcast_shapes(value.shape, self.state.shape)
                except RuntimeError as error:
                    raise ValueError(f"{name} must broadcast to [batch, n_neurons]") from error
                if shape != self.state.shape:
                    raise ValueError(f"{name} must broadcast to [batch, n_neurons]")
        state, dendrite, adaptation = self.compartment_dynamics.advance(
            self.state, self.dendritic_state, self.adaptation_state,
            current, drive, dendritic_drive=dendritic_drive,
        )
        self.dendritic_state = dendrite
        self.adaptation_state = adaptation
        return state


class DendriticMatrixRuntime(_CompartmentState, MatrixRuntime):
    """One CSR SpMM and O(B*N) work; two additional [B,N] state tensors.

    Construction resets both traces to zero. Use the checkpoint API to resume
    all three states. ``dynamics`` is the soma; ``compartment_dynamics`` owns
    the complete parameter set. No native fused kernel is selected implicitly.
    """

    def __init__(self, topology: SparseTopology, dynamics: DendriticDynamics,
                 initial_state: torch.Tensor) -> None:
        self._init_compartments(topology, dynamics, initial_state)
        super().__init__(topology, dynamics.soma, initial_state)

    def step(self, drive: torch.Tensor | float = 0.0, *,
             dendritic_drive: torch.Tensor | float = 0.0) -> torch.Tensor:
        self.state = self._advance_compartments(
            self._propagate(self.state), drive, dendritic_drive
        )
        return self.state


class DendriticPELSRuntime(_CompartmentState, HybridPELSRuntime):
    """PELS propagation with local compartment updates on EVERY tick.

    theta=0 matches DendriticMatrixRuntime; only somatic activity is emitted.
    Nonzero theta bounds the source residual, not the accumulated trajectory
    error. The original point-model uniform_error_budget is not a bound for
    this augmented dynamical system. Event propagation is the portable path.
    """

    def __init__(
        self, topology: SparseTopology, dynamics: DendriticDynamics,
        initial_state: torch.Tensor, *, theta: float | torch.Tensor,
        propagation: PropagationMode = "auto", event_edge_fraction: float = 0.12,
        policy_interval: int = 16,
    ) -> None:
        self._init_compartments(topology, dynamics, initial_state)
        super().__init__(
            topology, dynamics.soma, initial_state, theta=theta,
            propagation=propagation, event_edge_fraction=event_edge_fraction,
            policy_interval=policy_interval,
        )

    def step(
        self, drive: torch.Tensor | float = 0.0, *,
        dendritic_drive: torch.Tensor | float = 0.0, collect_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, RuntimeStats]:
        self._refresh_current()
        state = self._advance_compartments(
            self.persistent_current, drive, dendritic_drive
        )
        return self._emit(state, collect_stats=collect_stats)
