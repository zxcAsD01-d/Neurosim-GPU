"""Explicit multi-rate orchestration for real-time and continual simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .checkpoint import capture_runtime, restore_runtime
from .communication import DelayedMultiChannelCommunication
from .topology import SparseTopology


MigrationCallback = Callable[["MultiRateScheduler"], Any]
StructuralCallback = Callable[["MultiRateScheduler"], SparseTopology | None]
CommunicationRebuilder = Callable[
    ["MultiRateScheduler", SparseTopology],
    DelayedMultiChannelCommunication | None,
]


@dataclass(frozen=True)
class SchedulerConfig:
    migration_interval: int = 0
    structural_interval: int = 0

    def __post_init__(self) -> None:
        if self.migration_interval < 0 or self.structural_interval < 0:
            raise ValueError("scheduler intervals must be non-negative")


@dataclass(frozen=True)
class SchedulerStep:
    tick: int
    migration_applied: bool
    structural_applied: bool
    communication_applied: bool


class MultiRateScheduler:
    """Compose a stateful runtime with explicit slow-time boundaries.

    The scheduler treats communication as an auxiliary current added to the
    runtime drive. Structural callbacks return a complete replacement topology;
    partial in-place graph mutation is rejected by the replacement transaction.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        communication: DelayedMultiChannelCommunication | None = None,
        config: SchedulerConfig | None = None,
        migration_callback: MigrationCallback | None = None,
        structural_callback: StructuralCallback | None = None,
        communication_rebuilder: CommunicationRebuilder | None = None,
    ) -> None:
        if not hasattr(runtime, "state") or not hasattr(runtime, "topology"):
            raise ValueError("runtime must expose state and topology")
        if communication is not None and communication.n_neurons != runtime.state.shape[-1]:
            raise ValueError("communication neuron count must match runtime")
        self.runtime = runtime
        self.communication = communication
        self.config = config or SchedulerConfig()
        self.migration_callback = migration_callback
        self.structural_callback = structural_callback
        self.communication_rebuilder = communication_rebuilder
        self.tick = 0
        self.migration_epochs = 0
        self.structural_epochs = 0
        self._zero_rpe_cache: torch.Tensor | None = None

    @property
    def state(self) -> torch.Tensor:
        return self.runtime.state

    def _zero_rpe(self) -> torch.Tensor | None:
        chemistry = getattr(self.runtime, "chemistry", None)
        if chemistry is None:
            return None
        expected = (self.state.shape[0], chemistry.n_sources)
        if (
            self._zero_rpe_cache is None
            or self._zero_rpe_cache.shape != expected
            or self._zero_rpe_cache.device != self.state.device
            or self._zero_rpe_cache.dtype != self.state.dtype
        ):
            self._zero_rpe_cache = torch.zeros(
                *expected,
                dtype=self.state.dtype,
                device=self.state.device,
            )
        return self._zero_rpe_cache

    def _runtime_step(
        self,
        drive: torch.Tensor | float,
        rpe: torch.Tensor | None,
        source_release: torch.Tensor | None,
    ) -> torch.Tensor:
        if hasattr(self.runtime, "chemistry"):
            if rpe is None:
                rpe = self._zero_rpe()
            return self.runtime.step(
                drive,
                rpe=rpe,
                source_release=source_release,
            )
        return self.runtime.step(drive)

    def step(
        self,
        drive: torch.Tensor | float = 0.0,
        *,
        rpe: torch.Tensor | None = None,
        source_release: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SchedulerStep]:
        """Advance one neural tick and run due slow callbacks."""

        communication_applied = self.communication is not None
        if self.communication is not None:
            auxiliary_current = self.communication.step(self.state)
            drive = (
                auxiliary_current + drive
                if isinstance(drive, torch.Tensor)
                else auxiliary_current + drive
            )

        state = self._runtime_step(drive, rpe, source_release)
        self.tick += 1

        migration_applied = False
        if (
            self.migration_callback is not None
            and self.config.migration_interval > 0
            and self.tick % self.config.migration_interval == 0
        ):
            self.migration_callback(self)
            self.migration_epochs += 1
            migration_applied = True

        structural_applied = False
        if (
            self.structural_callback is not None
            and self.config.structural_interval > 0
            and self.tick % self.config.structural_interval == 0
        ):
            replacement = self.structural_callback(self)
            if replacement is not None:
                self.replace_topology(replacement)
                self.structural_epochs += 1
                structural_applied = True

        return state, SchedulerStep(
            tick=self.tick,
            migration_applied=migration_applied,
            structural_applied=structural_applied,
            communication_applied=communication_applied,
        )

    def replace_topology(self, replacement: SparseTopology) -> None:
        """Atomically replace a fixed-size topology at a structural boundary."""

        current = self.runtime.topology
        if replacement.n_neurons != current.n_neurons:
            raise ValueError("structural replacement must preserve neuron count")
        if replacement.weight.device != self.state.device:
            raise ValueError("structural replacement must use runtime device")
        if replacement.weight.dtype != self.state.dtype:
            raise ValueError("structural replacement dtype must match runtime")
        # Construct all derived objects before mutating the runtime.
        replacement_matrix = replacement.sparse_matrix()
        if self.communication is not None and self.communication.n_neurons != replacement.n_neurons:
            raise ValueError("communication graph is incompatible with replacement")
        if self.communication is not None and self.communication_rebuilder is None:
            raise ValueError(
                "structural replacement requires communication_rebuilder when "
                "a communication graph is attached"
            )

        replacement_communication = self.communication
        if self.communication_rebuilder is not None:
            replacement_communication = self.communication_rebuilder(
                self, replacement
            )
            if replacement_communication is not None and (
                replacement_communication.n_neurons != replacement.n_neurons
            ):
                raise ValueError(
                    "rebuilt communication graph neuron count differs"
                )

        runtime = self.runtime
        runtime.topology = replacement
        if hasattr(runtime, "weight"):
            runtime.weight = replacement_matrix
        if hasattr(runtime, "_src"):
            runtime._src = replacement.src.to(torch.int64)
            runtime._dst = replacement.dst.to(torch.int64)
        if hasattr(runtime, "eligibility"):
            runtime.eligibility = torch.zeros(
                replacement.n_edges,
                dtype=runtime.state.dtype,
                device=runtime.state.device,
            )
        if hasattr(runtime, "persistent_current"):
            runtime.persistent_current = torch.sparse.mm(
                replacement_matrix,
                runtime.emitted_state.T,
            ).T.contiguous()
            runtime._weight_version = replacement.weight_version
        if hasattr(runtime, "_weight_version"):
            runtime._weight_version = replacement.weight_version
        self.communication = replacement_communication

    def state_dict(self) -> dict[str, Any]:
        payload = capture_runtime(
            self.runtime,
            communication=self.communication,
        )
        payload["scheduler"] = {
            "tick": self.tick,
            "migration_epochs": self.migration_epochs,
            "structural_epochs": self.structural_epochs,
            "migration_interval": self.config.migration_interval,
            "structural_interval": self.config.structural_interval,
        }
        return payload

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        scheduler = payload.get("scheduler")
        if not isinstance(scheduler, dict):
            raise ValueError("checkpoint is missing scheduler state")
        if scheduler.get("migration_interval") != self.config.migration_interval:
            raise ValueError("migration interval differs from checkpoint")
        if scheduler.get("structural_interval") != self.config.structural_interval:
            raise ValueError("structural interval differs from checkpoint")
        restore_runtime(
            self.runtime,
            payload,
            communication=self.communication,
        )
        self.tick = int(scheduler["tick"])
        self.migration_epochs = int(scheduler["migration_epochs"])
        self.structural_epochs = int(scheduler["structural_epochs"])


__all__ = ["MultiRateScheduler", "SchedulerConfig", "SchedulerStep"]
