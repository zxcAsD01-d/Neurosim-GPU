from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .dynamics import Dynamics
from .topology import SparseTopology


PropagationMode = Literal["auto", "matrix", "event"]


def _validate_emission_threshold(
    theta: torch.Tensor,
    topology: SparseTopology,
) -> torch.Tensor:
    if theta.ndim == 0:
        theta = theta.expand(topology.n_neurons)
    if theta.shape != (topology.n_neurons,):
        raise ValueError("theta must be scalar or have one value per neuron")
    if bool(torch.any(theta < 0)) or bool(torch.any(torch.isnan(theta))):
        raise ValueError(
            "theta must be non-negative and finite, except +inf for "
            "zero-outdegree neurons"
        )
    positive_infinity = torch.isposinf(theta)
    invalid_infinity = positive_infinity & (topology.out_degree > 0)
    if bool(torch.any(torch.isneginf(theta))) or bool(torch.any(invalid_infinity)):
        raise ValueError(
            "infinite theta is allowed only for zero-outdegree neurons"
        )
    return theta


@dataclass(frozen=True)
class RuntimeStats:
    path: str
    active_neurons: int
    active_edges: int


class DenseOracle:
    """Small-network FP64/FP32 semantic oracle."""

    def __init__(
        self,
        topology: SparseTopology,
        dynamics: Dynamics,
        initial_state: torch.Tensor,
    ) -> None:
        self.topology = topology
        self.dynamics = dynamics
        self.weight = topology.dense_matrix()
        self.state = initial_state

    def step(self, drive: torch.Tensor | float = 0.0) -> torch.Tensor:
        synaptic_current = self.state @ self.weight.T
        self.state = self.dynamics.advance(self.state, synaptic_current, drive)
        return self.state


class MatrixRuntime:
    """GPU-first baseline: one sparse matrix multiplication plus O(B*N)."""

    def __init__(
        self,
        topology: SparseTopology,
        dynamics: Dynamics,
        initial_state: torch.Tensor,
    ) -> None:
        self.topology = topology
        self.dynamics = dynamics
        self.weight = topology.sparse_matrix()
        self.state = initial_state

    def _propagate(self, source: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.weight, source.T).T

    def step(self, drive: torch.Tensor | float = 0.0) -> torch.Tensor:
        synaptic_current = self._propagate(self.state)
        self.state = self.dynamics.advance(self.state, synaptic_current, drive)
        return self.state


class HybridPELSRuntime(MatrixRuntime):
    """Persistent synaptic current with matrix/event propagation switching.

    The state at entry is s_t and persistent_current represents W @ s_emit.
    A step first computes s_(t+1), then emits selected state deltas so the
    persistent current is ready for the next tick. Scheduler decisions are
    detached from autograd; selected values remain differentiable.
    """

    def __init__(
        self,
        topology: SparseTopology,
        dynamics: Dynamics,
        initial_state: torch.Tensor,
        *,
        theta: float | torch.Tensor,
        propagation: PropagationMode = "auto",
        event_edge_fraction: float = 0.12,
        policy_interval: int = 16,
    ) -> None:
        super().__init__(topology, dynamics, initial_state)
        if propagation not in ("auto", "matrix", "event"):
            raise ValueError("propagation must be auto, matrix or event")
        if not 0.0 <= event_edge_fraction <= 1.0:
            raise ValueError("event_edge_fraction must be in [0, 1]")
        if policy_interval <= 0:
            raise ValueError("policy_interval must be positive")
        self.theta = torch.as_tensor(
            theta,
            dtype=initial_state.dtype,
            device=initial_state.device,
        )
        self.theta = _validate_emission_threshold(self.theta, topology)
        self.propagation = propagation
        self.event_edge_fraction = event_edge_fraction
        self.policy_interval = policy_interval
        self.emitted_state = initial_state.clone()
        self.persistent_current = self._propagate(self.emitted_state)
        self._tick = 0
        self._selected_path = "matrix"
        self._weight_version = topology.weight_version

    def _select_path(
        self,
        active: torch.Tensor,
        collect_stats: bool,
    ) -> tuple[str, int]:
        # The common matrix path must stay asynchronous. Counting active edges
        # requires a device-to-host synchronization, so only do it for an
        # explicit statistics request or at an auto-policy sampling tick.
        if self.propagation == "matrix" and not collect_stats:
            return "matrix", -1
        policy_tick = self._tick % self.policy_interval == 0
        if (
            self.propagation == "auto"
            and not policy_tick
            and not collect_stats
        ):
            # Keep the selected path until the next policy tick. Counting
            # activity uses a device-to-host scalar and must not happen on
            # every event-path step.
            return self._selected_path, -1

        active_pairs = torch.nonzero(active, as_tuple=False)
        if active_pairs.numel() == 0:
            if self.propagation == "auto":
                self._selected_path = "event"
            return "idle", 0
        source = active_pairs[:, 1]
        active_edges = int(self.topology.out_degree[source].sum().item())
        if self.propagation != "auto":
            return self.propagation, active_edges
        if policy_tick:
            total = self.topology.n_edges * self.state.shape[0]
            fraction = active_edges / max(total, 1)
            self._selected_path = (
                "event" if fraction < self.event_edge_fraction else "matrix"
            )
        return self._selected_path, active_edges

    def _event_propagate(
        self,
        emitted_delta: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        active_pairs = torch.nonzero(active, as_tuple=False)
        batch = active_pairs[:, 0]
        source = active_pairs[:, 1]
        counts = self.topology.out_degree[source]

        pair_id = torch.repeat_interleave(
            torch.arange(source.numel(), device=source.device),
            counts,
        )
        repeated_start = torch.repeat_interleave(
            torch.cumsum(counts, dim=0) - counts,
            counts,
        )
        local_offset = torch.arange(
            int(counts.sum().item()),
            device=source.device,
        ) - repeated_start
        edge_id = self.topology.out_rowptr[source[pair_id]] + local_offset

        flat_destination = (
            batch[pair_id] * self.topology.n_neurons
            + self.topology.dst[edge_id]
        )
        contribution = (
            emitted_delta[batch[pair_id], source[pair_id]]
            * self.topology.weight[edge_id]
        )
        result = torch.zeros_like(self.persistent_current).flatten()
        result.index_add_(0, flat_destination, contribution)
        return result.view_as(self.persistent_current)

    def step(
        self,
        drive: torch.Tensor | float = 0.0,
        *,
        collect_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, RuntimeStats]:
        if self._weight_version != self.topology.weight_version:
            self.persistent_current = self._propagate(
                self.emitted_state
            ).contiguous()
            self._weight_version = self.topology.weight_version
        next_state = self.dynamics.advance(
            self.state,
            self.persistent_current,
            drive,
        )
        residual = next_state - self.emitted_state
        active = residual.detach().abs() >= self.theta
        emitted_delta = torch.where(active, residual, torch.zeros_like(residual))

        path, active_edges = self._select_path(active, collect_stats)
        if path == "matrix":
            current_delta = self._propagate(emitted_delta)
        elif path == "event":
            current_delta = self._event_propagate(emitted_delta, active)
        else:
            current_delta = torch.zeros_like(self.persistent_current)

        self.persistent_current = self.persistent_current + current_delta
        self.emitted_state = self.emitted_state + emitted_delta
        self.state = next_state
        self._tick += 1

        if not collect_stats:
            return self.state
        stats = RuntimeStats(
            path=path,
            active_neurons=int(active.sum().item()),
            active_edges=active_edges,
        )
        return self.state, stats
