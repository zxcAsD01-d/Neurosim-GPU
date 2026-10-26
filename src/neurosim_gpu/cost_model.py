"""Pure-Python asymptotic and byte-cost model with no CUDA allocation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostModel:
    n_neurons: int
    n_edges: int
    batch: int = 1
    n_regions: int = 1
    n_sources: int = 1
    region_edges: int = 0
    chemistry_interval: int = 1
    plasticity_interval: int = 1
    max_dopamine_delay: int = 0
    max_axon_delay: int = 0
    communication_channels: int = 1
    active_edge_fraction: float = 1.0
    candidate_edges: int = 0
    migration_interval: int = 0
    structural_interval: int = 0
    dtype_bytes: int = 4
    include_communication: bool = False
    include_eligibility: bool = False
    include_plasticity: bool = False
    include_development: bool = False

    def __post_init__(self) -> None:
        for name in (
            "n_neurons", "n_edges", "batch", "n_regions", "n_sources", "region_edges",
            "chemistry_interval", "plasticity_interval", "max_dopamine_delay",
            "max_axon_delay", "communication_channels", "candidate_edges",
            "migration_interval", "structural_interval", "dtype_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.n_neurons <= 0 or self.batch <= 0:
            raise ValueError("n_neurons and batch must be positive")
        if self.n_regions <= 0 or self.n_sources <= 0:
            raise ValueError("n_regions and n_sources must be positive")
        if self.chemistry_interval <= 0 or self.plasticity_interval <= 0:
            raise ValueError("intervals must be positive")
        if not 0.0 <= self.active_edge_fraction <= 1.0:
            raise ValueError("active_edge_fraction must lie in [0, 1]")

    @property
    def active_edges(self) -> float:
        return self.n_edges * self.active_edge_fraction

    @property
    def base_work(self) -> float:
        return self.batch * (self.n_edges + self.n_neurons)

    @property
    def event_work(self) -> float:
        return self.batch * self.n_neurons + self.batch * self.active_edges

    @property
    def chemistry_work_per_neural_tick(self) -> float:
        boundary = self.batch * (
            2 * self.n_regions + 2 * self.region_edges
            + self.n_sources * self.n_regions + self.n_neurons
        )
        return boundary / self.chemistry_interval

    @property
    def eligibility_work(self) -> float:
        return self.batch * (self.n_edges + 2 * self.n_neurons)

    @property
    def plasticity_work_per_neural_tick(self) -> float:
        return self.n_edges / (self.chemistry_interval * self.plasticity_interval)

    @property
    def persistent_resync_work_per_neural_tick(self) -> float:
        return self.batch * self.n_edges / (self.chemistry_interval * self.plasticity_interval)

    @property
    def communication_work(self) -> float:
        return self.batch * (self.n_edges + self.n_neurons)

    @property
    def migration_work_per_neural_tick(self) -> float:
        return self.n_neurons / self.migration_interval if self.migration_interval else 0.0

    @property
    def structural_work_per_neural_tick(self) -> float:
        if not self.structural_interval:
            return 0.0
        mlog = self.n_edges * math.log2(max(self.n_edges, 2))
        klog = self.candidate_edges * math.log2(max(self.candidate_edges, 2))
        return (self.n_neurons + mlog + klog) / self.structural_interval

    @property
    def full_matrix_work(self) -> float:
        return self._combined_work(self.base_work)

    @property
    def full_event_work(self) -> float:
        return self._combined_work(self.event_work)

    def _combined_work(self, propagation: float) -> float:
        work = propagation + self.chemistry_work_per_neural_tick
        if self.include_eligibility:
            work += self.eligibility_work
        if self.include_plasticity:
            work += self.plasticity_work_per_neural_tick
            work += self.persistent_resync_work_per_neural_tick
        if self.include_communication:
            work += self.communication_work
        if self.include_development:
            work += self.migration_work_per_neural_tick
            work += self.structural_work_per_neural_tick
        return work

    @property
    def slim_inference_bytes(self) -> int:
        return self.dtype_bytes * self.n_edges + 4 * (
            3 * self.n_edges + 2 * (self.n_neurons + 1)
        )

    @property
    def mutable_online_bytes(self) -> int:
        # Add a mutable source-major FP32/typed weight plus the source index
        # view that is absent from the slim destination-major inference layout.
        return self.slim_inference_bytes + (self.dtype_bytes + 4) * self.n_edges

    @property
    def estimated_persistent_bytes(self) -> int:
        """Selected-state lower-bound estimate, not a live allocator peak.

        Receptor signal tensors, RPE accumulators, event workspaces, candidate
        buffers, and allocator/checkpoint transients are intentionally omitted.
        """
        neural = self.batch * self.n_neurons * self.dtype_bytes * 6
        chemistry = self.batch * self.n_regions * self.dtype_bytes * 2
        total = self.mutable_online_bytes + neural + chemistry
        if self.include_eligibility:
            total += self.batch * self.n_neurons * self.dtype_bytes * 2
            total += self.n_edges * self.dtype_bytes
        if self.include_communication:
            total += self.batch * self.n_neurons * self.dtype_bytes * (self.max_axon_delay + 1)
            total += self.n_edges * (4 * 4 + 2 * self.dtype_bytes)
            total += self.n_neurons * self.communication_channels * self.dtype_bytes
        if self.max_dopamine_delay:
            total += self.batch * self.n_sources * self.dtype_bytes * self.max_dopamine_delay
        return total

    def as_dict(self) -> dict[str, float | int]:
        values = asdict(self)
        values.update({
            "active_edges": self.active_edges,
            "base_work": self.base_work,
            "event_work": self.event_work,
            "chemistry_work_per_neural_tick": self.chemistry_work_per_neural_tick,
            "eligibility_work": self.eligibility_work,
            "full_matrix_work": self.full_matrix_work,
            "full_event_work": self.full_event_work,
            "event_to_matrix_work_ratio": self.full_event_work / max(self.full_matrix_work, 1.0),
            "slim_inference_bytes": self.slim_inference_bytes,
            "mutable_online_bytes": self.mutable_online_bytes,
            "estimated_persistent_bytes": self.estimated_persistent_bytes,
        })
        return values


def estimate_cost(**kwargs: int | float) -> dict[str, float | int]:
    return CostModel(**kwargs).as_dict()


__all__ = ["CostModel", "estimate_cost"]
