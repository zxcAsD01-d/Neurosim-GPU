"""Regional sparse topology for the cortical sheet.

The graph is fixed for the whole run. It has three properties the language
model relies on:

* **Regions.** Neurons are split into contiguous regions; a configurable
  fraction of synapses stays inside a region, the rest is long-range.
* **Dale's law.** A neuron is either excitatory or inhibitory, and every
  outgoing synapse carries the same sign.
* **Balanced input.** For every destination neuron the incoming weights are
  rescaled to a fixed L1 norm. This keeps the recurrent current inside the
  ``tanh`` range independently of degree, so a wider sheet does not silently
  become a saturating one.

Edge weights are generated once from a CPU generator and then moved to the
target device. Rebuilding the same config therefore gives the same graph, which
is what checkpoint resume and the tests assume.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..topology import SparseTopology
from .config import BrainConfig


@dataclass(frozen=True)
class CorticalTopology:
    """A sparse graph plus the per-neuron state attached to it."""

    graph: SparseTopology
    region: torch.Tensor
    sign: torch.Tensor
    alpha: torch.Tensor
    base_gain: torch.Tensor
    n_regions: int

    @property
    def n_neurons(self) -> int:
        return self.graph.n_neurons

    @property
    def n_edges(self) -> int:
        return self.graph.n_edges

    def describe(self) -> dict[str, object]:
        return {
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "edges_per_neuron": self.n_edges / max(self.n_neurons, 1),
            "n_regions": self.n_regions,
            "inhibitory_fraction": float((self.sign < 0).float().mean()),
            "alpha_mean": float(self.alpha.mean()),
            "gain_mean": float(self.base_gain.mean()),
        }


def _move_graph(
    graph: SparseTopology,
    device: torch.device | str,
    dtype: torch.dtype,
) -> SparseTopology:
    return SparseTopology(
        n_neurons=graph.n_neurons,
        src=graph.src.to(device),
        dst=graph.dst.to(device),
        weight=graph.weight.to(device=device, dtype=dtype),
        out_rowptr=graph.out_rowptr.to(device),
        in_rowptr=graph.in_rowptr.to(device),
        in_col=graph.in_col.to(device),
        out_weight_index=graph.out_weight_index.to(device),
        in_weight=graph.in_weight.to(device=device, dtype=dtype),
    )


def build_cortical_topology(
    config: BrainConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CorticalTopology:
    """Build the deterministic regional graph described by ``config``."""

    n = config.n_neurons
    regions = config.regions
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    # Contiguous, almost equal regions.
    region = torch.div(
        torch.arange(n, dtype=torch.int64) * regions,
        n,
        rounding_mode="floor",
    )
    region_sizes = torch.bincount(region, minlength=regions).tolist()
    region_start = torch.cumsum(
        torch.tensor(region_sizes, dtype=torch.int64), dim=0
    ) - torch.tensor(region_sizes, dtype=torch.int64)

    # Dale's law: pick inhibitory neurons independently inside each region.
    sign = torch.ones(n, dtype=torch.float32)
    for r in range(regions):
        start = int(region_start[r])
        size = int(region_sizes[r])
        n_inhibitory = int(round(size * config.inhibitory_fraction))
        if n_inhibitory:
            order = torch.randperm(size, generator=generator) + start
            sign[order[:n_inhibitory]] = -1.0

    n_edges = config.n_edges
    n_local = int(round(n_edges * config.local_fraction))
    n_global = n_edges - n_local

    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    for r in range(regions):
        start = int(region_start[r])
        size = int(region_sizes[r])
        count = n_local // regions + (1 if r < n_local % regions else 0)
        if count == 0:
            continue
        src = start + torch.randint(size, (count,), generator=generator)
        dst = start + torch.randint(size, (count,), generator=generator)
        if size > 1:
            duplicate = dst == src
            dst = torch.where(
                duplicate,
                start + (dst - start + 1).remainder(size),
                dst,
            )
        src_parts.append(src)
        dst_parts.append(dst)

    if n_global:
        src = torch.randint(n, (n_global,), generator=generator)
        dst = torch.randint(n, (n_global,), generator=generator)
        if n > 1:
            duplicate = dst == src
            dst = torch.where(duplicate, (dst + 1).remainder(n), dst)
        src_parts.append(src)
        dst_parts.append(dst)

    src = torch.cat(src_parts) if src_parts else torch.empty(0, dtype=torch.int64)
    dst = torch.cat(dst_parts) if dst_parts else torch.empty(0, dtype=torch.int64)

    # Log-normal magnitudes, then per-destination L1 normalisation.
    magnitude = torch.exp(
        torch.randn(n_edges, generator=generator) * 0.35
    ) * config.weight_scale
    weight = sign[src] * magnitude
    incoming = torch.zeros(n, dtype=torch.float32).index_add_(
        0, dst, weight.abs()
    )
    weight = weight * (config.incoming_norm / incoming.clamp_min(1e-6))[dst]

    graph = SparseTopology.from_edges(n, src, dst, weight)

    alpha = config.alpha_min + (config.alpha_max - config.alpha_min) * torch.rand(
        n, generator=generator
    )
    base_gain = config.gain * (
        0.8 + 0.4 * torch.rand(n, generator=generator)
    )

    graph = _move_graph(graph, device, dtype)
    return CorticalTopology(
        graph=graph,
        region=region.to(device),
        sign=sign.to(device=device, dtype=dtype),
        alpha=alpha.to(device=device, dtype=dtype),
        base_gain=base_gain.to(device=device, dtype=dtype),
        n_regions=regions,
    )
