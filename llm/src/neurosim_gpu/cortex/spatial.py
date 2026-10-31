"""Spatial plate geometry and slow structural adaptation for Cortex-LM.

This is the first step towards the "two frozen plates" architecture: the
retained Qwen blocks are the front (input) and back (output) plates, separated
by a fixed distance of 50 nominal neuron spacings. The neuron sheet lives in
the slab between them.

The functions here are deliberately *offline* adaptation primitives. They take
the trained dense matrix, a sparse edge mask and per-neuron activity, then
return moved positions and a rewired mask. They do not modify the training
loop or the current long-run checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..development import (
    StructuralPlasticityConfig,
    migrate_neurons,
    score_candidate_edges,
)
from ..development import MigrationConfig


@dataclass(frozen=True)
class PlateGeometry:
    """Two plates separated by ``plate_distance`` nominal neuron spacings."""

    n_neurons: int
    mean_neuron_distance: float = 1.0
    plate_distance: float = 50.0

    def __post_init__(self) -> None:
        if self.n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        if self.mean_neuron_distance <= 0 or self.plate_distance <= 0:
            raise ValueError("distances must be positive")
        if self.plate_distance <= self.mean_neuron_distance:
            raise ValueError("plate distance must exceed neuron spacing")

    @property
    def side(self) -> float:
        """Square plate side for density ``1 / mean_distance**3``."""

        volume = self.n_neurons * self.mean_neuron_distance**3
        return math.sqrt(volume / self.plate_distance)

    @property
    def volume(self) -> float:
        return self.side * self.side * self.plate_distance

    @property
    def domain_min(self) -> torch.Tensor:
        return torch.tensor(
            [0.0, 0.0, 0.5 * self.mean_neuron_distance],
            dtype=torch.float32,
        )

    @property
    def domain_max(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.side,
                self.side,
                self.plate_distance - 0.5 * self.mean_neuron_distance,
            ],
            dtype=torch.float32,
        )

    def sample_positions(self, seed: int = 7) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        lower = self.domain_min
        upper = self.domain_max
        return lower + (upper - lower) * torch.rand(
            self.n_neurons, 3, generator=generator
        )

    def plate_vertices(self, z: float) -> list[list[tuple[float, float, float]]]:
        side = self.side
        return [
            [
                (0.0, 0.0, z),
                (side, 0.0, z),
                (side, side, z),
                (0.0, side, z),
            ]
        ]


def sparsify_weights(
    weights: torch.Tensor,
    positions: torch.Tensor,
    *,
    edges_per_neuron: int = 64,
    distance_decay: float = 5.0,
) -> torch.Tensor:
    """Keep the strongest distance-weighted outgoing synapses per neuron."""

    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be square")
    n = weights.shape[0]
    if positions.shape != (n, 3):
        raise ValueError("positions must have shape [n_neurons, 3]")
    if not 0 < edges_per_neuron < n:
        raise ValueError("edges_per_neuron must be in (0, n_neurons)")
    if distance_decay <= 0:
        raise ValueError("distance_decay must be positive")

    distance = torch.cdist(positions.float(), positions.float())
    score = weights.detach().float().abs() * torch.exp(
        -distance / float(distance_decay)
    )
    indices = torch.topk(score, edges_per_neuron, dim=1).indices
    mask = torch.zeros(
        n, n, dtype=torch.bool, device=weights.device
    )
    mask.scatter_(1, indices, True)
    return mask


def migrate(
    positions: torch.Tensor,
    activity: torch.Tensor,
    geometry: PlateGeometry,
    *,
    maximum_speed: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move active neurons towards the back plate; return displacement vector."""

    if positions.shape != (geometry.n_neurons, 3):
        raise ValueError("positions do not match geometry")
    if activity.shape != (geometry.n_neurons,):
        raise ValueError("activity must have one value per neuron")
    activity = activity.detach().float().clamp(0.0, 1.0)
    target = positions.clone()
    target[:, 2] = geometry.plate_distance * (0.5 + 0.4 * activity)
    target[:, 0].clamp_(0.0, geometry.side)
    target[:, 1].clamp_(0.0, geometry.side)
    config = MigrationConfig(
        dt=1.0,
        attraction_gain=1.0,
        attraction_radius=geometry.mean_neuron_distance * 3.0,
        maximum_gradient_norm=1.0,
        maximum_speed=maximum_speed,
    )
    moved = migrate_neurons(
        positions,
        target,
        torch.zeros_like(positions),
        motility=1.0,
        domain_minimum=geometry.domain_min.to(positions.device),
        domain_maximum=geometry.domain_max.to(positions.device),
        config=config,
    )
    displacement = (moved - positions).norm(dim=1)
    return moved, displacement


@torch.no_grad()
def loss_driven_rewire(
    weights: torch.Tensor,
    gradient: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    *,
    fraction: float = 0.01,
    candidates_per_source: int = 8,
    distance_decay: float = 3.0,
    initial_weight_magnitude: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Prune low-|w·g| edges and form high-|g| nearby candidates.

    ``gradient`` is the loss gradient of the dense recurrent matrix from the
    latest backward pass. Unlike the activity-only rewire, the ranking here is
    driven by the language objective.
    """

    if gradient.shape != weights.shape:
        raise ValueError("gradient must match weights")
    n = weights.shape[0]
    existing = mask.nonzero(as_tuple=False)
    existing_src = existing[:, 0]
    existing_dst = existing[:, 1]
    importance = (
        weights.detach().float().abs()
        * gradient.detach().float().abs()
    )
    existing_importance = importance[existing_src, existing_dst]
    n_prune = max(1, int(existing_src.numel() * fraction))
    prune = torch.topk(
        existing_importance, n_prune, largest=False
    ).indices
    prune_src = existing_src[prune]
    prune_dst = existing_dst[prune]

    distance = torch.cdist(positions.float(), positions.float())
    candidate_score = (
        gradient.detach().float().abs()
        * torch.exp(-distance / float(distance_decay))
    )
    candidate_score.masked_fill_(mask, -1.0)
    candidate_score.fill_diagonal_(-1.0)
    candidate_idx = torch.topk(
        candidate_score, candidates_per_source, dim=1
    ).indices
    candidate_src = torch.arange(
        n, device=weights.device
    ).repeat_interleave(candidates_per_source)
    candidate_dst = candidate_idx.reshape(-1)
    candidate_values = candidate_score[candidate_src, candidate_dst]
    valid = candidate_values > 0
    candidate_src = candidate_src[valid]
    candidate_dst = candidate_dst[valid]
    candidate_values = candidate_values[valid]
    n_form = min(n_prune, candidate_values.numel())
    form = torch.topk(candidate_values, n_form).indices
    form_src = candidate_src[form]
    form_dst = candidate_dst[form]

    new_mask = mask.clone()
    new_weights = weights.clone()
    new_mask[prune_src, prune_dst] = False
    new_weights[prune_src, prune_dst] = 0.0
    new_mask[form_src, form_dst] = True
    new_weights[form_src, form_dst] = (
        -torch.sign(gradient[form_src, form_dst])
        * initial_weight_magnitude
    )
    return new_weights, new_mask, {
        "formed": float(n_form),
        "pruned": float(n_prune),
        "rewire_ratio": float((n_form + n_prune) / max(mask.sum().item(), 1)),
        "loss_driven": 1.0,
    }


@torch.no_grad()
def rewire(
    weights: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    activity: torch.Tensor,
    *,
    fraction: float = 0.01,
    candidates_per_source: int = 8,
    distance_decay: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Prune weak existing edges and form nearby active candidates.

    Existing and candidate edges are scored by the engine's
    ``score_candidate_edges``. The mask is kept dense for the optimizer; only
    the selected entries change.
    """

    n = weights.shape[0]
    flow = weights.detach().float() * mask
    existing = mask.nonzero(as_tuple=False)
    existing_src = existing[:, 0]
    existing_dst = existing[:, 1]
    activity = activity.detach().float().clamp(0.0, 1.0)
    config = StructuralPlasticityConfig(
        distance_scale=float(distance_decay),
        distance_weight=1.0,
        activity_weight=1.0,
        chemical_weight=0.0,
        formation_threshold=0.5,
        prune_threshold=0.2,
        maximum_new_edges_per_source=candidates_per_source,
        initial_weight_magnitude=0.01,
        allow_self_connections=True,
    )
    existing_scores = score_candidate_edges(
        positions,
        activity,
        existing_src,
        existing_dst,
        torch.ones(existing_src.numel(), device=positions.device),
        config=config,
    ).total
    n_prune = max(1, int(existing_src.numel() * fraction))
    prune = torch.topk(existing_scores, n_prune, largest=False).indices
    prune_src = existing_src[prune]
    prune_dst = existing_dst[prune]

    distance = torch.cdist(positions.float(), positions.float())
    candidate_score = torch.exp(-distance / float(distance_decay)) * (
        activity[:, None] * activity[None, :]
    )
    candidate_score.masked_fill_(mask, -1.0)
    candidate_score.fill_diagonal_(-1.0)
    candidate_idx = torch.topk(
        candidate_score, candidates_per_source, dim=1
    ).indices
    candidate_src = torch.arange(
        n, device=positions.device
    ).repeat_interleave(candidates_per_source)
    candidate_dst = candidate_idx.reshape(-1)
    valid = candidate_dst >= 0
    candidate_src = candidate_src[valid]
    candidate_dst = candidate_dst[valid]
    candidate_scores = score_candidate_edges(
        positions,
        activity,
        candidate_src,
        candidate_dst,
        torch.ones(candidate_src.numel(), device=positions.device),
        config=config,
    ).total
    n_form = min(n_prune, int(valid.sum()))
    form = torch.topk(candidate_scores, n_form).indices
    form_src = candidate_src[form]
    form_dst = candidate_dst[form]

    new_mask = mask.clone()
    new_weights = weights.clone()
    new_mask[prune_src, prune_dst] = False
    new_weights[prune_src, prune_dst] = 0.0
    sign = torch.sign(flow.mean(dim=1)).clamp(min=-1.0, max=1.0)
    new_mask[form_src, form_dst] = True
    new_weights[form_src, form_dst] = (
        sign[form_src] * config.initial_weight_magnitude
    )

    return new_weights, new_mask, {
        "formed": float(n_form),
        "pruned": float(n_prune),
        "rewire_ratio": float((n_form + n_prune) / max(mask.sum().item(), 1)),
        "mean_displacement": 0.0,
    }
