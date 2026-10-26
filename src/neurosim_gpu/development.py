from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MigrationConfig:
    """Parameters for one slow developmental migration step.

    Target attraction and chemotactic gradients are independently norm-bounded
    before motility is applied. ``maximum_speed`` then supplies a final physical
    displacement bound of ``dt * maximum_speed`` per developmental step.
    """

    dt: float
    attraction_gain: float = 1.0
    attraction_radius: float = 1.0
    chemotaxis_gain: float = 1.0
    maximum_gradient_norm: float = 1.0
    noise_scale: float = 0.0
    maximum_speed: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "dt": self.dt,
            "attraction_radius": self.attraction_radius,
            "maximum_gradient_norm": self.maximum_gradient_norm,
        }
        non_negative = {
            "attraction_gain": self.attraction_gain,
            "chemotaxis_gain": self.chemotaxis_gain,
            "noise_scale": self.noise_scale,
            "maximum_speed": self.maximum_speed,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class CandidateEdgeScores:
    """Scores aligned one-for-one with an explicitly supplied candidate list."""

    distance: torch.Tensor
    activity: torch.Tensor
    chemical: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class StructuralPlasticityConfig:
    """Slow-timescale structural formation and pruning parameters.

    Component weights form a normalized convex combination, so both thresholds
    use a stable ``[0, 1]`` scale. A formation threshold at least as high as the
    prune threshold introduces hysteresis and prevents edge chattering.
    """

    distance_scale: float = 1.0
    distance_weight: float = 1.0
    activity_weight: float = 1.0
    chemical_weight: float = 1.0
    formation_threshold: float = 0.75
    prune_threshold: float = 0.25
    maximum_new_edges_per_source: int = 8
    initial_weight_magnitude: float = 0.01
    allow_self_connections: bool = False

    def __post_init__(self) -> None:
        finite_values = {
            "distance_scale": self.distance_scale,
            "distance_weight": self.distance_weight,
            "activity_weight": self.activity_weight,
            "chemical_weight": self.chemical_weight,
            "formation_threshold": self.formation_threshold,
            "prune_threshold": self.prune_threshold,
            "initial_weight_magnitude": self.initial_weight_magnitude,
        }
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.distance_scale <= 0:
            raise ValueError("distance_scale must be positive")
        if min(
            self.distance_weight,
            self.activity_weight,
            self.chemical_weight,
        ) < 0:
            raise ValueError("score weights must be non-negative")
        if self.score_weight_sum <= 0:
            raise ValueError("at least one score weight must be positive")
        if not 0 <= self.prune_threshold <= self.formation_threshold <= 1:
            raise ValueError(
                "thresholds must satisfy 0 <= prune <= formation <= 1"
            )
        if (
            not isinstance(self.maximum_new_edges_per_source, int)
            or isinstance(self.maximum_new_edges_per_source, bool)
            or self.maximum_new_edges_per_source < 0
        ):
            raise ValueError("maximum_new_edges_per_source must be non-negative")
        if self.initial_weight_magnitude <= 0:
            raise ValueError("initial_weight_magnitude must be positive")

    @property
    def score_weight_sum(self) -> float:
        return (
            self.distance_weight
            + self.activity_weight
            + self.chemical_weight
        )


@dataclass(frozen=True)
class ConnectomeUpdate:
    """Canonical source-major tensors and candidate-aligned decisions.

    ``src``, ``dst`` and ``weight`` can be passed directly to
    :meth:`SparseTopology.from_edges`. ``formed_mask`` and ``pruned_mask`` are
    aligned with the caller's candidate list, while ``scores`` exposes the
    individual biological terms for diagnostics.
    """

    src: torch.Tensor
    dst: torch.Tensor
    weight: torch.Tensor
    formed_mask: torch.Tensor
    pruned_mask: torch.Tensor
    scores: CandidateEdgeScores

    @property
    def n_formed(self) -> int:
        return int(self.formed_mask.sum())

    @property
    def n_pruned(self) -> int:
        return int(self.pruned_mask.sum())


def _validate_xyz(name: str, value: torch.Tensor, like: torch.Tensor) -> None:
    if value.shape != like.shape or value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [neurons, 3]")
    if value.dtype != like.dtype or value.device != like.device:
        raise ValueError(f"{name} must match positions' dtype and device")
    if not value.is_floating_point() or not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must contain finite floating-point values")


def _domain_vector(
    name: str,
    value: torch.Tensor | float,
    positions: torch.Tensor,
) -> torch.Tensor:
    result = torch.as_tensor(
        value,
        dtype=positions.dtype,
        device=positions.device,
    )
    if result.ndim == 0:
        result = result.expand(3)
    if result.shape != (3,) or not bool(torch.all(torch.isfinite(result))):
        raise ValueError(f"{name} must be finite and scalar or shape [3]")
    return result


def _neuron_parameter(
    name: str,
    value: torch.Tensor | float,
    positions: torch.Tensor,
) -> torch.Tensor:
    result = torch.as_tensor(
        value,
        dtype=positions.dtype,
        device=positions.device,
    )
    if result.ndim == 0:
        result = result.expand(positions.shape[0])
    if result.shape != (positions.shape[0],):
        raise ValueError(f"{name} must be scalar or shape [neurons]")
    if not bool(torch.all(torch.isfinite(result))) or bool(torch.any(result < 0)):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _clip_vector_norm(value: torch.Tensor, maximum: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=1, keepdim=True)
    scale = torch.clamp(maximum / norm.clamp_min(torch.finfo(value.dtype).tiny), max=1.0)
    return value * scale


def migrate_neurons(
    positions: torch.Tensor,
    target_positions: torch.Tensor,
    chemotactic_gradient: torch.Tensor,
    motility: torch.Tensor | float,
    *,
    domain_minimum: torch.Tensor | float,
    domain_maximum: torch.Tensor | float,
    config: MigrationConfig,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance vectorized neuron positions by one slow developmental step.

    ``noise`` is caller-supplied rather than sampled internally. This makes a
    run reproducible across checkpoints and lets the owning simulation choose
    its stochastic process. The operation costs ``O(N)`` work and storage; it
    never constructs neuron pairs.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [neurons, 3]")
    if not positions.is_floating_point() or not bool(
        torch.all(torch.isfinite(positions))
    ):
        raise ValueError("positions must contain finite floating-point values")
    _validate_xyz("target_positions", target_positions, positions)
    _validate_xyz("chemotactic_gradient", chemotactic_gradient, positions)
    if noise is not None:
        _validate_xyz("noise", noise, positions)

    lower = _domain_vector("domain_minimum", domain_minimum, positions)
    upper = _domain_vector("domain_maximum", domain_maximum, positions)
    if bool(torch.any(lower >= upper)):
        raise ValueError("domain_minimum must be below domain_maximum")
    tolerance = 16 * torch.finfo(positions.dtype).eps
    if bool(torch.any(positions < lower - tolerance)) or bool(
        torch.any(positions > upper + tolerance)
    ):
        raise ValueError("positions must start inside the domain")

    motility_vector = _neuron_parameter("motility", motility, positions)
    target_offset = target_positions - positions
    attraction = _clip_vector_norm(target_offset, config.attraction_radius)
    gradient = _clip_vector_norm(
        chemotactic_gradient,
        config.maximum_gradient_norm,
    )
    drive = (
        config.attraction_gain * attraction
        + config.chemotaxis_gain * gradient
    )
    if noise is not None and config.noise_scale:
        drive = drive + config.noise_scale * noise

    velocity = motility_vector[:, None] * drive
    velocity = _clip_vector_norm(velocity, config.maximum_speed)
    advanced = positions + config.dt * velocity
    return torch.minimum(torch.maximum(advanced, lower), upper)


def _validate_edge_indices(
    name: str,
    src: torch.Tensor,
    dst: torch.Tensor,
    n_neurons: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if src.ndim != 1 or dst.ndim != 1 or src.shape != dst.shape:
        raise ValueError(f"{name}_src and {name}_dst must be equal rank-1 tensors")
    if src.dtype not in (torch.int32, torch.int64) or dst.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"{name} indices must use torch.int32 or torch.int64")
    if src.device != device or dst.device != device:
        raise ValueError(f"{name} indices must be on positions' device")
    src64 = src.to(torch.int64)
    dst64 = dst.to(torch.int64)
    if src64.numel() and (
        int(src64.min()) < 0
        or int(dst64.min()) < 0
        or int(src64.max()) >= n_neurons
        or int(dst64.max()) >= n_neurons
    ):
        raise ValueError(f"{name} edge index out of range")
    return src64, dst64


def _validate_unit_interval(
    name: str,
    value: torch.Tensor,
    length: int,
    positions: torch.Tensor,
) -> None:
    if value.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}]")
    if (
        not value.is_floating_point()
        or value.dtype != positions.dtype
        or value.device != positions.device
    ):
        raise ValueError(f"{name} must match positions' dtype and device")
    if not bool(torch.all(torch.isfinite(value))) or bool(
        torch.any((value < 0) | (value > 1))
    ):
        raise ValueError(f"{name} must contain finite values in [0, 1]")


def score_candidate_edges(
    positions: torch.Tensor,
    activity: torch.Tensor,
    candidate_src: torch.Tensor,
    candidate_dst: torch.Tensor,
    chemical_affinity: torch.Tensor,
    *,
    config: StructuralPlasticityConfig,
) -> CandidateEdgeScores:
    """Score exactly ``K`` supplied edges without any all-pairs expansion.

    Activity is a normalized non-negative firing/activity statistic and
    ``chemical_affinity`` is a caller-computed ligand/receptor or local-field
    compatibility score. Work and temporary storage are ``O(N + K)``.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [neurons, 3]")
    if not positions.is_floating_point() or not bool(
        torch.all(torch.isfinite(positions))
    ):
        raise ValueError("positions must contain finite floating-point values")
    n_neurons = positions.shape[0]
    src, dst = _validate_edge_indices(
        "candidate",
        candidate_src,
        candidate_dst,
        n_neurons,
        positions.device,
    )
    _validate_unit_interval("activity", activity, n_neurons, positions)
    _validate_unit_interval(
        "chemical_affinity",
        chemical_affinity,
        src.numel(),
        positions,
    )

    displacement = positions.index_select(0, src) - positions.index_select(0, dst)
    distance = torch.linalg.vector_norm(displacement, dim=1)
    distance_score = torch.exp(-distance / config.distance_scale)
    activity_score = activity.index_select(0, src) * activity.index_select(0, dst)
    total = (
        config.distance_weight * distance_score
        + config.activity_weight * activity_score
        + config.chemical_weight * chemical_affinity
    ) / config.score_weight_sum
    return CandidateEdgeScores(
        distance=distance_score,
        activity=activity_score,
        chemical=chemical_affinity,
        total=total,
    )


def _validate_unique_edges(name: str, key: torch.Tensor) -> None:
    if key.numel() > 1:
        ordered = torch.sort(key, stable=True).values
        if bool(torch.any(ordered[1:] == ordered[:-1])):
            raise ValueError(f"{name} edge list must not contain duplicates")


def _top_per_source(
    eligible: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    score: torch.Tensor,
    n_neurons: int,
    maximum: int,
) -> torch.Tensor:
    selected = torch.zeros_like(eligible)
    if maximum == 0:
        return selected
    index = torch.nonzero(eligible, as_tuple=False).flatten()
    if index.numel() == 0:
        return selected

    # Stable sorts compose into source asc, score desc, destination asc. The
    # destination key gives a device-independent tie breaker for equal scores.
    index = index[torch.argsort(dst[index], stable=True)]
    index = index[torch.argsort(score[index], descending=True, stable=True)]
    index = index[torch.argsort(src[index], stable=True)]
    sorted_src = src[index]
    count = torch.bincount(sorted_src, minlength=n_neurons)
    row_start = torch.empty(n_neurons, dtype=torch.int64, device=src.device)
    row_start[0] = 0
    if n_neurons > 1:
        row_start[1:] = torch.cumsum(count[:-1], dim=0)
    rank = torch.arange(index.numel(), device=src.device) - row_start[sorted_src]
    selected[index[rank < maximum]] = True
    return selected


@torch.no_grad()
def update_structural_connectome(
    positions: torch.Tensor,
    activity: torch.Tensor,
    existing_src: torch.Tensor,
    existing_dst: torch.Tensor,
    existing_weight: torch.Tensor,
    candidate_src: torch.Tensor,
    candidate_dst: torch.Tensor,
    chemical_affinity: torch.Tensor,
    neuron_type: torch.Tensor,
    source_sign: torch.Tensor,
    type_compatibility: torch.Tensor,
    *,
    config: StructuralPlasticityConfig,
) -> ConnectomeUpdate:
    """Apply one deterministic slow structural-plasticity epoch.

    Only explicitly supplied candidate edges are scored. Existing edges absent
    from that list are preserved. Listed existing edges below the prune
    threshold are removed, while non-existing edges above the formation
    threshold compete for a per-source top-k budget. New weights take their
    sign from the presynaptic neuron and are created only for compatible type
    pairs.

    For ``M`` existing and ``K`` candidate edges, this epoch costs
    ``O(N + M log M + K log K)`` due to deterministic canonical sorting and
    ``O(N + M + K)`` storage, never ``O(N^2)``. If run once every ``S`` neural
    ticks, its amortized per-tick cost is divided by ``S``.
    """

    scores = score_candidate_edges(
        positions,
        activity,
        candidate_src,
        candidate_dst,
        chemical_affinity,
        config=config,
    )
    n_neurons = positions.shape[0]
    old_src, old_dst = _validate_edge_indices(
        "existing",
        existing_src,
        existing_dst,
        n_neurons,
        positions.device,
    )
    candidate_src64, candidate_dst64 = _validate_edge_indices(
        "candidate",
        candidate_src,
        candidate_dst,
        n_neurons,
        positions.device,
    )
    if existing_weight.shape != old_src.shape:
        raise ValueError("existing_weight must match the existing edge count")
    if (
        not existing_weight.is_floating_point()
        or existing_weight.dtype != positions.dtype
        or existing_weight.device != positions.device
        or not bool(torch.all(torch.isfinite(existing_weight)))
    ):
        raise ValueError("existing_weight must match positions and be finite")

    if neuron_type.shape != (n_neurons,) or neuron_type.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("neuron_type must be an integer tensor of shape [neurons]")
    if neuron_type.device != positions.device:
        raise ValueError("neuron_type must be on positions' device")
    if source_sign.shape != (n_neurons,) or source_sign.device != positions.device:
        raise ValueError("source_sign must have shape [neurons] on positions' device")
    sign = source_sign.to(dtype=positions.dtype)
    if not bool(torch.all((sign == -1) | (sign == 1))):
        raise ValueError("source_sign entries must be exactly -1 or +1")
    if type_compatibility.ndim != 2 or type_compatibility.shape[0] != type_compatibility.shape[1]:
        raise ValueError("type_compatibility must be a square matrix")
    if type_compatibility.dtype != torch.bool or type_compatibility.device != positions.device:
        raise ValueError("type_compatibility must be boolean on positions' device")
    types = neuron_type.to(torch.int64)
    n_types = type_compatibility.shape[0]
    if types.numel() and (
        int(types.min()) < 0 or int(types.max()) >= n_types
    ):
        raise ValueError("neuron_type index out of type_compatibility range")

    old_key = old_src * n_neurons + old_dst
    candidate_key = candidate_src64 * n_neurons + candidate_dst64
    _validate_unique_edges("existing", old_key)
    _validate_unique_edges("candidate", candidate_key)

    old_order = torch.argsort(old_key, stable=True)
    old_key = old_key[old_order]
    old_src = old_src[old_order]
    old_dst = old_dst[old_order]
    old_weight = existing_weight[old_order]

    is_existing = torch.zeros_like(candidate_key, dtype=torch.bool)
    existing_index = torch.full_like(candidate_key, -1)
    if old_key.numel() and candidate_key.numel():
        insertion = torch.searchsorted(old_key, candidate_key)
        in_range = insertion < old_key.numel()
        clamped = insertion.clamp_max(old_key.numel() - 1)
        is_existing = in_range & (old_key[clamped] == candidate_key)
        existing_index[is_existing] = clamped[is_existing]

    compatible = type_compatibility[
        types[candidate_src64],
        types[candidate_dst64],
    ]
    if config.allow_self_connections:
        permitted = compatible
    else:
        permitted = compatible & (candidate_src64 != candidate_dst64)

    pruned_mask = is_existing & (
        (~permitted) | (scores.total < config.prune_threshold)
    )
    remove_existing = torch.zeros(old_key.numel(), dtype=torch.bool, device=positions.device)
    if bool(torch.any(pruned_mask)):
        remove_existing[existing_index[pruned_mask]] = True

    formation_eligible = (
        (~is_existing)
        & permitted
        & (scores.total >= config.formation_threshold)
    )
    formed_mask = _top_per_source(
        formation_eligible,
        candidate_src64,
        candidate_dst64,
        scores.total,
        n_neurons,
        config.maximum_new_edges_per_source,
    )

    retained = ~remove_existing
    new_src = candidate_src64[formed_mask]
    new_dst = candidate_dst64[formed_mask]
    new_weight = (
        config.initial_weight_magnitude * sign.index_select(0, new_src)
    )
    output_src = torch.cat((old_src[retained], new_src))
    output_dst = torch.cat((old_dst[retained], new_dst))
    output_weight = torch.cat((old_weight[retained], new_weight))
    output_key = output_src * n_neurons + output_dst
    output_order = torch.argsort(output_key, stable=True)

    return ConnectomeUpdate(
        src=output_src[output_order],
        dst=output_dst[output_order],
        weight=output_weight[output_order],
        formed_mask=formed_mask,
        pruned_mask=pruned_mask,
        scores=scores,
    )
