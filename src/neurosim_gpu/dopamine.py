from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch


BatchReduction = Literal["mean", "sum"]


@dataclass(frozen=True)
class TemporalDifferenceRPE:
    """Clipped temporal-difference signal for an external or learned critic."""

    discount: float = 0.99
    maximum_absolute: float = 5.0

    def __post_init__(self) -> None:
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must lie in [0, 1]")
        if not math.isfinite(self.maximum_absolute) or self.maximum_absolute <= 0:
            raise ValueError("maximum_absolute must be positive")

    def compute(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor,
        terminal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reward.shape != value.shape or reward.shape != next_value.shape:
            raise ValueError("reward and values must have identical shapes")
        if reward.device != value.device or reward.device != next_value.device:
            raise ValueError("reward and values must share a device")
        if reward.dtype != value.dtype or reward.dtype != next_value.dtype:
            raise ValueError("reward and values must share a dtype")
        if not bool(torch.all(torch.isfinite(reward))):
            raise ValueError("reward must be finite")
        if not bool(torch.all(torch.isfinite(value))) or not bool(
            torch.all(torch.isfinite(next_value))
        ):
            raise ValueError("values must be finite")
        continuation: torch.Tensor | float = 1.0
        if terminal is not None:
            if terminal.shape != reward.shape or terminal.device != reward.device:
                raise ValueError("terminal must match reward shape and device")
            if not bool(torch.all(torch.isfinite(terminal))):
                raise ValueError("terminal must be finite")
            continuation = 1.0 - terminal.to(reward.dtype)
            if bool(torch.any((terminal < 0) | (terminal > 1))):
                raise ValueError("terminal must lie in [0, 1]")
        signal = reward + self.discount * continuation * next_value - value
        return torch.clamp(
            signal,
            min=-self.maximum_absolute,
            max=self.maximum_absolute,
        )


def _parameter(
    value: torch.Tensor | float,
    reference: torch.Tensor,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> torch.Tensor:
    result = torch.as_tensor(
        value,
        dtype=reference.dtype,
        device=reference.device,
    )
    if result.ndim > 1 or (result.ndim == 1 and result.shape != reference.shape):
        raise ValueError(f"{name} must be scalar or have one value per neuron")
    if not bool(torch.all(torch.isfinite(result))):
        raise ValueError(f"{name} must be finite")
    if positive and bool(torch.any(result <= 0)):
        raise ValueError(f"{name} must be positive")
    if nonnegative and bool(torch.any(result < 0)):
        raise ValueError(f"{name} must be non-negative")
    return result


def _neuron_field(
    value: torch.Tensor | float,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(
        value,
        dtype=reference.dtype,
        device=reference.device,
    )
    if result.ndim == 0:
        return result.expand_as(reference)
    if result.shape[-1] != reference.numel():
        raise ValueError(f"{name} must be scalar or end in the neuron dimension")
    return result


def _hill_occupancy(
    concentration: torch.Tensor,
    half_saturation: torch.Tensor,
    hill_coefficient: torch.Tensor,
) -> torch.Tensor:
    # The log-domain form remains stable for concentrations far above Kd.
    safe_concentration = torch.clamp_min(
        concentration,
        torch.finfo(concentration.dtype).tiny,
    )
    log_ratio = torch.log(safe_concentration) - torch.log(half_saturation)
    occupancy = torch.sigmoid(hill_coefficient * log_ratio)
    return torch.where(concentration > 0, occupancy, torch.zeros_like(occupancy))


@dataclass(frozen=True)
class DopamineSignals:
    """Receptor occupancy and signed deviation from tonic dopamine.

    D1 pathway values have the sign of the occupancy change. D2 pathway
    values have the opposite sign. Consequently a burst potentiates a pure
    D1 pathway and depresses a pure D2 pathway; a dip reverses both effects.
    """

    concentration: torch.Tensor
    d1_occupancy: torch.Tensor
    d2_occupancy: torch.Tensor
    d1_pathway: torch.Tensor
    d2_pathway: torch.Tensor
    net_phasic: torch.Tensor


@dataclass(frozen=True)
class DopamineReceptors:
    """Per-neuron D1/D2 receptor densities and binding parameters."""

    d1_density: torch.Tensor
    d2_density: torch.Tensor
    tonic_concentration: torch.Tensor | float = 1.0
    d1_half_saturation: torch.Tensor | float = 1.0
    d2_half_saturation: torch.Tensor | float = 0.25
    d1_hill_coefficient: torch.Tensor | float = 1.0
    d2_hill_coefficient: torch.Tensor | float = 1.0

    def __post_init__(self) -> None:
        if self.d1_density.ndim != 1 or self.d2_density.ndim != 1:
            raise ValueError("receptor densities must be rank-1 tensors")
        if self.d1_density.shape != self.d2_density.shape:
            raise ValueError("D1 and D2 receptor densities must have equal shape")
        if self.d1_density.device != self.d2_density.device:
            raise ValueError("D1 and D2 receptor densities must share a device")
        if self.d1_density.dtype != self.d2_density.dtype:
            raise ValueError("D1 and D2 receptor densities must share a dtype")
        if not self.d1_density.is_floating_point():
            raise ValueError("receptor densities must use a floating dtype")
        for name, density in (
            ("d1_density", self.d1_density),
            ("d2_density", self.d2_density),
        ):
            if not bool(torch.all(torch.isfinite(density))):
                raise ValueError(f"{name} must be finite")
            if bool(torch.any(density < 0)):
                raise ValueError(f"{name} must be non-negative")

        reference = self.d1_density
        tonic = _parameter(
            self.tonic_concentration,
            reference,
            "tonic_concentration",
            nonnegative=True,
        )
        d1_kd = _parameter(
            self.d1_half_saturation,
            reference,
            "d1_half_saturation",
            positive=True,
        )
        d2_kd = _parameter(
            self.d2_half_saturation,
            reference,
            "d2_half_saturation",
            positive=True,
        )
        d1_hill = _parameter(
            self.d1_hill_coefficient,
            reference,
            "d1_hill_coefficient",
            positive=True,
        )
        d2_hill = _parameter(
            self.d2_hill_coefficient,
            reference,
            "d2_hill_coefficient",
            positive=True,
        )
        tonic = tonic.expand_as(reference)
        d1_kd = d1_kd.expand_as(reference)
        d2_kd = d2_kd.expand_as(reference)
        d1_hill = d1_hill.expand_as(reference)
        d2_hill = d2_hill.expand_as(reference)
        object.__setattr__(self, "_tonic", tonic)
        object.__setattr__(self, "_d1_kd", d1_kd)
        object.__setattr__(self, "_d2_kd", d2_kd)
        object.__setattr__(self, "_d1_hill", d1_hill)
        object.__setattr__(self, "_d2_hill", d2_hill)
        object.__setattr__(
            self, "_tonic_d1", _hill_occupancy(tonic, d1_kd, d1_hill)
        )
        object.__setattr__(
            self, "_tonic_d2", _hill_occupancy(tonic, d2_kd, d2_hill)
        )

    @property
    def n_neurons(self) -> int:
        return self.d1_density.numel()

    def response(
        self,
        concentration: torch.Tensor | float,
    ) -> DopamineSignals:
        """Evaluate D1/D2 effects relative to tonic receptor occupancy.

        Runtime concentrations are projected onto the physically meaningful
        non-negative domain. Leading batch or region dimensions are allowed as
        long as the last dimension indexes neurons.
        """

        reference = self.d1_density
        concentration_tensor = _neuron_field(
            concentration,
            reference,
            "concentration",
        ).clamp_min(0.0)
        return self._response_tensor(concentration_tensor)

    def response_unchecked(self, concentration: torch.Tensor) -> DopamineSignals:
        """Evaluate a shape/device-validated concentration without CPU sync."""

        return self._response_tensor(concentration.clamp_min(0.0))

    def _response_tensor(self, concentration_tensor: torch.Tensor) -> DopamineSignals:
        d1_occupancy = _hill_occupancy(
            concentration_tensor, self._d1_kd, self._d1_hill
        )
        d2_occupancy = _hill_occupancy(
            concentration_tensor, self._d2_kd, self._d2_hill
        )
        d1_pathway = self.d1_density * (d1_occupancy - self._tonic_d1)
        d2_pathway = -self.d2_density * (d2_occupancy - self._tonic_d2)
        return DopamineSignals(
            concentration=concentration_tensor,
            d1_occupancy=d1_occupancy,
            d2_occupancy=d2_occupancy,
            d1_pathway=d1_pathway,
            d2_pathway=d2_pathway,
            net_phasic=d1_pathway + d2_pathway,
        )


@dataclass(frozen=True)
class NeuronModulation:
    drive: torch.Tensor
    gain: torch.Tensor


def modulate_neuron_dynamics(
    base_drive: torch.Tensor | float,
    base_gain: torch.Tensor | float,
    dopamine: DopamineSignals | torch.Tensor,
    *,
    drive_scale: torch.Tensor | float = 1.0,
    gain_scale: torch.Tensor | float = 1.0,
) -> NeuronModulation:
    """Apply signed dopamine to neuronal drive and multiplicative gain.

    Exponential gain modulation preserves the non-negative gain invariant.
    The dopamine tensor may include leading batch dimensions.
    """

    signal = dopamine.net_phasic if isinstance(dopamine, DopamineSignals) else dopamine
    if not isinstance(signal, torch.Tensor) or not signal.is_floating_point():
        raise ValueError("dopamine must provide a floating tensor")
    drive = torch.as_tensor(base_drive, dtype=signal.dtype, device=signal.device)
    gain = torch.as_tensor(base_gain, dtype=signal.dtype, device=signal.device)
    if bool(torch.any(gain < 0)):
        raise ValueError("base_gain must be non-negative")
    drive_factor = torch.as_tensor(
        drive_scale,
        dtype=signal.dtype,
        device=signal.device,
    )
    gain_factor = torch.as_tensor(
        gain_scale,
        dtype=signal.dtype,
        device=signal.device,
    )
    return NeuronModulation(
        drive=drive + drive_factor * signal,
        gain=gain * torch.exp(torch.clamp(gain_factor * signal, -20.0, 20.0)),
    )


def _validate_edges(src: torch.Tensor, dst: torch.Tensor) -> None:
    if src.ndim != 1 or dst.ndim != 1 or src.shape != dst.shape:
        raise ValueError("src and dst must be equal-length rank-1 tensors")
    if src.dtype not in (torch.int32, torch.int64) or dst.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("src and dst must use an integer dtype")
    if src.device != dst.device:
        raise ValueError("src and dst must share a device")
    if src.numel() and bool(torch.any(src[1:] < src[:-1])):
        raise ValueError("edges must be source-major")


def update_eligibility_trace(
    trace: torch.Tensor,
    pre_state: torch.Tensor,
    post_state: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    decay: torch.Tensor | float,
    batch_reduction: BatchReduction = "mean",
) -> torch.Tensor:
    """Update one local eligibility value per source-major edge.

    The update is ``decay * trace + reduce_batch(pre[src] * post[dst])``.
    It uses O(M) persistent state and O(B*M) work for a batch of size B.
    ``post_state`` can be activity, a postsynaptic derivative, or another
    locally available learning factor.
    """

    _validate_edges(src, dst)
    if trace.ndim != 1 or trace.shape != src.shape:
        raise ValueError("trace must have one value per edge")
    if not trace.is_floating_point():
        raise ValueError("trace must use a floating dtype")
    if pre_state.ndim < 1 or post_state.ndim < 1:
        raise ValueError("pre_state and post_state must include a neuron dimension")
    if pre_state.shape[:-1] != post_state.shape[:-1]:
        raise ValueError("pre_state and post_state batch dimensions must match")
    if not (
        trace.device
        == pre_state.device
        == post_state.device
        == src.device
        == dst.device
    ):
        raise ValueError("trace, states and edge indices must share a device")
    if batch_reduction not in ("mean", "sum"):
        raise ValueError("batch_reduction must be mean or sum")

    src64 = src.to(torch.int64)
    dst64 = dst.to(torch.int64)
    if src.numel() and (
        bool(torch.any(src64 < 0))
        or bool(torch.any(dst64 < 0))
        or int(src64.max()) >= pre_state.shape[-1]
        or int(dst64.max()) >= post_state.shape[-1]
    ):
        raise ValueError("edge index is outside the state neuron dimension")
    decay_tensor = torch.as_tensor(
        decay,
        dtype=trace.dtype,
        device=trace.device,
    )
    if decay_tensor.ndim > 1 or (
        decay_tensor.ndim == 1 and decay_tensor.shape != trace.shape
    ):
        raise ValueError("decay must be scalar or have one value per edge")
    if bool(torch.any((decay_tensor < 0) | (decay_tensor > 1))):
        raise ValueError("decay must lie in [0, 1]")

    pre = pre_state.reshape(-1, pre_state.shape[-1]).to(trace.dtype)
    post = post_state.reshape(-1, post_state.shape[-1]).to(trace.dtype)
    samples = pre.index_select(1, src64) * post.index_select(1, dst64)
    increment = (
        samples.mean(dim=0)
        if batch_reduction == "mean"
        else samples.sum(dim=0)
    )
    return decay_tensor * trace + increment


@dataclass(frozen=True)
class PlasticityUpdate:
    weight: torch.Tensor
    delta: torch.Tensor
    dopamine_gate: torch.Tensor


def _edge_dopamine_gate(
    dopamine: DopamineSignals | torch.Tensor,
    weight: torch.Tensor,
    dst: torch.Tensor | None,
    batch_reduction: BatchReduction,
) -> torch.Tensor:
    signal = dopamine.net_phasic if isinstance(dopamine, DopamineSignals) else dopamine
    signal = torch.as_tensor(signal, dtype=weight.dtype, device=weight.device)
    if signal.ndim == 0:
        return signal.expand_as(weight)

    if dst is not None:
        if dst.ndim != 1 or dst.shape != weight.shape:
            raise ValueError("dst must have one index per weight")
        if dst.device != weight.device:
            raise ValueError("dst and weight must share a device")
        dst64 = dst.to(torch.int64)
        if dst.numel() and (
            bool(torch.any(dst64 < 0)) or int(dst64.max()) >= signal.shape[-1]
        ):
            raise ValueError("dst index is outside the dopamine neuron dimension")
        edge_signal = signal.index_select(-1, dst64)
    else:
        if signal.shape[-1] != weight.numel():
            raise ValueError(
                "edge dopamine must end in M values when dst is not supplied"
            )
        edge_signal = signal

    if edge_signal.ndim == 1:
        return edge_signal
    flattened = edge_signal.reshape(-1, weight.numel())
    return (
        flattened.mean(dim=0)
        if batch_reduction == "mean"
        else flattened.sum(dim=0)
    )


def dopamine_gated_weight_update(
    weight: torch.Tensor,
    eligibility: torch.Tensor,
    dopamine: DopamineSignals | torch.Tensor,
    *,
    dst: torch.Tensor | None = None,
    learning_rate: torch.Tensor | float = 1.0,
    weight_decay: torch.Tensor | float = 0.0,
    minimum_weight: torch.Tensor | float | None = None,
    maximum_weight: torch.Tensor | float | None = None,
    preserve_sign: bool = False,
    update_magnitude: bool = False,
    batch_reduction: BatchReduction = "mean",
) -> PlasticityUpdate:
    """Apply the third-factor rule ``eta * dopamine * eligibility``.

    Dopamine can be scalar, edge-local, or neuron-local when ``dst`` is
    supplied. Leading batch dimensions are reduced before the edge update.
    Weight decay is a fractional decay in [0, 1]. Sign preservation projects
    excitatory and inhibitory weights onto their original half axes.
    """

    if weight.ndim != 1 or eligibility.shape != weight.shape:
        raise ValueError("weight and eligibility must be equal-length vectors")
    if not weight.is_floating_point() or not eligibility.is_floating_point():
        raise ValueError("weight and eligibility must use floating dtypes")
    if weight.device != eligibility.device:
        raise ValueError("weight and eligibility must share a device")
    if batch_reduction not in ("mean", "sum"):
        raise ValueError("batch_reduction must be mean or sum")

    gate = _edge_dopamine_gate(
        dopamine,
        weight,
        dst,
        batch_reduction,
    )
    rate = torch.as_tensor(
        learning_rate,
        dtype=weight.dtype,
        device=weight.device,
    )
    decay = torch.as_tensor(
        weight_decay,
        dtype=weight.dtype,
        device=weight.device,
    )
    for name, parameter in (("learning_rate", rate), ("weight_decay", decay)):
        if parameter.ndim > 1 or (
            parameter.ndim == 1 and parameter.shape != weight.shape
        ):
            raise ValueError(f"{name} must be scalar or have one value per edge")
    if bool(torch.any(rate < 0)):
        raise ValueError("learning_rate must be non-negative")
    if bool(torch.any((decay < 0) | (decay > 1))):
        raise ValueError("weight_decay must lie in [0, 1]")

    lower = (
        torch.full_like(weight, -torch.inf)
        if minimum_weight is None
        else torch.as_tensor(
            minimum_weight,
            dtype=weight.dtype,
            device=weight.device,
        )
    )
    upper = (
        torch.full_like(weight, torch.inf)
        if maximum_weight is None
        else torch.as_tensor(
            maximum_weight,
            dtype=weight.dtype,
            device=weight.device,
        )
    )
    for name, bound in (("minimum_weight", lower), ("maximum_weight", upper)):
        if bound.ndim > 1 or (bound.ndim == 1 and bound.shape != weight.shape):
            raise ValueError(f"{name} must be scalar or have one value per edge")
    lower = torch.broadcast_to(lower, weight.shape)
    upper = torch.broadcast_to(upper, weight.shape)

    if preserve_sign:
        zero = torch.zeros_like(weight)
        lower = torch.where(weight > 0, torch.maximum(lower, zero), lower)
        upper = torch.where(weight < 0, torch.minimum(upper, zero), upper)
        lower = torch.where(weight == 0, zero, lower)
        upper = torch.where(weight == 0, zero, upper)
    if bool(torch.any(lower > upper)):
        raise ValueError("weight bounds are inconsistent with each other or sign")

    direction = torch.sign(weight) if update_magnitude else 1.0
    candidate = (
        weight * (1.0 - decay)
        + rate * direction * gate * eligibility
    )
    updated = torch.minimum(torch.maximum(candidate, lower), upper)
    return PlasticityUpdate(
        weight=updated,
        delta=updated - weight,
        dopamine_gate=gate,
    )


def dopamine_gated_batch_weight_update(
    weight: torch.Tensor,
    eligibility_by_batch: torch.Tensor,
    dopamine_by_batch: torch.Tensor,
    *,
    dst: torch.Tensor,
    learning_rate: torch.Tensor | float = 1.0,
    weight_decay: torch.Tensor | float = 0.0,
    minimum_weight: torch.Tensor | float | None = None,
    maximum_weight: torch.Tensor | float | None = None,
    preserve_sign: bool = False,
    update_magnitude: bool = False,
    batch_reduction: BatchReduction = "mean",
) -> PlasticityUpdate:
    """Exact batch three-factor update by reducing eligibility*dopamine.

    Unlike separately averaged one-dimensional traces and dopamine gates, this
    function preserves within-sample correlation. It intentionally requires
    ``O(B*M)`` eligibility storage and is therefore an offline/small-batch API.
    """

    if eligibility_by_batch.ndim != 2 or eligibility_by_batch.shape[1] != weight.numel():
        raise ValueError("eligibility_by_batch must have shape [batch, edges]")
    if dopamine_by_batch.ndim != 2 or dopamine_by_batch.shape[0] != eligibility_by_batch.shape[0]:
        raise ValueError("dopamine_by_batch must have shape [batch, neurons]")
    if batch_reduction not in ("mean", "sum"):
        raise ValueError("batch_reduction must be mean or sum")
    if not (
        eligibility_by_batch.device
        == dopamine_by_batch.device
        == weight.device
        == dst.device
    ):
        raise ValueError("batch plasticity tensors must share a device")
    edge_gate = dopamine_by_batch.index_select(-1, dst.to(torch.int64))
    joint = eligibility_by_batch * edge_gate
    reduced = joint.mean(0) if batch_reduction == "mean" else joint.sum(0)
    return dopamine_gated_weight_update(
        weight,
        reduced,
        torch.ones_like(weight),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
        preserve_sign=preserve_sign,
        update_magnitude=update_magnitude,
    )
