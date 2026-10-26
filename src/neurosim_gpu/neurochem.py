from __future__ import annotations

from dataclasses import dataclass

import torch

from .chemistry import RegionalDopamineField
from .dopamine import (
    DopamineReceptors,
    DopamineSignals,
)
from .dynamics import Dynamics
from .runtime import MatrixRuntime, _validate_emission_threshold
from .topology import SparseTopology


@dataclass(frozen=True)
class DopaminePlasticityConfig:
    drive_scale: float = 1.0
    log_gain_scale: float = 0.0
    maximum_drive_shift: float = 1.0
    minimum_gain_scale: float = 0.5
    maximum_gain_scale: float = 2.0
    pre_trace_decay: float = 0.9
    post_trace_decay: float = 0.9
    eligibility_decay: float = 0.95
    potentiation_scale: float = 1.0
    depression_scale: float = 1.0
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    minimum_weight: float = -1.0
    maximum_weight: float = 1.0
    preserve_weight_sign: bool = True
    plasticity_interval: int = 1
    track_eligibility: bool = True
    chemistry_interval: int = 1
    dopamine_delay_steps: int = 0
    plasticity_updates_magnitude: bool = True
    validate_inputs: bool = True
    refresh_receptors: bool = True

    def __post_init__(self) -> None:
        finite = (
            self.drive_scale,
            self.log_gain_scale,
            self.maximum_drive_shift,
            self.minimum_gain_scale,
            self.maximum_gain_scale,
            self.pre_trace_decay,
            self.post_trace_decay,
            self.eligibility_decay,
            self.potentiation_scale,
            self.depression_scale,
            self.learning_rate,
            self.weight_decay,
            self.minimum_weight,
            self.maximum_weight,
        )
        if not all(torch.isfinite(torch.tensor(value)) for value in finite):
            raise ValueError("dopamine plasticity configuration must be finite")
        if self.maximum_drive_shift < 0:
            raise ValueError("maximum_drive_shift must be non-negative")
        if not 0 <= self.minimum_gain_scale <= self.maximum_gain_scale:
            raise ValueError("gain scale bounds are inconsistent")
        for name, value in (
            ("pre_trace_decay", self.pre_trace_decay),
            ("post_trace_decay", self.post_trace_decay),
            ("eligibility_decay", self.eligibility_decay),
            ("weight_decay", self.weight_decay),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.potentiation_scale < 0 or self.depression_scale < 0:
            raise ValueError("eligibility scales must be non-negative")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if self.minimum_weight > self.maximum_weight:
            raise ValueError("weight bounds are inconsistent")
        if self.plasticity_interval <= 0:
            raise ValueError("plasticity_interval must be positive")
        if self.chemistry_interval <= 0:
            raise ValueError("chemistry_interval must be positive")
        if self.dopamine_delay_steps < 0:
            raise ValueError("dopamine_delay_steps must be non-negative")
        if not self.track_eligibility and (
            self.learning_rate > 0 or self.weight_decay > 0
        ):
            raise ValueError(
                "weight updates require track_eligibility to be enabled"
            )
        if not self.refresh_receptors and (
            self.drive_scale != 0
            or self.log_gain_scale != 0
            or self.learning_rate != 0
            or self.weight_decay != 0
        ):
            raise ValueError(
                "refresh_receptors=False requires zero modulation and plasticity"
            )


@dataclass(frozen=True)
class DopamineStepStats:
    concentration_minimum: float
    concentration_maximum: float
    phasic_minimum: float
    phasic_maximum: float
    eligibility_maximum: float
    weight_delta_maximum: float
    plasticity_applied: bool
    chemistry_applied: bool


@dataclass(frozen=True)
class DopaminePersistentStats(DopamineStepStats):
    active_neurons: int


class DopamineMatrixRuntime(MatrixRuntime):
    """Causal regional dopamine and three-factor plasticity runtime.

    At entry to tick t, state, concentration and weights are s_t, C_t and W_t.
    C_t modulates the transition to s_(t+1). Eligibility is then updated from
    that transition. The supplied RPE advances chemistry to C_(t+1), which gates
    a W_(t+1) update that can affect only the following neural transition.
    """

    def __init__(
        self,
        topology: SparseTopology,
        dynamics: Dynamics,
        initial_state: torch.Tensor,
        *,
        chemistry: RegionalDopamineField,
        receptors: DopamineReceptors,
        neuron_region: torch.Tensor,
        config: DopaminePlasticityConfig | None = None,
        initial_concentration: torch.Tensor | None = None,
    ) -> None:
        super().__init__(topology, dynamics, initial_state)
        resolved_config = config or DopaminePlasticityConfig()
        if initial_state.ndim != 2:
            raise ValueError("initial_state must have shape [batch, neurons]")
        if topology.n_neurons != dynamics.n_neurons:
            raise ValueError("topology and dynamics neuron counts differ")
        if initial_state.shape[1] != topology.n_neurons:
            raise ValueError("initial_state neuron dimension does not match topology")
        if receptors.n_neurons != topology.n_neurons:
            raise ValueError("receptors must provide one entry per neuron")
        if neuron_region.ndim != 1 or neuron_region.numel() != topology.n_neurons:
            raise ValueError("neuron_region must have one index per neuron")
        if neuron_region.dtype not in (torch.int32, torch.int64):
            raise ValueError("neuron_region must use an integer dtype")
        if neuron_region.device != initial_state.device:
            raise ValueError("neuron_region and state must share a device")
        region64 = neuron_region.to(torch.int64)
        if region64.numel() and (
            bool(torch.any(region64 < 0))
            or int(region64.max()) >= chemistry.n_regions
        ):
            raise ValueError("neuron_region index is outside the chemistry field")
        if chemistry.tonic_baseline.device != initial_state.device:
            raise ValueError("chemistry and neural state must share a device")
        if chemistry.tonic_baseline.dtype != initial_state.dtype:
            raise ValueError("chemistry and neural state must share a dtype")

        batch = initial_state.shape[0]
        if batch != 1 and (
            resolved_config.learning_rate > 0
            or resolved_config.weight_decay > 0
        ):
            raise ValueError(
                "exact shared dopamine-gated online plasticity currently "
                "requires batch=1"
            )
        if initial_concentration is None:
            concentration = chemistry.tonic_baseline.expand(batch, -1).clone()
        else:
            concentration = initial_concentration
        chemistry._validate_concentration(concentration)
        if concentration.shape[0] != batch:
            raise ValueError("concentration batch dimension must match state")

        baseline_per_neuron = chemistry.tonic_baseline.index_select(0, region64)
        baseline_signals = receptors.response(baseline_per_neuron)
        tolerance = 32 * torch.finfo(initial_state.dtype).eps
        if bool(torch.any(baseline_signals.d1_pathway.abs() > tolerance)) or bool(
            torch.any(baseline_signals.d2_pathway.abs() > tolerance)
        ):
            raise ValueError(
                "receptor tonic concentration must match regional tonic baseline"
            )

        self.chemistry = chemistry
        self.receptors = receptors
        self.neuron_region = region64
        self.config = resolved_config
        self.concentration = concentration
        self.pre_trace = torch.zeros_like(initial_state)
        self.post_trace = torch.zeros_like(initial_state)
        self.eligibility = (
            torch.zeros_like(topology.weight)
            if self.config.track_eligibility
            else torch.empty(0, dtype=initial_state.dtype, device=initial_state.device)
        )
        self._empty_weight_delta = torch.empty(
            0, dtype=initial_state.dtype, device=initial_state.device
        )
        self._src = topology.src.to(torch.int64)
        self._dst = topology.dst.to(torch.int64)
        self._tick = 0
        self._chemistry_tick = 0
        self._rpe_accumulator = torch.zeros(
            batch,
            chemistry.n_sources,
            dtype=initial_state.dtype,
            device=initial_state.device,
        )
        self._source_release_accumulator = torch.zeros_like(
            self._rpe_accumulator
        )
        self._has_source_release = False
        self._rpe_delay = torch.zeros(
            self.config.dopamine_delay_steps,
            batch,
            chemistry.n_sources,
            dtype=initial_state.dtype,
            device=initial_state.device,
        )
        self.current_signals = self._signals(self.concentration)
        self._drive_shift, self._gain_scale = self._modulation(
            self.current_signals
        )

    def _validate_step_inputs(
        self,
        rpe: torch.Tensor,
        source_release: torch.Tensor | None,
    ) -> None:
        expected = self._rpe_accumulator.shape
        if (
            rpe.shape != expected
            or rpe.device != self.state.device
            or rpe.dtype != self.state.dtype
        ):
            raise ValueError("rpe must match [batch, dopamine sources]")
        if source_release is not None and (
            source_release.shape != expected
            or source_release.device != self.state.device
            or source_release.dtype != self.state.dtype
        ):
            raise ValueError(
                "source_release must match [batch, dopamine sources]"
            )
        if self.config.validate_inputs:
            if not bool(torch.all(torch.isfinite(rpe))):
                raise ValueError("rpe must be finite")
            if source_release is not None:
                if not bool(torch.all(torch.isfinite(source_release))):
                    raise ValueError("source_release must be finite")
                if bool(torch.any(source_release < 0)):
                    raise ValueError("source_release must be non-negative")

    def _signals(self, concentration: torch.Tensor) -> DopamineSignals:
        neuron_concentration = concentration.index_select(1, self.neuron_region)
        return self.receptors.response_unchecked(neuron_concentration)

    def _modulation(
        self, signals: DopamineSignals
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signal = signals.net_phasic
        drive_shift = torch.clamp(
            self.config.drive_scale * signal,
            min=-self.config.maximum_drive_shift,
            max=self.config.maximum_drive_shift,
        )
        gain_scale = torch.clamp(
            torch.exp(self.config.log_gain_scale * signal),
            min=self.config.minimum_gain_scale,
            max=self.config.maximum_gain_scale,
        )
        return drive_shift, gain_scale

    def _update_eligibility(
        self,
        previous_state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> None:
        if not self.config.track_eligibility:
            return
        config = self.config
        self.pre_trace = (
            config.pre_trace_decay * self.pre_trace
            + (1.0 - config.pre_trace_decay) * previous_state
        )
        self.post_trace = (
            config.post_trace_decay * self.post_trace
            + (1.0 - config.post_trace_decay) * next_state
        )
        potentiation = (
            self.pre_trace.index_select(1, self._src)
            * next_state.index_select(1, self._dst)
        )
        depression = (
            previous_state.index_select(1, self._src)
            * self.post_trace.index_select(1, self._dst)
        )
        local = (
            config.potentiation_scale * potentiation
            - config.depression_scale * depression
        ).mean(dim=0)
        increment = torch.clamp(local, min=-1.0, max=1.0)
        self.eligibility = torch.clamp(
            config.eligibility_decay * self.eligibility
            + (1.0 - config.eligibility_decay) * increment,
            min=-1.0,
            max=1.0,
        )

    def _advance_chemistry_and_weights(
        self,
        rpe: torch.Tensor,
        source_release: torch.Tensor | None,
    ) -> tuple[torch.Tensor, DopamineSignals, torch.Tensor, bool, bool]:
        expected = self._rpe_accumulator.shape
        if rpe.shape != expected or rpe.device != self.state.device:
            raise ValueError("rpe must have shape [batch, dopamine sources]")
        if rpe.dtype != self.state.dtype:
            raise ValueError("rpe dtype must match neural state")
        if self.config.dopamine_delay_steps:
            delay_slot = self._tick % self.config.dopamine_delay_steps
            delivered_rpe = self._rpe_delay[delay_slot].clone()
            self._rpe_delay[delay_slot].copy_(rpe)
        else:
            delivered_rpe = rpe
        self._rpe_accumulator.add_(delivered_rpe)
        if source_release is not None:
            if (
                source_release.shape != expected
                or source_release.device != self.state.device
                or source_release.dtype != self.state.dtype
            ):
                raise ValueError(
                    "source_release must match [batch, dopamine sources]"
                )
            self._source_release_accumulator.add_(source_release)
            self._has_source_release = True

        chemistry_applied = (
            (self._tick + 1) % self.config.chemistry_interval == 0
        )
        if not chemistry_applied:
            return (
                self.concentration,
                self.current_signals,
                self._empty_weight_delta,
                False,
                False,
            )

        scale = 1.0 / self.config.chemistry_interval
        mean_rpe = self._rpe_accumulator * scale
        mean_release = (
            self._source_release_accumulator * scale
            if self._has_source_release
            else None
        )
        next_concentration = self.chemistry.advance_unchecked(
            self.concentration,
            mean_rpe,
            mean_release,
        )
        self._rpe_accumulator.zero_()
        self._source_release_accumulator.zero_()
        self._has_source_release = False
        next_signals = (
            self._signals(next_concentration)
            if self.config.refresh_receptors
            else self.current_signals
        )
        self._chemistry_tick += 1
        plasticity_applied = (
            self._chemistry_tick % self.config.plasticity_interval == 0
        )
        weight_delta = torch.empty(
            0, dtype=self.state.dtype, device=self.state.device
        )
        if self.config.track_eligibility and plasticity_applied and (
            self.config.learning_rate > 0 or self.config.weight_decay > 0
        ):
            weight_delta = self._apply_weight_update(next_signals)
        return (
            next_concentration,
            next_signals,
            weight_delta,
            plasticity_applied,
            True,
        )

    def _apply_weight_update(
        self,
        signals: DopamineSignals,
    ) -> torch.Tensor:
        edge_gate = signals.net_phasic.index_select(1, self._dst).mean(0)
        plasticity_direction = (
            torch.sign(self.topology.weight)
            if self.config.plasticity_updates_magnitude
            else 1.0
        )
        candidate = self.topology.weight * (
            1.0 - self.config.weight_decay
        ) + (
            self.config.learning_rate
            * plasticity_direction
            * edge_gate
            * self.eligibility
        )
        lower = torch.full_like(candidate, self.config.minimum_weight)
        upper = torch.full_like(candidate, self.config.maximum_weight)
        if self.config.preserve_weight_sign:
            zero = torch.zeros_like(candidate)
            lower = torch.where(
                self.topology.weight > 0,
                torch.maximum(lower, zero),
                lower,
            )
            upper = torch.where(
                self.topology.weight < 0,
                torch.minimum(upper, zero),
                upper,
            )
            lower = torch.where(self.topology.weight == 0, zero, lower)
            upper = torch.where(self.topology.weight == 0, zero, upper)
        updated_weight = torch.minimum(torch.maximum(candidate, lower), upper)
        weight_delta = updated_weight - self.topology.weight
        self.topology.apply_source_weights_(updated_weight)
        return weight_delta

    def _step_stats(
        self,
        concentration: torch.Tensor,
        signals: DopamineSignals,
        weight_delta: torch.Tensor,
        plasticity_applied: bool,
        chemistry_applied: bool,
    ) -> DopamineStepStats:
        return DopamineStepStats(
            concentration_minimum=float(concentration.min()),
            concentration_maximum=float(concentration.max()),
            phasic_minimum=float(signals.net_phasic.min()),
            phasic_maximum=float(signals.net_phasic.max()),
            eligibility_maximum=(
                float(self.eligibility.abs().max())
                if self.eligibility.numel()
                else 0.0
            ),
            weight_delta_maximum=(
                float(weight_delta.abs().max()) if weight_delta.numel() else 0.0
            ),
            plasticity_applied=plasticity_applied,
            chemistry_applied=chemistry_applied,
        )

    @torch.no_grad()
    def step(
        self,
        drive: torch.Tensor | float = 0.0,
        *,
        rpe: torch.Tensor,
        source_release: torch.Tensor | None = None,
        collect_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, DopamineStepStats]:
        self._validate_step_inputs(rpe, source_release)
        synaptic_current = self._propagate(self.state)
        previous_state = self.state
        next_state = self.dynamics.advance_modulated(
            previous_state,
            synaptic_current,
            drive,
            drive_shift=self._drive_shift,
            gain_scale=self._gain_scale,
        )
        self._update_eligibility(previous_state, next_state)

        (
            next_concentration,
            next_signals,
            weight_delta,
            plasticity_applied,
            chemistry_applied,
        ) = self._advance_chemistry_and_weights(
            rpe,
            source_release,
        )

        self.state = next_state
        self.concentration = next_concentration
        self.current_signals = next_signals
        if chemistry_applied and self.config.refresh_receptors:
            self._drive_shift, self._gain_scale = self._modulation(next_signals)
        self._tick += 1
        if not collect_stats:
            return self.state
        stats = self._step_stats(
            next_concentration,
            next_signals,
            weight_delta,
            plasticity_applied,
            chemistry_applied,
        )
        return self.state, stats


class DopaminePersistentRuntime(DopamineMatrixRuntime):
    """Dopamine runtime with bounded-error persistent synaptic current."""

    def __init__(
        self,
        topology: SparseTopology,
        dynamics: Dynamics,
        initial_state: torch.Tensor,
        *,
        chemistry: RegionalDopamineField,
        receptors: DopamineReceptors,
        neuron_region: torch.Tensor,
        theta: float | torch.Tensor,
        config: DopaminePlasticityConfig | None = None,
        initial_concentration: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            topology,
            dynamics,
            initial_state,
            chemistry=chemistry,
            receptors=receptors,
            neuron_region=neuron_region,
            config=config,
            initial_concentration=initial_concentration,
        )
        theta_tensor = torch.as_tensor(
            theta,
            dtype=initial_state.dtype,
            device=initial_state.device,
        )
        self.theta = _validate_emission_threshold(theta_tensor, topology)
        self.emitted_state = initial_state.clone()
        self.persistent_current = self._propagate(self.emitted_state).contiguous()
        self._weight_version = topology.weight_version

    @torch.no_grad()
    def step(
        self,
        drive: torch.Tensor | float = 0.0,
        *,
        rpe: torch.Tensor,
        source_release: torch.Tensor | None = None,
        collect_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, DopaminePersistentStats]:
        self._validate_step_inputs(rpe, source_release)
        if self._weight_version != self.topology.weight_version:
            self.persistent_current = self._propagate(
                self.emitted_state
            ).contiguous()
            self._weight_version = self.topology.weight_version
        previous_state = self.state
        next_state = self.dynamics.advance_modulated(
            previous_state,
            self.persistent_current,
            drive,
            drive_shift=self._drive_shift,
            gain_scale=self._gain_scale,
        )

        residual = next_state - self.emitted_state
        active = residual.abs() >= self.theta
        emitted_delta = torch.where(active, residual, torch.zeros_like(residual))
        self.persistent_current = (
            self.persistent_current + self._propagate(emitted_delta)
        ).contiguous()
        self.emitted_state = self.emitted_state + emitted_delta
        self._update_eligibility(previous_state, next_state)

        (
            next_concentration,
            next_signals,
            weight_delta,
            plasticity_applied,
            chemistry_applied,
        ) = self._advance_chemistry_and_weights(rpe, source_release)
        if self.config.track_eligibility and plasticity_applied and (
            self.config.learning_rate > 0 or self.config.weight_decay > 0
        ):
            # Weight changes invalidate the persistent-current cache even when
            # no neuron emitted. Rebuild exactly at the plasticity boundary.
            self.persistent_current = self._propagate(
                self.emitted_state
            ).contiguous()
            self._weight_version = self.topology.weight_version

        self.state = next_state
        self.concentration = next_concentration
        self.current_signals = next_signals
        if chemistry_applied and self.config.refresh_receptors:
            self._drive_shift, self._gain_scale = self._modulation(next_signals)
        self._tick += 1
        if not collect_stats:
            return self.state
        base = self._step_stats(
            next_concentration,
            next_signals,
            weight_delta,
            plasticity_applied,
            chemistry_applied,
        )
        stats = DopaminePersistentStats(
            **base.__dict__,
            active_neurons=int(active.sum()),
        )
        return self.state, stats


class NativeDopamineMatrixRuntime(DopamineMatrixRuntime):
    """Single-kernel sparse propagation and cached dopamine modulation."""

    def __init__(self, *args, blocks: int = 4096, extension=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.state.device.type != "cuda" or self.state.dtype != torch.float32:
            raise ValueError("native dopamine runtime requires CUDA float32 state")
        if blocks <= 0:
            raise ValueError("blocks must be positive")
        if extension is None:
            from .native import load_native_extension

            extension = load_native_extension()
        self.extension = extension
        self.blocks = blocks

    def _update_eligibility(
        self,
        previous_state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> None:
        if not self.config.track_eligibility:
            return
        self.extension.update_eligibility_(
            self.pre_trace,
            self.post_trace,
            self.eligibility,
            previous_state,
            next_state,
            self.topology.src,
            self.topology.dst,
            self.config.pre_trace_decay,
            self.config.post_trace_decay,
            self.config.eligibility_decay,
            self.config.potentiation_scale,
            self.config.depression_scale,
            torch.cuda.current_stream(self.state.device).cuda_stream,
        )

    def _apply_weight_update(
        self,
        signals: DopamineSignals,
    ) -> torch.Tensor:
        maximum_delta = self.extension.dopamine_weight_update_(
            self.topology.weight,
            self.topology.in_weight,
            self.topology.out_weight_index,
            self.eligibility,
            signals.net_phasic,
            self.topology.dst,
            self.config.learning_rate,
            self.config.weight_decay,
            self.config.minimum_weight,
            self.config.maximum_weight,
            self.config.preserve_weight_sign,
            self.config.plasticity_updates_magnitude,
            torch.cuda.current_stream(self.state.device).cuda_stream,
        )
        self.topology.mark_weights_mutated_()
        return maximum_delta

    @torch.no_grad()
    def step(
        self,
        drive: float = 0.0,
        *,
        rpe: torch.Tensor,
        source_release: torch.Tensor | None = None,
        collect_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, DopamineStepStats]:
        self._validate_step_inputs(rpe, source_release)
        previous_state = self.state
        if isinstance(drive, torch.Tensor):
            next_state = self.dynamics.advance_modulated(
                previous_state,
                self._propagate(previous_state),
                drive,
                drive_shift=self._drive_shift,
                gain_scale=self._gain_scale,
            )
        else:
            next_state = self.extension.canonical_modulated_step(
                previous_state,
                self.topology.in_rowptr,
                self.topology.in_col,
                self.topology.in_weight,
                self.dynamics.alpha,
                self.dynamics.gain,
                self._drive_shift.contiguous(),
                self._gain_scale.contiguous(),
                float(drive),
                self.blocks,
                torch.cuda.current_stream(self.state.device).cuda_stream,
            )
        self._update_eligibility(previous_state, next_state)
        (
            next_concentration,
            next_signals,
            weight_delta,
            plasticity_applied,
            chemistry_applied,
        ) = self._advance_chemistry_and_weights(rpe, source_release)
        self.state = next_state
        self.concentration = next_concentration
        self.current_signals = next_signals
        if chemistry_applied and self.config.refresh_receptors:
            self._drive_shift, self._gain_scale = self._modulation(next_signals)
        self._tick += 1
        if not collect_stats:
            return self.state
        return self.state, self._step_stats(
            next_concentration,
            next_signals,
            weight_delta,
            plasticity_applied,
            chemistry_applied,
        )


class NativeDopamineEventRuntime(DopaminePersistentRuntime):
    """Native sparse-event PELS with cached dopamine modulation."""

    def __init__(
        self,
        *args,
        scatter_blocks: int = 128,
        extension=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.state.device.type != "cuda" or self.state.dtype != torch.float32:
            raise ValueError("native dopamine runtime requires CUDA float32 state")
        if scatter_blocks <= 0:
            raise ValueError("scatter_blocks must be positive")
        if extension is None:
            from .native import load_native_extension

            extension = load_native_extension()
        self.extension = extension
        self.scatter_blocks = scatter_blocks
        self.active_index = torch.empty(
            self.state.numel(), dtype=torch.int64, device=self.state.device
        )
        self.active_delta = torch.empty_like(self.state).flatten()
        self.active_count = torch.zeros(
            1, dtype=torch.int32, device=self.state.device
        )

    def _update_eligibility(
        self,
        previous_state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> None:
        if not self.config.track_eligibility:
            return
        self.extension.update_eligibility_(
            self.pre_trace,
            self.post_trace,
            self.eligibility,
            previous_state,
            next_state,
            self.topology.src,
            self.topology.dst,
            self.config.pre_trace_decay,
            self.config.post_trace_decay,
            self.config.eligibility_decay,
            self.config.potentiation_scale,
            self.config.depression_scale,
            torch.cuda.current_stream(self.state.device).cuda_stream,
        )

    def _apply_weight_update(self, signals: DopamineSignals) -> torch.Tensor:
        maximum_delta = self.extension.dopamine_weight_update_(
            self.topology.weight,
            self.topology.in_weight,
            self.topology.out_weight_index,
            self.eligibility,
            signals.net_phasic,
            self.topology.dst,
            self.config.learning_rate,
            self.config.weight_decay,
            self.config.minimum_weight,
            self.config.maximum_weight,
            self.config.preserve_weight_sign,
            self.config.plasticity_updates_magnitude,
            torch.cuda.current_stream(self.state.device).cuda_stream,
        )
        self.topology.mark_weights_mutated_()
        return maximum_delta

    @torch.no_grad()
    def step(
        self,
        drive: float = 0.0,
        *,
        rpe: torch.Tensor,
        source_release: torch.Tensor | None = None,
        collect_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, DopaminePersistentStats]:
        self._validate_step_inputs(rpe, source_release)
        if self._weight_version != self.topology.weight_version:
            self.persistent_current = self._propagate(
                self.emitted_state
            ).contiguous()
            self._weight_version = self.topology.weight_version
        previous_state = self.state
        if isinstance(drive, torch.Tensor):
            next_state = self.dynamics.advance_modulated(
                previous_state,
                self.persistent_current,
                drive,
                drive_shift=self._drive_shift,
                gain_scale=self._gain_scale,
            )
        else:
            next_state = self.extension.pels_dopamine_event_step_(
                previous_state,
                self.persistent_current,
                self.emitted_state,
                self.dynamics.alpha,
                self.dynamics.gain,
                self._drive_shift.contiguous(),
                self._gain_scale.contiguous(),
                self.topology.out_rowptr,
                self.topology.dst,
                self.topology.out_weight_index,
                self.topology.in_weight,
                self.theta.contiguous(),
                self.active_index,
                self.active_delta,
                self.active_count,
                float(drive),
                self.scatter_blocks,
                torch.cuda.current_stream(self.state.device).cuda_stream,
            )
        if isinstance(drive, torch.Tensor):
            residual = next_state - self.emitted_state
            active = residual.abs() >= self.theta
            emitted_delta = torch.where(active, residual, torch.zeros_like(residual))
            self.persistent_current = (
                self.persistent_current + self._propagate(emitted_delta)
            ).contiguous()
            self.emitted_state = self.emitted_state + emitted_delta
        self._update_eligibility(previous_state, next_state)
        (
            next_concentration,
            next_signals,
            weight_delta,
            plasticity_applied,
            chemistry_applied,
        ) = self._advance_chemistry_and_weights(rpe, source_release)
        if self.config.track_eligibility and plasticity_applied and (
            self.config.learning_rate > 0 or self.config.weight_decay > 0
        ):
            self.persistent_current = self._propagate(
                self.emitted_state
            ).contiguous()
            self._weight_version = self.topology.weight_version
        self.state = next_state
        self.concentration = next_concentration
        self.current_signals = next_signals
        if chemistry_applied and self.config.refresh_receptors:
            self._drive_shift, self._gain_scale = self._modulation(next_signals)
        self._tick += 1
        if not collect_stats:
            return self.state
        base = self._step_stats(
            next_concentration,
            next_signals,
            weight_delta,
            plasticity_applied,
            chemistry_applied,
        )
        return self.state, DopaminePersistentStats(
            **base.__dict__,
            active_neurons=int(self.active_count.item()),
        )
