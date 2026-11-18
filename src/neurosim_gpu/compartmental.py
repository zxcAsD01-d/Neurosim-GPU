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
from .precision import Precision, quantize_dequantize, validate_precision
from .runtime import HybridPELSRuntime, MatrixRuntime, PropagationMode, RuntimeStats
from .topology import SparseTopology



def _csr(row: torch.Tensor, col: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    n = row.numel() - 1
    return torch.sparse_csr_tensor(row, col, values, size=(n, n), check_invariants=False)


class _FP16SpMM(torch.autograd.Function):
    """FP16 cuSPARSE forward and FP32 derivatives using the rounded operands."""

    @staticmethod
    def forward(ctx, weights, source, matrix, topology):
        source_half = source.half()
        ctx.save_for_backward(matrix, source_half)
        ctx.topology = topology
        return torch.sparse.mm(matrix, source_half.T).T.float()

    @staticmethod
    def backward(ctx, gradient):
        matrix, source_half = ctx.saved_tensors
        topology = ctx.topology
        edge_gradient = source_gradient = None
        with torch.autocast(device_type="cuda", enabled=False):
            if ctx.needs_input_grad[0]:
                # CUDA's sampled_addmm lacks Half support. SDDMM computes only
                # the stored CSR entries and never materializes an N-by-N gradient.
                edge_gradient = torch.sparse.sampled_addmm(
                    matrix.float(), gradient.T.contiguous(), source_half.float().contiguous(),
                    beta=0.0,
                ).values()
            if ctx.needs_input_grad[1]:
                transposed = _csr(
                    topology.out_rowptr, topology.dst,
                    matrix.values().float().index_select(0, topology.out_weight_index),
                )
                source_gradient = torch.sparse.mm(transposed, gradient.T).T
        return edge_gradient, source_gradient, None, None


_PRECISION_EXECUTION = {
    "fp32": "torch-csr-input-dtype",
    "fp16": "torch-csr-fp16-forward-fp32-backward",
    "fp8": "torch-csr-fp32-with-fp8-e4m3-qdq-emulation",
    "nvfp4": "torch-csr-fp32-with-nvfp4-e2m1-qdq-emulation",
}


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

    @property
    def precision_execution(self) -> str:
        return _PRECISION_EXECUTION[self.precision]

    def _init_precision(self, initial_state: torch.Tensor, precision: str) -> None:
        self.precision: Precision = validate_precision(precision)
        if self.precision != "fp32" and initial_state.dtype != torch.float32:
            raise ValueError("reduced precision requires FP32 master weights and states")
        if self.precision == "fp16" and not initial_state.is_cuda:
            raise ValueError("FP16 CSR propagation requires CUDA")
        self._invalidate_precision_cache()

    def _invalidate_precision_cache(self) -> None:
        self._precision_cache = None

    def _propagate(self, source: torch.Tensor) -> torch.Tensor:
        if self.precision == "fp32":
            return super()._propagate(source)
        topology = self.topology
        train_weight = torch.is_grad_enabled() and topology.in_weight.requires_grad
        version = topology.weight_version
        cached = self._precision_cache
        grad = torch.is_grad_enabled() and (train_weight or source.requires_grad)
        if (not train_weight and cached is not None and cached[0] == version
                and not (grad and cached[1].is_inference())):
            matrix = cached[1]
        else:
            values = (topology.in_weight.detach().half() if self.precision == "fp16" else
                      quantize_dequantize(topology.in_weight, self.precision))
            matrix = _csr(topology.in_rowptr, topology.in_col, values)
            if not train_weight:
                self._precision_cache = (version, matrix)
        with torch.autocast(device_type=source.device.type, enabled=False):
            if self.precision == "fp16":
                return (_FP16SpMM.apply(topology.in_weight, source, matrix, topology) if grad else
                        torch.sparse.mm(matrix, source.half().T).T.float())
            operand = quantize_dequantize(source, self.precision)
            return torch.sparse.mm(matrix, operand.T).T

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
                if self.precision != "fp32" and (value.dtype != self.state.dtype or value.device != self.state.device):
                    raise ValueError(f"{name} must match state device and dtype")
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
                 initial_state: torch.Tensor, *, precision: str = "fp32") -> None:
        self._init_precision(initial_state, precision)
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
        precision: str = "fp32",
    ) -> None:
        self._init_precision(initial_state, precision)
        if self.precision != "fp32" and propagation != "matrix":
            raise ValueError("reduced-precision PELS requires propagation='matrix'")
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
        if self.precision != "fp32":
            residual = state - self.emitted_state
            active = residual.detach().abs() >= self.theta
            self.emitted_state = torch.where(active, state, self.emitted_state)
            self.state = state
            self.persistent_current = self._propagate(self.emitted_state).contiguous()
            self._tick += 1
            if not collect_stats:
                return self.state
            active_pairs = torch.nonzero(active, as_tuple=False)
            active_edges = int(
                self.topology.out_degree[active_pairs[:, 1]].sum().item()
            ) if active_pairs.numel() else 0
            return self.state, RuntimeStats(
                path="matrix-precision",
                active_neurons=int(active.sum().item()),
                active_edges=active_edges,
            )
        return self._emit(state, collect_stats=collect_stats)
