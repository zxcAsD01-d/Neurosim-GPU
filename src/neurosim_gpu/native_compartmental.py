"""Explicit CUDA inference accelerators with differentiable reference fallback."""

from __future__ import annotations

import torch

from .compartmental import DendriticDynamics, DendriticMatrixRuntime, DendriticPELSRuntime
from .native import load_native_extension
from .runtime import RuntimeStats
from .topology import SparseTopology


class _NativeCompartments:
    def _init_native(self, initial_state: torch.Tensor, extension) -> None:
        if initial_state.device.type != "cuda" or initial_state.dtype != torch.float32:
            raise ValueError("native compartment runtime requires CUDA float32")
        self.extension = extension if extension is not None else load_native_extension()
        self.last_backend = "unstepped"

    def _parameters(self) -> list[torch.Tensor]:
        model = self.compartment_dynamics
        return [model.soma.alpha, model.soma.gain, model.dendrite_decay,
                model.adaptation_decay, model.coupling, model.adaptation_strength]

    def _needs_autograd(self, drive, dendritic_drive) -> bool:
        if not torch.is_grad_enabled():
            return False
        values = [self.state, self.dendritic_state, self.adaptation_state,
                  self.topology.weight, self.topology.in_weight,
                  *self._parameters(), drive, dendritic_drive]
        if hasattr(self, "persistent_current"):
            values += [self.persistent_current, self.emitted_state]
        return any(isinstance(value, torch.Tensor) and value.requires_grad for value in values)

    def _drive(self, value: torch.Tensor | float) -> tuple[torch.Tensor | None, float]:
        if not isinstance(value, torch.Tensor):
            return None, float(value)
        if value.device != self.state.device or value.dtype != self.state.dtype:
            raise ValueError("drive must match state device and dtype")
        try:
            return value.expand_as(self.state), 0.0
        except RuntimeError as error:
            raise ValueError("drive must broadcast to [batch, n_neurons]") from error


class NativeDendriticMatrixRuntime(_NativeCompartments, DendriticMatrixRuntime):
    """One CUDA launch for CSR propagation plus all three neuron states.

    States, parameters, drives and accumulation are CUDA float32. Optional
    FP16, FP8 and NVFP4 CSR storage decode each weight before FP32 multiplication.
    Reduced storage is inference-only. Float32 supports a differentiable fallback.
    Returned states remain valid after later ticks (no ping-pong aliasing).
    ``threads_per_row`` chooses 8/16/32 lanes; None uses mean degree.
    """

    def __init__(self, topology: SparseTopology, dynamics: DendriticDynamics,
                 initial_state: torch.Tensor, *, threads_per_row: int | None = None,
                 extension=None, weight_dtype: torch.dtype = torch.float32,
                 weight_format: str | None = None) -> None:
        if threads_per_row not in (None, 8, 16, 32):
            raise ValueError("threads_per_row must be 8, 16, 32 or None")
        if initial_state.device.type != "cuda" or initial_state.dtype != torch.float32:
            raise ValueError("native compartment runtime requires CUDA float32")
        if weight_dtype not in (torch.float32, torch.float16):
            raise ValueError("weight_dtype must be torch.float32 or torch.float16")
        if weight_format not in (None, "fp8", "nvfp4"):
            raise ValueError("weight_format must be None, 'fp8' or 'nvfp4'")
        if weight_format is not None and weight_dtype != torch.float32:
            raise ValueError("weight_format and non-default weight_dtype conflict")
        super().__init__(topology, dynamics, initial_state)
        degree = topology.n_edges / max(topology.n_neurons, 1)
        self.threads_per_row = threads_per_row or (8 if degree <= 16 else 16 if degree <= 48 else 32)
        self._init_native(initial_state, extension)
        self.weight_dtype = weight_dtype
        self.weight_format = weight_format
        self._invalidate_precision_cache()

    def _invalidate_precision_cache(self) -> None:
        self._packed_weight = None
        self._packed_weight_version = -1

    @property
    def checkpoint_precision(self) -> str:
        if self.weight_format is not None:
            return self.weight_format
        return "fp16" if self.weight_dtype == torch.float16 else "fp32"

    @property
    def precision_execution(self) -> str:
        return f"cuda-fused-{self.checkpoint_precision}-weights-fp32-fma"

    @property
    def weight_storage_bytes(self) -> int:
        """Logical inference payload in bytes, excluding the original topology."""
        edges = self.topology.n_edges
        if self.weight_format == "nvfp4":
            return (edges + 1) // 2 + (edges + 15) // 16 + 4
        if self.weight_format == "fp8":
            return edges + 4
        return edges * (2 if self.weight_dtype == torch.float16 else 4)

    def step(self, drive: torch.Tensor | float = 0.0, *,
             dendritic_drive: torch.Tensor | float = 0.0) -> torch.Tensor:
        if self._needs_autograd(drive, dendritic_drive):
            if self.weight_format is not None or self.weight_dtype == torch.float16:
                raise RuntimeError("native reduced-precision weight storage is inference-only")
            self.last_backend = "reference-autograd"
            return super().step(drive, dendritic_drive=dendritic_drive)
        u, us = self._drive(drive)
        v, vs = self._drive(dendritic_drive)
        weight = self.topology.in_weight
        scale_args = ()
        backend = "native"
        if self.weight_format is not None or self.weight_dtype == torch.float16:
            if self._packed_weight_version != self.topology.weight_version:
                if self.weight_format is None:
                    self._packed_weight = (weight.to(torch.float16),)
                else:
                    from .precision import encode_sparse_weights

                    self._packed_weight = encode_sparse_weights(weight, self.weight_format)
                self._packed_weight_version = self.topology.weight_version
            weight, *scale_args = self._packed_weight
            backend = f"native-{self.weight_format or 'fp16'}-weight"
        self.state, self.dendritic_state, self.adaptation_state = self.extension.dendritic_step(
            self.state, self.dendritic_state, self.adaptation_state, self._parameters(),
            self.topology.in_rowptr, self.topology.in_col, weight,
            u, us, v, vs, self.threads_per_row, *scale_args,
        )
        self.last_backend = backend
        return self.state


class NativeDendriticPELSRuntime(_NativeCompartments, DendriticPELSRuntime):
    """Two CUDA launches: compartment advance/emission, then sparse edge scatter.

    Scans B*N sources on device and traverses outgoing edges only for nonzero
    emitted deltas. No host counts or event compaction. FP32 atomics make scatter
    nondeterministic. Statistics and autograd requests use the reference path;
    neither fallback silently drops gradients. Checkpoints include all states.
    """

    @property
    def precision_execution(self) -> str:
        return "cuda-fused-pels-fp32"

    def __init__(self, topology: SparseTopology, dynamics: DendriticDynamics,
                 initial_state: torch.Tensor, *, theta: float | torch.Tensor,
                 extension=None) -> None:
        if initial_state.device.type != "cuda" or initial_state.dtype != torch.float32:
            raise ValueError("native compartment runtime requires CUDA float32")
        super().__init__(topology, dynamics, initial_state, theta=theta, propagation="event")
        self.persistent_current = self.persistent_current.contiguous()
        self._init_native(initial_state, extension)

    def step(self, drive: torch.Tensor | float = 0.0, *,
             dendritic_drive: torch.Tensor | float = 0.0,
             collect_stats: bool = False) -> torch.Tensor | tuple[torch.Tensor, RuntimeStats]:
        if collect_stats or self._needs_autograd(drive, dendritic_drive):
            self.last_backend = "reference-stats" if collect_stats else "reference-autograd"
            return super().step(drive, dendritic_drive=dendritic_drive, collect_stats=collect_stats)
        u, us = self._drive(drive)
        v, vs = self._drive(dendritic_drive)
        self._refresh_current()
        self.persistent_current = self.persistent_current.contiguous()
        # Current is updated in-place only after the advance kernel has read it.
        self.state, self.dendritic_state, self.adaptation_state, self.emitted_state = (
            self.extension.dendritic_event(
                self.state, self.dendritic_state, self.adaptation_state, self._parameters(),
                self.persistent_current, self.emitted_state, self.theta,
                self.topology.out_rowptr, self.topology.dst, self.topology.out_weight_index,
                self.topology.in_weight, u, us, v, vs,
            )
        )
        self._tick += 1
        self.last_backend = "native"
        return self.state
