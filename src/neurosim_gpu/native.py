from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import torch

from .dynamics import Dynamics
from .runtime import MatrixRuntime, _validate_emission_threshold
from .topology import SparseTopology


def _host_flags() -> list[str]:
    """Host compiler flags for the current platform.

    MSVC accepts ``/O2`` and rejects ``-O3``; gcc and clang do the opposite.
    ``NEUROSIM_HOST_FLAGS`` overrides the default when a toolchain needs it.
    """

    override = os.environ.get("NEUROSIM_HOST_FLAGS")
    if override:
        return override.split()
    if sys.platform == "win32":
        return ["/O2"]
    return ["-O3"]


def _cuda_flags() -> list[str]:
    override = os.environ.get("NEUROSIM_CUDA_FLAGS")
    if override:
        return override.split()
    return ["-O3", "--use_fast_math"]


def _native_sources() -> list[str]:
    """Locate the extension sources.

    The kernels live in the repository's ``csrc/`` directory, which is part of a
    source checkout (editable install) but not of a prebuilt wheel. A packaged
    copy next to the module is accepted as a fallback.
    """

    here = Path(__file__).resolve()
    candidates = [here.parents[2] / "csrc", here.parent / "csrc"]
    for directory in candidates:
        sources = [directory / "bindings.cpp", directory / "runtime.cu"]
        if all(source.exists() for source in sources):
            return [str(source) for source in sources]
    raise RuntimeError(
        "native extension sources (csrc/bindings.cpp, csrc/runtime.cu) were not "
        "found. The native kernels require a source checkout: install with "
        "`pip install -e .` from a clone. See docs/native-build.md."
    )


@lru_cache(maxsize=1)
def load_native_extension(verbose: bool = False):
    from torch.utils.cpp_extension import load

    return load(
        name="neurosim_gpu_native",
        sources=_native_sources(),
        extra_cflags=_host_flags(),
        extra_cuda_cflags=_cuda_flags(),
        verbose=verbose,
    )


class _NativeAdvanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, current, alpha, gain, drive, extension):
        if alpha.requires_grad or gain.requires_grad:
            raise RuntimeError("native training advance treats alpha and gain as fixed")
        output, state_derivative, current_derivative = (
            extension.advance_train_forward(
                state,
                current,
                alpha,
                gain,
                float(drive),
                torch.cuda.current_stream(state.device).cuda_stream,
            )
        )
        ctx.save_for_backward(state_derivative, current_derivative)
        ctx.extension = extension
        return output

    @staticmethod
    def backward(ctx, grad_output):
        state_derivative, current_derivative = ctx.saved_tensors
        grad_state, grad_current = ctx.extension.advance_train_backward(
            grad_output.contiguous(),
            state_derivative,
            current_derivative,
            torch.cuda.current_stream(grad_output.device).cuda_stream,
        )
        return grad_state, grad_current, None, None, None, None


def native_advance_training(
    state: torch.Tensor,
    current: torch.Tensor,
    dynamics: Dynamics,
    drive: float,
    extension=None,
) -> torch.Tensor:
    extension = extension or load_native_extension()
    return _NativeAdvanceFunction.apply(
        state,
        current,
        dynamics.alpha,
        dynamics.gain,
        float(drive),
        extension,
    )


class NativeMatrixRuntime(MatrixRuntime):
    """CSR propagation followed by one fused native neuron-update kernel."""

    def __init__(
        self,
        topology: SparseTopology,
        dynamics: Dynamics,
        initial_state: torch.Tensor,
        *,
        extension=None,
    ) -> None:
        if initial_state.device.type != "cuda" or initial_state.dtype != torch.float32:
            raise ValueError("native runtime requires CUDA float32 tensors")
        super().__init__(topology, dynamics, initial_state.contiguous())
        self.extension = extension or load_native_extension()

    def step(self, drive: float = 0.0) -> torch.Tensor:
        if torch.is_grad_enabled() and self.state.requires_grad:
            raise RuntimeError("native inference kernels do not implement autograd")
        if isinstance(drive, torch.Tensor):
            current = self._propagate(self.state)
            self.state = self.dynamics.advance(self.state, current, drive)
            return self.state
        current = self._propagate(self.state)
        self.state = self.extension.advance(
            self.state,
            current,
            self.dynamics.alpha,
            self.dynamics.gain,
            float(drive),
            torch.cuda.current_stream(self.state.device).cuda_stream,
        )
        return self.state


class NativeCanonicalLinear:
    """Raw same-parameter sparse linear layer using one canonical weight array."""

    def __init__(
        self,
        topology: SparseTopology,
        initial_state: torch.Tensor,
        *,
        blocks: int = 4096,
        extension=None,
    ) -> None:
        if initial_state.device.type != "cuda" or initial_state.dtype != torch.float32:
            raise ValueError("native runtime requires CUDA float32 tensors")
        self.topology = topology
        self.state = initial_state.contiguous()
        self.output = torch.empty_like(self.state)
        self.blocks = blocks
        self.extension = extension or load_native_extension()

    def step(self, drive: float = 0.0) -> torch.Tensor:
        del drive
        self.output = self.extension.canonical_spmm(
            self.state,
            self.topology.in_rowptr,
            self.topology.in_col,
            self.topology.in_weight,
            self.blocks,
            torch.cuda.current_stream(self.state.device).cuda_stream,
        )
        return self.output


class NativeCanonicalMatrixRuntime:
    """Single-kernel canonical SpMM fused with the full neural state update."""

    def __init__(
        self,
        topology: SparseTopology,
        dynamics: Dynamics,
        initial_state: torch.Tensor,
        *,
        blocks: int = 4096,
        extension=None,
    ) -> None:
        if initial_state.device.type != "cuda" or initial_state.dtype != torch.float32:
            raise ValueError("native runtime requires CUDA float32 tensors")
        self.topology = topology
        self.dynamics = dynamics
        self.state = initial_state.contiguous()
        self.blocks = blocks
        self.extension = extension or load_native_extension()

    def step(self, drive: float = 0.0) -> torch.Tensor:
        if isinstance(drive, torch.Tensor):
            current = self._propagate(self.state)
            self.state = self.dynamics.advance(self.state, current, drive)
            return self.state
        if torch.is_grad_enabled() and self.state.requires_grad:
            raise RuntimeError("native inference kernels do not implement autograd")
        self.state = self.extension.canonical_step(
            self.state,
            self.topology.in_rowptr,
            self.topology.in_col,
            self.topology.in_weight,
            self.dynamics.alpha,
            self.dynamics.gain,
            float(drive),
            self.blocks,
            torch.cuda.current_stream(self.state.device).cuda_stream,
        )
        return self.state


class NativeEventPELSRuntime(MatrixRuntime):
    """Two-kernel PELS path with no per-tick device-to-host synchronization."""

    def __init__(
        self,
        topology: SparseTopology,
        dynamics: Dynamics,
        initial_state: torch.Tensor,
        *,
        theta: float | torch.Tensor,
        scatter_blocks: int = 128,
        extension=None,
    ) -> None:
        if initial_state.device.type != "cuda" or initial_state.dtype != torch.float32:
            raise ValueError("native runtime requires CUDA float32 tensors")
        super().__init__(topology, dynamics, initial_state.contiguous())
        self.extension = extension or load_native_extension()
        theta_tensor = torch.as_tensor(
            theta,
            dtype=initial_state.dtype,
            device=initial_state.device,
        )
        self.theta = _validate_emission_threshold(
            theta_tensor, topology
        ).contiguous()
        self.scatter_blocks = scatter_blocks
        self.emitted_state = self.state.clone()
        self.persistent_current = self._propagate(self.emitted_state).contiguous()
        self.active_index = torch.empty(
            self.state.numel(),
            dtype=torch.int64,
            device=self.state.device,
        )
        self.active_delta = torch.empty_like(self.state).flatten()
        self.active_count = torch.zeros(
            1,
            dtype=torch.int32,
            device=self.state.device,
        )
        self._weight_version = topology.weight_version

    def step(self, drive: float = 0.0) -> torch.Tensor:
        if torch.is_grad_enabled() and self.state.requires_grad:
            raise RuntimeError("native inference kernels do not implement autograd")
        if isinstance(drive, torch.Tensor):
            if self._weight_version != self.topology.weight_version:
                self.persistent_current = self._propagate(
                    self.emitted_state
                ).contiguous()
                self._weight_version = self.topology.weight_version
            next_state = self.dynamics.advance(
                self.state, self.persistent_current, drive
            )
            residual = next_state - self.emitted_state
            active = residual.abs() >= self.theta
            emitted_delta = torch.where(active, residual, torch.zeros_like(residual))
            self.persistent_current = (
                self.persistent_current + self._propagate(emitted_delta)
            ).contiguous()
            self.emitted_state = self.emitted_state + emitted_delta
            self.state = next_state
            return self.state
        if self._weight_version != self.topology.weight_version:
            self.persistent_current = self._propagate(
                self.emitted_state
            ).contiguous()
            self._weight_version = self.topology.weight_version
        self.extension.pels_event_step_(
            self.state,
            self.persistent_current,
            self.emitted_state,
            self.dynamics.alpha,
            self.dynamics.gain,
            self.topology.out_rowptr,
            self.topology.dst,
            self.topology.out_weight_index,
            self.topology.in_weight,
            self.theta,
            self.active_index,
            self.active_delta,
            self.active_count,
            float(drive),
            self.scatter_blocks,
            torch.cuda.current_stream(self.state.device).cuda_stream,
        )
        return self.state

    def read_active_count(self) -> int:
        return int(self.active_count.item())
