"""Checkpoint and restore helpers for stateful simulation runtimes.

The helpers deliberately serialize tensors on CPU and keep topology metadata in
the compact topology schema. They are intended for episode boundaries,
preemption, and deterministic research replay; they are not a distributed
checkpoint protocol.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any, BinaryIO, Mapping

import torch

from .communication import DelayedMultiChannelCommunication
from .topology import SparseTopology
from .topology_io import topology_from_dict, topology_to_dict


FORMAT = "neurosim_gpu.runtime_checkpoint"
VERSION = 1

_TENSOR_FIELDS = (
    "state",
    "emitted_state",
    "persistent_current",
    "concentration",
    "pre_trace",
    "post_trace",
    "eligibility",
    "_rpe_accumulator",
    "_source_release_accumulator",
    "_rpe_delay",
    "_drive_shift",
    "_gain_scale",
    "theta",
)
_SCALAR_FIELDS = (
    "_tick",
    "_chemistry_tick",
    "_weight_version",
    "_cursor",
    "_has_source_release",
    "_chemistry_applied",
)


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").clone()


def _runtime_signature(runtime: Any) -> str:
    """Hash immutable dynamics/chemistry/receptor/config parameters."""

    digest = hashlib.sha256()

    def add_tensor(value: torch.Tensor) -> None:
        # Clone after the device transfer so a CPU view cannot retain a larger
        # backing storage than its logical tensor extent.
        cpu = value.detach().to(device="cpu").contiguous().clone()
        digest.update(str(cpu.dtype).encode())
        digest.update(repr(tuple(cpu.shape)).encode())
        # Hash the raw tensor storage without routing through ``Tensor.numpy``.
        # Checkpoint validation must work in minimal CUDA/PyTorch environments
        # where NumPy is intentionally not installed.
        digest.update(bytes(cpu.untyped_storage()))

    def add_value(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            add_tensor(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            digest.update(json.dumps(value, sort_keys=True).encode())
        elif isinstance(value, (tuple, list)):
            for item in value:
                add_value(item)
        elif hasattr(value, "__dataclass_fields__"):
            for name in value.__dataclass_fields__:
                digest.update(name.encode())
                add_value(getattr(value, name))

    dynamics = getattr(runtime, "dynamics", None)
    if dynamics is not None:
        add_value(dynamics.alpha)
        add_value(dynamics.gain)
    chemistry = getattr(runtime, "chemistry", None)
    if chemistry is not None:
        for name in (
            "tonic_baseline", "uptake_rate", "decay_rate", "release_projection",
            "edge_src", "edge_dst", "diffusion_rate", "region_volume",
            "burst_gain", "dip_gain", "autoreceptor_gain",
            "autoreceptor_half_saturation",
        ):
            if hasattr(chemistry, name):
                add_value(getattr(chemistry, name))
    receptors = getattr(runtime, "receptors", None)
    if receptors is not None:
        add_value(receptors)
    if hasattr(runtime, "neuron_region"):
        add_value(runtime.neuron_region)
    if hasattr(runtime, "config"):
        add_value(runtime.config)
    if hasattr(runtime, "theta"):
        add_value(runtime.theta)
    return digest.hexdigest()


def capture_runtime(
    runtime: Any,
    *,
    communication: DelayedMultiChannelCommunication | None = None,
) -> dict[str, Any]:
    """Capture a runtime and optional communication history into a CPU payload."""

    topology = getattr(runtime, "topology", None)
    if not isinstance(topology, SparseTopology):
        raise ValueError("runtime must own a SparseTopology")
    payload: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "runtime_type": type(runtime).__name__,
        "runtime_signature": _runtime_signature(runtime),
        "topology": topology_to_dict(topology),
        "tensors": {},
        "scalars": {},
    }
    tensors: dict[str, torch.Tensor] = payload["tensors"]
    for name in _TENSOR_FIELDS:
        value = getattr(runtime, name, None)
        if isinstance(value, torch.Tensor):
            tensors[name] = _cpu_clone(value)
    scalars: dict[str, Any] = payload["scalars"]
    for name in _SCALAR_FIELDS:
        if hasattr(runtime, name):
            value = getattr(runtime, name)
            if isinstance(value, (bool, int, float)):
                scalars[name] = value

    if communication is not None:
        payload["communication"] = _capture_communication(communication)
    return payload


def _capture_communication(
    communication: DelayedMultiChannelCommunication,
) -> dict[str, Any]:
    history = communication.history
    return {
        "tick": int(communication.tick),
        "cursor": int(communication._cursor),
        "history": _cpu_clone(history) if history.numel() else history.cpu().clone(),
    }


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("format") != FORMAT or payload.get("version") != VERSION:
        raise ValueError("unsupported runtime checkpoint format or version")
    if not isinstance(payload.get("topology"), Mapping):
        raise ValueError("checkpoint is missing topology payload")
    if not isinstance(payload.get("tensors"), Mapping):
        raise ValueError("checkpoint is missing tensor payload")
    if not isinstance(payload.get("scalars"), Mapping):
        raise ValueError("checkpoint is missing scalar payload")


def _copy_tensor_field(runtime: Any, name: str, value: torch.Tensor) -> None:
    current = getattr(runtime, name, None)
    if not isinstance(current, torch.Tensor):
        raise ValueError(f"runtime does not expose checkpoint tensor {name}")
    if current.shape != value.shape:
        raise ValueError(
            f"checkpoint tensor {name} shape {tuple(value.shape)} does not "
            f"match runtime shape {tuple(current.shape)}"
        )
    if current.dtype != value.dtype:
        raise ValueError(f"checkpoint tensor {name} dtype does not match runtime")
    # Rebinding avoids failures when the target is an expanded zero-stride view
    # (for example a batch-broadcasted theta or modulation tensor).
    setattr(runtime, name, value.to(device=current.device).clone())


def restore_runtime(
    runtime: Any,
    payload: Mapping[str, Any],
    *,
    communication: DelayedMultiChannelCommunication | None = None,
    strict_runtime_type: bool = False,
) -> None:
    """Restore a captured payload into an already-constructed runtime.

    Topology structure must match. Weights are restored through the topology
    mutation API, then the saved version is applied so persistent caches can
    detect later external changes normally.
    """

    _validate_payload(payload)
    if strict_runtime_type and payload.get("runtime_type") != type(runtime).__name__:
        raise ValueError("checkpoint runtime type does not match target runtime")
    saved_signature = payload.get("runtime_signature")
    if saved_signature is not None and saved_signature != _runtime_signature(runtime):
        raise ValueError("checkpoint runtime parameters differ from target runtime")
    saved_topology = topology_from_dict(payload["topology"], device="cpu")
    target_topology = getattr(runtime, "topology", None)
    if not isinstance(target_topology, SparseTopology):
        raise ValueError("runtime must own a SparseTopology")
    if (
        saved_topology.n_neurons != target_topology.n_neurons
        or saved_topology.n_edges != target_topology.n_edges
    ):
        raise ValueError("checkpoint topology shape does not match runtime")
    for name in ("src", "dst", "out_rowptr", "in_rowptr", "in_col", "out_weight_index"):
        if not torch.equal(
            getattr(saved_topology, name).cpu(), getattr(target_topology, name).cpu()
        ):
            raise ValueError(f"checkpoint topology structure differs in {name}")

    target_topology.apply_source_weights_(
        saved_topology.weight.to(device=target_topology.weight.device)
    )
    saved_version = int(payload["scalars"].get("_weight_version", 0))
    object.__setattr__(target_topology, "weight_version", saved_version)

    with torch.no_grad():
        for name, value in payload["tensors"].items():
            _copy_tensor_field(runtime, name, value)
    for name, value in payload["scalars"].items():
        if name == "_weight_version":
            continue
        if hasattr(runtime, name):
            setattr(runtime, name, value)

    # Rebuild derived receptor caches from restored concentration rather than
    # trusting constructor-time values from the target runtime.
    if (
        hasattr(runtime, "concentration")
        and hasattr(runtime, "_signals")
        and hasattr(runtime, "_modulation")
    ):
        runtime.current_signals = runtime._signals(runtime.concentration)
        runtime._drive_shift, runtime._gain_scale = runtime._modulation(
            runtime.current_signals
        )

    if communication is not None and "communication" in payload:
        _restore_communication(communication, payload["communication"])


def _restore_communication(
    communication: DelayedMultiChannelCommunication,
    payload: Mapping[str, Any],
) -> None:
    history = payload.get("history")
    if not isinstance(history, torch.Tensor):
        raise ValueError("communication checkpoint history is missing")
    if history.numel():
        communication.history = history.to(
            device=communication.weight.device,
            dtype=communication.weight.dtype,
        ).contiguous()
    else:
        communication.history = communication.weight.new_empty((0,))
    communication._cursor = int(payload.get("cursor", 0))
    communication._tick = int(payload.get("tick", 0))


def save_runtime_checkpoint(
    runtime: Any,
    destination: str | Path | BinaryIO,
    *,
    communication: DelayedMultiChannelCommunication | None = None,
) -> None:
    torch.save(capture_runtime(runtime, communication=communication), destination)


def load_runtime_checkpoint(
    source: str | Path | BinaryIO,
    runtime: Any,
    *,
    communication: DelayedMultiChannelCommunication | None = None,
    strict_runtime_type: bool = False,
) -> None:
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("runtime checkpoint must contain a mapping")
    restore_runtime(
        runtime,
        payload,
        communication=communication,
        strict_runtime_type=strict_runtime_type,
    )


__all__ = [
    "FORMAT",
    "VERSION",
    "capture_runtime",
    "load_runtime_checkpoint",
    "restore_runtime",
    "save_runtime_checkpoint",
]
