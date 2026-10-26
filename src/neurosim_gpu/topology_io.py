"""Portable serialization and read-only views for sparse neural topologies.

The runtime keeps a source-major canonical edge list and a destination-major
CSR view.  This module serializes those linear-size arrays on the CPU and
reconstructs the :class:`~neurosim_gpu.topology.SparseTopology` directly.  It
never calls ``random_topology``, builds a dense matrix, or allocates an
``N``-by-``N`` intermediate.

The on-disk schema stores one canonical (source-major) weight vector.  The
destination weight vector is deterministic from ``out_weight_index`` and is
therefore reconstructed on load.  ``include_destination_weights=True`` is
available for interchange with older producers that want to store that
redundant validation copy as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import torch

from .topology import SparseTopology


FORMAT = "neurosim_gpu.sparse_topology"
SCHEMA_VERSION = 1

_INDEX_FIELDS = (
    "src",
    "dst",
    "out_rowptr",
    "in_rowptr",
    "in_col",
    "out_weight_index",
)
_REQUIRED_TENSOR_FIELDS = _INDEX_FIELDS + ("weight",)


def _as_python_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a Python integer")
    return int(value)


def _validate_neuron_count(value: Any) -> int:
    n_neurons = _as_python_int(value, "n_neurons")
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")
    # Every index is intentionally int32 so native kernels can use one compact
    # representation.  The edge count has the same bound.
    if n_neurons >= 2**31:
        raise ValueError("n_neurons must be less than 2^31")
    return n_neurons


def _validate_tensor_fields(payload: Mapping[str, Any]) -> tuple[int, torch.dtype, torch.device]:
    for name in _REQUIRED_TENSOR_FIELDS:
        if name not in payload:
            raise ValueError(f"topology payload is missing {name!r}")
        if not isinstance(payload[name], torch.Tensor):
            raise ValueError(f"topology payload field {name!r} must be a tensor")

    tensors = {name: payload[name] for name in _REQUIRED_TENSOR_FIELDS}
    for name, tensor in tensors.items():
        if tensor.ndim != 1:
            raise ValueError(f"topology payload field {name!r} must be rank-1")

    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise ValueError("topology payload tensors must share one device")

    for name in _INDEX_FIELDS:
        if tensors[name].dtype != torch.int32:
            raise ValueError(f"topology payload field {name!r} must use torch.int32")

    weight = tensors["weight"]
    if not weight.is_floating_point():
        raise ValueError("topology weights must use a floating-point dtype")
    if weight.numel() and not bool(torch.all(torch.isfinite(weight))):
        raise ValueError("topology weights must be finite")

    n_edges = weight.numel()
    if n_edges >= 2**31:
        raise ValueError("this compact topology supports fewer than 2^31 edges")
    for name in ("src", "dst"):
        if tensors[name].numel() != n_edges:
            raise ValueError(f"{name} length does not match weight length")
    if tensors["out_weight_index"].numel() != n_edges:
        raise ValueError("out_weight_index length does not match weight length")
    return n_edges, weight.dtype, next(iter(devices))


def _validate_rowptr(
    name: str,
    rowptr: torch.Tensor,
    n_rows: int,
    n_edges: int,
) -> torch.Tensor:
    if rowptr.shape != (n_rows + 1,):
        raise ValueError(f"{name} must have shape [{n_rows + 1}]")
    values = rowptr.to(torch.int64)
    if int(values[0]) != 0 or int(values[-1]) != n_edges:
        raise ValueError(f"{name} must start at 0 and end at n_edges")
    if values.numel() > 1 and bool(torch.any(values[1:] < values[:-1])):
        raise ValueError(f"{name} must be monotonically non-decreasing")
    return values


def _expected_rowptr(rows: torch.Tensor, n_rows: int) -> torch.Tensor:
    counts = torch.bincount(rows.to(torch.int64), minlength=n_rows)
    expected = torch.empty(n_rows + 1, dtype=torch.int64, device=rows.device)
    expected[0] = 0
    torch.cumsum(counts, dim=0, out=expected[1:])
    return expected


def _validate_payload(payload: Mapping[str, Any]) -> tuple[int, int, torch.dtype, torch.device]:
    if not isinstance(payload, Mapping):
        raise ValueError("topology payload must be a mapping")
    if payload.get("format") != FORMAT:
        raise ValueError("unsupported topology payload format")

    version_values = []
    for key in ("version", "schema_version"):
        if key in payload:
            version_values.append(_as_python_int(payload[key], key))
    if not version_values or any(value != SCHEMA_VERSION for value in version_values):
        raise ValueError(f"unsupported topology schema version (expected {SCHEMA_VERSION})")
    if len(set(version_values)) != 1:
        raise ValueError("topology schema version fields disagree")

    n_neurons = _validate_neuron_count(payload.get("n_neurons"))
    saved_weight_version = _as_python_int(
        payload.get("weight_version", 0),
        "weight_version",
    )
    if saved_weight_version < 0:
        raise ValueError("weight_version must be non-negative")

    n_edges, weight_dtype, device = _validate_tensor_fields(payload)
    src = payload["src"]
    dst = payload["dst"]
    out_rowptr = payload["out_rowptr"]
    in_rowptr = payload["in_rowptr"]
    in_col = payload["in_col"]
    out_weight_index = payload["out_weight_index"]

    if out_rowptr.device != device or in_rowptr.device != device:
        raise ValueError("row pointers must share the topology tensor device")
    out_values = _validate_rowptr("out_rowptr", out_rowptr, n_neurons, n_edges)
    in_values = _validate_rowptr("in_rowptr", in_rowptr, n_neurons, n_edges)

    src64 = src.to(torch.int64)
    dst64 = dst.to(torch.int64)
    if n_edges:
        if bool(torch.any(src64 < 0)) or bool(torch.any(dst64 < 0)):
            raise ValueError("edge indices must be non-negative")
        if bool(torch.any(src64 >= n_neurons)) or bool(torch.any(dst64 >= n_neurons)):
            raise ValueError("edge index out of range")

        # Canonical source-major order makes edge IDs deterministic.  Checking
        # adjacent keys detects duplicates without allocating an N-by-N mask.
        keys = src64 * n_neurons + dst64
        if n_edges > 1:
            delta = keys[1:] - keys[:-1]
            if bool(torch.any(delta < 0)):
                raise ValueError("source/destination edges are not source-major ordered")
            if bool(torch.any(delta == 0)):
                raise ValueError("duplicate source/destination edge")

    expected_out = _expected_rowptr(src, n_neurons)
    expected_in = _expected_rowptr(dst, n_neurons)
    if not torch.equal(out_values, expected_out):
        raise ValueError("out_rowptr does not match source-major edges")
    if not torch.equal(in_values, expected_in):
        raise ValueError("in_rowptr does not match destination counts")

    if in_col.numel() != n_edges:
        raise ValueError("in_col length does not match weight length")
    if n_edges:
        index64 = out_weight_index.to(torch.int64)
        if bool(torch.any(index64 < 0)) or bool(torch.any(index64 >= n_edges)):
            raise ValueError("out_weight_index is out of range")
        seen_index = torch.zeros(n_edges, dtype=torch.bool, device=device)
        seen_index[index64] = True
        if not bool(torch.all(seen_index)):
            raise ValueError("out_weight_index must be a permutation")

        # The permutation maps each source-major edge to its destination-major
        # position. Validate through its inverse, which is the destination
        # position -> source-major edge mapping.
        row_counts = in_values[1:] - in_values[:-1]
        destination_rows = torch.repeat_interleave(
            torch.arange(n_neurons, dtype=torch.int64, device=device),
            row_counts,
        )
        inverse_index = torch.empty_like(index64)
        inverse_index[index64] = torch.arange(
            n_edges, dtype=torch.int64, device=device
        )
        mapped_dst = dst64[inverse_index]
        mapped_src = src64[inverse_index]
        if not torch.equal(mapped_dst, destination_rows):
            raise ValueError("out_weight_index does not match in_rowptr")
        if not torch.equal(in_col.to(torch.int64), mapped_src):
            raise ValueError("in_col does not match out_weight_index")
        destination_keys = destination_rows * n_neurons + in_col.to(torch.int64)
        if destination_keys.numel() > 1 and bool(
            torch.any(destination_keys[1:] < destination_keys[:-1])
        ):
            raise ValueError("destination CSR view is not destination-major ordered")

    if "in_weight" in payload:
        in_weight = payload["in_weight"]
        if not isinstance(in_weight, torch.Tensor) or in_weight.ndim != 1:
            raise ValueError("in_weight must be a rank-1 tensor")
        if in_weight.dtype != weight_dtype or in_weight.device != device:
            raise ValueError("in_weight must match weight dtype and device")
        if in_weight.numel() != n_edges:
            raise ValueError("in_weight length does not match weight length")
        expected_in_weight = payload["weight"][inverse_index]
        if not torch.equal(in_weight, expected_in_weight):
            raise ValueError("in_weight does not match weight and out_weight_index")
        if in_weight.numel() and not bool(torch.all(torch.isfinite(in_weight))):
            raise ValueError("in_weight must be finite")

    return n_neurons, saved_weight_version, weight_dtype, device


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous()


def topology_to_dict(
    topology: SparseTopology,
    *,
    include_destination_weights: bool = False,
) -> dict[str, Any]:
    """Return a validated, CPU-only serialization payload.

    The payload contains one source-major weight vector.  Destination weights
    are a deterministic view and are omitted by default to keep checkpoints
    compact; they can be included as a redundant consistency check when
    exchanging files with another implementation.
    """

    _validate_topology_instance(topology)
    payload: dict[str, Any] = {
        "format": FORMAT,
        "version": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "n_neurons": int(topology.n_neurons),
        # This is provenance only.  Loading resets the runtime counter by
        # default because no persistent-current cache is serialized.
        "weight_version": int(topology.weight_version),
        "weight_version_reset_on_load": True,
        "src": _cpu_tensor(topology.src),
        "dst": _cpu_tensor(topology.dst),
        "weight": _cpu_tensor(topology.weight),
        "out_rowptr": _cpu_tensor(topology.out_rowptr),
        "in_rowptr": _cpu_tensor(topology.in_rowptr),
        "in_col": _cpu_tensor(topology.in_col),
        "out_weight_index": _cpu_tensor(topology.out_weight_index),
    }
    if include_destination_weights:
        payload["in_weight"] = _cpu_tensor(topology.in_weight)
    # Validate the CPU payload as a final guard against accidental schema drift.
    _validate_payload(payload)
    return payload


def topology_from_dict(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
    reset_weight_version: bool = True,
    single_weight: bool = False,
) -> SparseTopology | "SingleWeightInferenceView":
    """Build a topology directly from a serialized payload.

    ``device`` defaults to CPU, making the format portable across hosts.  No
    sorting, random generation, or dense matrix construction is performed.
    ``reset_weight_version`` defaults to true because persistent-current and
    native caches are intentionally not part of this file format.
    """

    n_neurons, saved_version, _, source_device = _validate_payload(payload)
    target = torch.device("cpu" if device is None else device)

    def move(name: str) -> torch.Tensor:
        return payload[name].detach().to(device=target).contiguous()

    src = move("src")
    dst = move("dst")
    weight = move("weight")
    out_rowptr = move("out_rowptr")
    in_rowptr = move("in_rowptr")
    in_col = move("in_col")
    out_weight_index = move("out_weight_index")
    index64 = out_weight_index.to(torch.int64)
    inverse_index = torch.empty_like(index64)
    inverse_index[index64] = torch.arange(
        index64.numel(), dtype=torch.int64, device=target
    )
    in_weight = weight[inverse_index]

    version = 0 if reset_weight_version else saved_version
    if single_weight:
        return SingleWeightInferenceView(
            n_neurons=n_neurons,
            src=src,
            dst=dst,
            weight=in_weight,
            out_rowptr=out_rowptr,
            in_rowptr=in_rowptr,
            in_col=in_col,
            out_weight_index=out_weight_index,
            weight_version=version,
        )
    # ``source_device`` is intentionally read above as part of validation.  A
    # payload may be supplied from a GPU, but all fields are moved together.
    del source_device
    return SparseTopology(
        n_neurons=n_neurons,
        src=src,
        dst=dst,
        weight=weight,
        out_rowptr=out_rowptr,
        in_rowptr=in_rowptr,
        in_col=in_col,
        out_weight_index=out_weight_index,
        in_weight=in_weight,
        weight_version=version,
    )


def save_topology(
    topology: SparseTopology,
    destination: str | Path | BinaryIO,
    *,
    include_destination_weights: bool = False,
) -> None:
    """Save a topology as a CPU/GPU-neutral ``torch.save`` dictionary."""

    payload = topology_to_dict(
        topology,
        include_destination_weights=include_destination_weights,
    )
    torch.save(payload, destination)


def _load_torch_payload(source: str | Path | BinaryIO) -> Mapping[str, Any]:
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # Torch versions before the weights_only argument.
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("serialized topology must contain a mapping")
    return payload


def load_topology(
    source: str | Path | BinaryIO,
    *,
    device: torch.device | str | None = None,
    reset_weight_version: bool = True,
    single_weight: bool = False,
) -> SparseTopology | "SingleWeightInferenceView":
    """Load a topology and optionally move all arrays to ``device``."""

    return topology_from_dict(
        _load_torch_payload(source),
        device=device,
        reset_weight_version=reset_weight_version,
        single_weight=single_weight,
    )


def _validate_topology_instance(topology: SparseTopology) -> None:
    if not isinstance(topology, SparseTopology):
        raise ValueError("topology must be a SparseTopology")
    payload = {
        "format": FORMAT,
        "version": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "n_neurons": int(topology.n_neurons),
        "weight_version": int(topology.weight_version),
        "src": topology.src,
        "dst": topology.dst,
        "weight": topology.weight,
        "out_rowptr": topology.out_rowptr,
        "in_rowptr": topology.in_rowptr,
        "in_col": topology.in_col,
        "out_weight_index": topology.out_weight_index,
        "in_weight": topology.in_weight,
    }
    _validate_payload(payload)


@dataclass(frozen=True)
class TopologyMemoryAccounting:
    """Byte-level accounting for the topology's two graph layouts."""

    source_index_bytes: int
    destination_index_bytes: int
    source_weight_bytes: int
    destination_weight_bytes: int

    @property
    def canonical_bytes(self) -> int:
        """Bytes in the full ``SparseTopology`` representation."""

        return (
            self.source_index_bytes
            + self.destination_index_bytes
            + self.source_weight_bytes
            + self.destination_weight_bytes
        )

    @property
    def compact_serialized_bytes(self) -> int:
        """Tensor bytes in the compact single-weight serialization payload."""

        return self.source_index_bytes + self.destination_index_bytes + self.source_weight_bytes

    @property
    def single_weight_inference_bytes(self) -> int:
        """Bytes for native destination-major inference plus event mapping."""

        return self.destination_index_bytes + self.destination_weight_bytes + self.source_index_bytes - (
            self.source_weight_bytes
        )

    @property
    def total_bytes(self) -> int:
        return self.canonical_bytes

    def as_dict(self) -> dict[str, int]:
        return {
            "source_index_bytes": self.source_index_bytes,
            "destination_index_bytes": self.destination_index_bytes,
            "source_weight_bytes": self.source_weight_bytes,
            "destination_weight_bytes": self.destination_weight_bytes,
            "canonical_bytes": self.canonical_bytes,
            "compact_serialized_bytes": self.compact_serialized_bytes,
            "single_weight_inference_bytes": self.single_weight_inference_bytes,
            "total_bytes": self.total_bytes,
        }

    def __getitem__(self, key: str) -> int:
        return self.as_dict()[key]


def memory_accounting(topology: SparseTopology) -> TopologyMemoryAccounting:
    """Return storage accounting without constructing a dense matrix."""

    _validate_topology_instance(topology)
    source_index_bytes = sum(
        topology.__dict__[name].numel() * topology.__dict__[name].element_size()
        for name in ("src", "dst", "out_rowptr")
    )
    destination_index_bytes = sum(
        topology.__dict__[name].numel() * topology.__dict__[name].element_size()
        for name in ("in_rowptr", "in_col", "out_weight_index")
    )
    return TopologyMemoryAccounting(
        source_index_bytes=source_index_bytes,
        destination_index_bytes=destination_index_bytes,
        source_weight_bytes=topology.weight.numel() * topology.weight.element_size(),
        destination_weight_bytes=topology.in_weight.numel()
        * topology.in_weight.element_size(),
    )


def topology_memory_accounting(topology: SparseTopology) -> TopologyMemoryAccounting:
    """Explicit alias for :func:`memory_accounting`."""

    return memory_accounting(topology)


@dataclass(frozen=True)
class SingleWeightInferenceView:
    """Read-only API view backed by one destination-major weight tensor.

    ``weight`` is the destination-major CSR value array used by SpMM.  The
    source-major event path addresses it through ``out_weight_index``.  The
    dataclass intentionally exposes no mutating weight/update methods.  As
    with all PyTorch tensors, callers that need hard storage protection should
    keep the owning topology private and pass this view to inference code.
    """

    n_neurons: int
    src: torch.Tensor
    dst: torch.Tensor
    weight: torch.Tensor
    out_rowptr: torch.Tensor
    in_rowptr: torch.Tensor
    in_col: torch.Tensor
    out_weight_index: torch.Tensor
    weight_version: int = 0

    @property
    def n_edges(self) -> int:
        return self.weight.numel()

    @property
    def out_degree(self) -> torch.Tensor:
        return self.out_rowptr[1:] - self.out_rowptr[:-1]

    @property
    def in_weight(self) -> torch.Tensor:
        """Compatibility name for the destination-major value array."""

        return self.weight

    @property
    def source_weight(self) -> torch.Tensor:
        """Return source-major values indexed through the destination array."""

        return self.weight[self.out_weight_index.to(torch.int64)]

    def sparse_matrix(self) -> torch.Tensor:
        return torch.sparse_csr_tensor(
            self.in_rowptr,
            self.in_col,
            self.weight,
            size=(self.n_neurons, self.n_neurons),
            dtype=self.weight.dtype,
            device=self.weight.device,
            check_invariants=False,
        )

    def storage_bytes(self) -> int:
        tensors = (
            self.dst,
            self.out_weight_index,
            self.out_rowptr,
            self.in_rowptr,
            self.in_col,
            self.weight,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def to(self, device: torch.device | str) -> "SingleWeightInferenceView":
        target = torch.device(device)
        return SingleWeightInferenceView(
            n_neurons=self.n_neurons,
            src=self.src.to(target),
            dst=self.dst.to(target),
            weight=self.weight.to(target),
            out_rowptr=self.out_rowptr.to(target),
            in_rowptr=self.in_rowptr.to(target),
            in_col=self.in_col.to(target),
            out_weight_index=self.out_weight_index.to(target),
            weight_version=self.weight_version,
        )


ReadOnlySingleWeightView = SingleWeightInferenceView


def single_weight_inference_view(
    topology: SparseTopology,
) -> SingleWeightInferenceView:
    """Create a destination-major, single-weight inference view."""

    _validate_topology_instance(topology)
    return SingleWeightInferenceView(
        n_neurons=topology.n_neurons,
        src=topology.src.detach(),
        dst=topology.dst.detach(),
        weight=topology.in_weight.detach(),
        out_rowptr=topology.out_rowptr.detach(),
        in_rowptr=topology.in_rowptr.detach(),
        in_col=topology.in_col.detach(),
        out_weight_index=topology.out_weight_index.detach(),
        weight_version=topology.weight_version,
    )


# Friendly aliases for callers that use "serialize" terminology.
serialize_topology = save_topology
deserialize_topology = load_topology
topology_to_payload = topology_to_dict
topology_from_payload = topology_from_dict


__all__ = [
    "FORMAT",
    "SCHEMA_VERSION",
    "ReadOnlySingleWeightView",
    "SingleWeightInferenceView",
    "TopologyMemoryAccounting",
    "deserialize_topology",
    "load_topology",
    "memory_accounting",
    "serialize_topology",
    "save_topology",
    "single_weight_inference_view",
    "topology_from_dict",
    "topology_from_payload",
    "topology_memory_accounting",
    "topology_to_dict",
    "topology_to_payload",
]
