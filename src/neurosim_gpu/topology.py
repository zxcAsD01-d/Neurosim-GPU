from __future__ import annotations

from dataclasses import dataclass

import torch


def _row_pointer(rows: torch.Tensor, n_rows: int) -> torch.Tensor:
    counts = torch.bincount(rows, minlength=n_rows)
    rowptr = torch.empty(n_rows + 1, dtype=torch.int64, device=rows.device)
    rowptr[0] = 0
    torch.cumsum(counts, dim=0, out=rowptr[1:])
    return rowptr


@dataclass(frozen=True)
class SparseTopology:
    """A fixed graph with source-major canonical edges and a destination CSR view."""

    n_neurons: int
    src: torch.Tensor
    dst: torch.Tensor
    weight: torch.Tensor
    out_rowptr: torch.Tensor
    in_rowptr: torch.Tensor
    in_col: torch.Tensor
    out_weight_index: torch.Tensor
    in_weight: torch.Tensor
    weight_version: int = 0

    @classmethod
    def from_edges(
        cls,
        n_neurons: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        weight: torch.Tensor,
    ) -> "SparseTopology":
        if src.ndim != 1 or dst.ndim != 1 or weight.ndim != 1:
            raise ValueError("src, dst and weight must be rank-1 tensors")
        if not (src.numel() == dst.numel() == weight.numel()):
            raise ValueError("src, dst and weight must have equal length")
        if src.dtype not in (torch.int32, torch.int64) or dst.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("edge indices must use torch.int32 or torch.int64")
        if src.device != dst.device or src.device != weight.device:
            raise ValueError("all edge tensors must be on the same device")
        src64 = src.to(torch.int64)
        dst64 = dst.to(torch.int64)
        if src.numel() and (
            int(src64.min()) < 0
            or int(dst64.min()) < 0
            or int(src64.max()) >= n_neurons
            or int(dst64.max()) >= n_neurons
        ):
            raise ValueError("edge index out of range")
        if src.numel() >= 2**31:
            raise ValueError("this compact topology supports fewer than 2^31 edges")

        # Source-major edge IDs are canonical. Event propagation can therefore
        # expand a source interval without touching inactive edges.
        out_key = src64 * n_neurons + dst64
        out_order = torch.argsort(out_key, stable=True)
        sorted_key = out_key[out_order]
        sorted_weight = weight[out_order]

        # Parallel edges have identical dynamics to one edge with their summed
        # weight. Coalescing makes canonical edge IDs unique and avoids an
        # ambiguous/unsupported CSR gradient for repeated column indices.
        unique_key, inverse = torch.unique_consecutive(
            sorted_key,
            return_inverse=True,
        )
        canonical_weight = torch.zeros(
            unique_key.numel(),
            dtype=weight.dtype,
            device=weight.device,
        ).index_add(0, inverse, sorted_weight)
        canonical_src64 = torch.div(unique_key, n_neurons, rounding_mode="floor")
        canonical_dst64 = torch.remainder(unique_key, n_neurons)
        out_rowptr = _row_pointer(canonical_src64, n_neurons).to(torch.int32)

        # cuSPARSE consumes a destination-major CSR matrix W[dst, src].
        in_order = torch.argsort(
            canonical_dst64 * n_neurons + canonical_src64,
            stable=True,
        )
        in_dst = canonical_dst64[in_order]
        in_rowptr = _row_pointer(in_dst, n_neurons).to(torch.int32)
        in_col = canonical_src64[in_order].to(torch.int32)
        in_weight = canonical_weight[in_order]
        out_weight_index = torch.empty(
            canonical_weight.numel(),
            dtype=torch.int32,
            device=weight.device,
        )
        out_weight_index[in_order] = torch.arange(
            canonical_weight.numel(),
            dtype=torch.int32,
            device=weight.device,
        )

        return cls(
            n_neurons=n_neurons,
            src=canonical_src64.to(torch.int32),
            dst=canonical_dst64.to(torch.int32),
            weight=canonical_weight,
            out_rowptr=out_rowptr,
            in_rowptr=in_rowptr,
            in_col=in_col,
            out_weight_index=out_weight_index,
            in_weight=in_weight,
        )

    @property
    def n_edges(self) -> int:
        return self.weight.numel()

    @property
    def out_degree(self) -> torch.Tensor:
        return self.out_rowptr[1:] - self.out_rowptr[:-1]

    def sparse_matrix(self) -> torch.Tensor:
        return torch.sparse_csr_tensor(
            self.in_rowptr,
            self.in_col,
            self.in_weight,
            size=(self.n_neurons, self.n_neurons),
            dtype=self.weight.dtype,
            device=self.weight.device,
            check_invariants=False,
        )

    def dense_matrix(self) -> torch.Tensor:
        matrix = torch.zeros(
            (self.n_neurons, self.n_neurons),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        matrix.index_put_(
            (self.dst.to(torch.int64), self.src.to(torch.int64)),
            self.weight,
            accumulate=True,
        )
        return matrix

    def storage_bytes(self) -> int:
        tensors = (
            self.src,
            self.dst,
            self.weight,
            self.out_rowptr,
            self.in_rowptr,
            self.in_col,
            self.out_weight_index,
            self.in_weight,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def torch_hybrid_storage_bytes(self) -> int:
        """Logical storage for the PyTorch matrix plus event representation."""

        tensors = (
            self.src,
            self.dst,
            self.weight,
            self.out_rowptr,
            self.in_rowptr,
            self.in_col,
            self.in_weight,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def native_storage_bytes(self) -> int:
        """Storage needed by native matrix and event kernels.

        The single deployed weight array is destination-major for fast SpMM.
        Source-major event traversal uses out_weight_index to address it.
        """

        tensors = (
            self.dst,
            self.out_weight_index,
            self.out_rowptr,
            self.in_rowptr,
            self.in_col,
            self.in_weight,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @torch.no_grad()
    def apply_source_weights_(self, source_major_weight: torch.Tensor) -> None:
        """Update both graph views from one source-major weight vector."""

        if source_major_weight.shape != self.weight.shape:
            raise ValueError("source_major_weight shape does not match topology")
        if source_major_weight.device != self.weight.device:
            raise ValueError("source_major_weight must be on the topology device")
        if source_major_weight.dtype != self.weight.dtype:
            raise ValueError("source_major_weight dtype does not match topology")
        self.weight.copy_(source_major_weight)
        self.in_weight[self.out_weight_index.to(torch.int64)] = source_major_weight
        self.mark_weights_mutated_()

    def mark_weights_mutated_(self) -> None:
        """Invalidate persistent-current caches after an in-place native update."""

        object.__setattr__(self, "weight_version", self.weight_version + 1)


def random_topology(
    n_neurons: int,
    n_edges: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    weight_scale: float = 0.15,
) -> SparseTopology:
    if n_neurons <= 0 or n_edges < 0:
        raise ValueError("n_neurons must be positive and n_edges non-negative")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    src = torch.randint(
        n_neurons,
        (n_edges,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    dst = torch.randint(
        n_neurons,
        (n_edges,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    mean_degree = max(n_edges / n_neurons, 1.0)
    weight = torch.randn(
        n_edges,
        dtype=dtype,
        device=device,
        generator=generator,
    ) * (weight_scale / mean_degree**0.5)
    return SparseTopology.from_edges(n_neurons, src, dst, weight)
