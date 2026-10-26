from __future__ import annotations

import torch
from torch import nn


class DelayedMultiChannelCommunication(nn.Module):
    """Vectorized fixed-graph communication with discrete conduction delays.

    For an input state ``s_t`` with shape ``[batch, neuron]``, edge ``e`` adds

    ``weight[e] * receptor_gain[dst[e], transmitter[e]]
        * s_(t-delay[e])[src[e]]``

    to its destination. Missing pre-history is zero. Transmitter IDs select a
    receptor channel only; excitatory/inhibitory sign is carried independently
    by ``weight``. Edges are stored source-major after construction.

    The delayed path uses a ``max_delay + 1`` state ring and vectorized gather
    and scatter-add operations. Its tick cost is O(B*M + B*N) and storage is
    O(B*N*(D + 1) + M). An all-zero-delay graph bypasses the history ring.

    Args:
        n_neurons: Number of source and destination neurons.
        src: Source neuron ID for each edge.
        dst: Destination neuron ID for each edge.
        weight: Signed scalar strength for each edge.
        delay: Non-negative integer conduction delay in ticks for each edge.
        transmitter: Integer channel ID for each edge.
        receptor_gain: Either global gains ``[channels]`` or destination-local
            gains ``[neurons, channels]``. Gains are fixed after construction.
    """

    def __init__(
        self,
        n_neurons: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        weight: torch.Tensor,
        delay: torch.Tensor,
        transmitter: torch.Tensor,
        receptor_gain: torch.Tensor,
    ) -> None:
        super().__init__()
        self._validate_graph(
            n_neurons,
            src,
            dst,
            weight,
            delay,
            transmitter,
            receptor_gain,
        )

        self.n_neurons = n_neurons
        self.n_channels = int(receptor_gain.shape[-1])
        self._global_receptor_gain = receptor_gain.ndim == 1

        src64 = src.to(torch.int64)
        dst64 = dst.to(torch.int64)
        delay64 = delay.to(torch.int64)
        transmitter64 = transmitter.to(torch.int64)
        order = torch.argsort(src64, stable=True)

        sorted_src = src64[order]
        sorted_dst = dst64[order]
        sorted_weight = weight[order]
        sorted_delay = delay64[order]
        sorted_transmitter = transmitter64[order]
        fixed_gain = receptor_gain.clone()
        if self._global_receptor_gain:
            edge_gain = fixed_gain[sorted_transmitter]
        else:
            edge_gain = fixed_gain[sorted_dst, sorted_transmitter]

        self.register_buffer("src", sorted_src)
        self.register_buffer("dst", sorted_dst)
        self.register_buffer("weight", sorted_weight)
        self.register_buffer("delay", sorted_delay)
        self.register_buffer("transmitter", sorted_transmitter)
        self.register_buffer("receptor_gain", fixed_gain)
        self.register_buffer("edge_receptor_gain", edge_gain)

        self.max_delay = int(sorted_delay.max().item()) if delay.numel() else 0
        self.history_length = self.max_delay + 1
        self._all_zero_delay = self.max_delay == 0
        self.register_buffer(
            "history",
            weight.new_empty((0,)),
            persistent=False,
        )
        self._cursor = 0
        self._tick = 0

    @staticmethod
    def _validate_graph(
        n_neurons: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        weight: torch.Tensor,
        delay: torch.Tensor,
        transmitter: torch.Tensor,
        receptor_gain: torch.Tensor,
    ) -> None:
        if not isinstance(n_neurons, int) or isinstance(n_neurons, bool):
            raise ValueError("n_neurons must be an integer")
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")

        edge_tensors = (src, dst, weight, delay, transmitter)
        if any(tensor.ndim != 1 for tensor in edge_tensors):
            raise ValueError("all edge tensors must be rank-1")
        lengths = {tensor.numel() for tensor in edge_tensors}
        if len(lengths) != 1:
            raise ValueError("all edge tensors must have equal length")

        integer_types = (torch.int32, torch.int64)
        if src.dtype not in integer_types or dst.dtype not in integer_types:
            raise ValueError("src and dst must use torch.int32 or torch.int64")
        if delay.dtype not in integer_types:
            raise ValueError("delay must use torch.int32 or torch.int64")
        if transmitter.dtype not in integer_types:
            raise ValueError(
                "transmitter must use torch.int32 or torch.int64"
            )
        if not torch.is_floating_point(weight):
            raise ValueError("weight must be floating point")
        if receptor_gain.ndim not in (1, 2):
            raise ValueError("receptor_gain must have shape [C] or [N, C]")
        if receptor_gain.ndim == 2 and receptor_gain.shape[0] != n_neurons:
            raise ValueError(
                "destination-local receptor_gain must have n_neurons rows"
            )
        if receptor_gain.shape[-1] <= 0:
            raise ValueError("receptor_gain must define at least one channel")
        if not torch.is_floating_point(receptor_gain):
            raise ValueError("receptor_gain must be floating point")
        if receptor_gain.dtype != weight.dtype:
            raise ValueError("receptor_gain and weight must have the same dtype")

        graph_device = weight.device
        if any(tensor.device != graph_device for tensor in edge_tensors):
            raise ValueError("all edge tensors must be on the same device")
        if receptor_gain.device != graph_device:
            raise ValueError("receptor_gain must be on the edge tensor device")
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("weight must contain only finite values")
        if not bool(torch.isfinite(receptor_gain).all()):
            raise ValueError("receptor_gain must contain only finite values")

        if src.numel() == 0:
            return
        src64 = src.to(torch.int64)
        dst64 = dst.to(torch.int64)
        delay64 = delay.to(torch.int64)
        transmitter64 = transmitter.to(torch.int64)
        if bool((src64 < 0).any()) or bool((src64 >= n_neurons).any()):
            raise ValueError("source neuron index out of range")
        if bool((dst64 < 0).any()) or bool((dst64 >= n_neurons).any()):
            raise ValueError("destination neuron index out of range")
        if bool((delay64 < 0).any()):
            raise ValueError("conduction delays must be non-negative")
        n_channels = receptor_gain.shape[-1]
        if bool((transmitter64 < 0).any()) or bool(
            (transmitter64 >= n_channels).any()
        ):
            raise ValueError("transmitter channel index out of range")

    @property
    def n_edges(self) -> int:
        return self.weight.numel()

    @property
    def tick(self) -> int:
        return self._tick

    def reset_history(self, batch_size: int | None = None) -> None:
        """Clear causal history, optionally preallocating it for a batch."""

        if batch_size is not None and (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if self._all_zero_delay or batch_size is None:
            self.history = self.weight.new_empty((0,))
        else:
            self.history = self.weight.new_zeros(
                (self.history_length, batch_size, self.n_neurons)
            )
        self._cursor = 0
        self._tick = 0

    def _validate_state(self, state: torch.Tensor) -> None:
        if state.ndim != 2 or state.shape[1] != self.n_neurons:
            raise ValueError("state must have shape [batch, n_neurons]")
        if state.shape[0] <= 0:
            raise ValueError("state batch must be non-empty")
        if state.device != self.weight.device:
            raise ValueError("state must be on the communication graph device")
        if state.dtype != self.weight.dtype:
            raise ValueError("state dtype must match weight dtype")
        if self.history.numel() and self.history.shape[1] != state.shape[0]:
            raise ValueError(
                "state batch size changed; call reset_history before reuse"
            )

    def _accumulate(self, edge_source: torch.Tensor) -> torch.Tensor:
        contribution = edge_source * (
            self.weight * self.edge_receptor_gain
        ).unsqueeze(0)
        current = contribution.new_zeros(
            (contribution.shape[0], self.n_neurons)
        )
        if self.n_edges:
            current.scatter_add_(
                1,
                self.dst.unsqueeze(0).expand(contribution.shape[0], -1),
                contribution,
            )
        return current

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Advance one communication tick and return destination current."""

        self._validate_state(state)
        if self._all_zero_delay:
            self._tick += 1
            return self._accumulate(state[:, self.src])

        if not self.history.numel():
            self.reset_history(state.shape[0])

        # Write s_t before gathering so delay zero means same-tick delivery.
        self.history[self._cursor].copy_(state)
        delayed_slot = torch.remainder(
            self._cursor - self.delay,
            self.history_length,
        )
        batch = torch.arange(state.shape[0], device=state.device).unsqueeze(1)
        edge_source = self.history[
            delayed_slot.unsqueeze(0),
            batch,
            self.src.unsqueeze(0),
        ]
        current = self._accumulate(edge_source)
        self._cursor = (self._cursor + 1) % self.history_length
        self._tick += 1
        return current

    step = forward

    def extra_repr(self) -> str:
        return (
            f"n_neurons={self.n_neurons}, n_edges={self.n_edges}, "
            f"n_channels={self.n_channels}, max_delay={self.max_delay}"
        )


__all__ = ["DelayedMultiChannelCommunication"]
