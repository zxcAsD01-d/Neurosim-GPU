"""The recurrent cortical sheet that sits between the text encoder and decoder.

For every token the core receives a drive vector (the encoder hidden state
projected to ``N`` neurons), advances the fixed sparse graph ``ticks_per_token``
times, and reads the population state back into the backbone hidden size. The
neuron state is *not* reset between tokens: this is a recurrent core, and the
only reset points are sequence/block boundaries and truncated-BPTT detaches.

The edge weights are frozen. PyTorch's CSR ``sparse.mm`` does not produce a
usable gradient for the CSR values on the reference path (the gradient comes
back with the coalesced edge count), so plasticity lives in the interfaces,
per-neuron gains and the decoder adapters. Section 4 of ``docs/cortex-lm.md``
records that trade-off and what would be needed to lift it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..dynamics import Dynamics
from ..runtime import MatrixRuntime
from ..topology import SparseTopology
from .config import BrainConfig
from .spatial import (
    PlateGeometry,
    loss_driven_rewire,
    migrate,
    rewire,
    sparsify_weights,
)
from .topology import build_cortical_topology


class _CachedRuntime(MatrixRuntime):
    """A ``MatrixRuntime`` that reuses the CSR matrix built once per forward."""

    def __init__(
        self,
        topology: SparseTopology,
        dynamics: Dynamics,
        initial_state: torch.Tensor,
        matrix: torch.Tensor,
    ) -> None:
        self.topology = topology
        self.dynamics = dynamics
        self.state = initial_state
        self.weight = matrix


class CorticalCore(nn.Module):
    """A neuron sheet with trainable interfaces and gains.

    ``weight_mode="sparse"`` uses the repository's fixed sparse CSR graph;
    ``weight_mode="dense"`` keeps an ``N x N`` trainable recurrent matrix so
    every synapse can be updated (used by the Qwen3 3+2 transplant).
    """

    def __init__(
        self,
        config: BrainConfig,
        d_model: int,
        init_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.config = config
        self.d_model = int(d_model)
        self.weight_mode = config.weight_mode
        self.trace: list[torch.Tensor] | None = None

        topology = build_cortical_topology(config, device="cpu")
        graph = topology.graph
        self.n_neurons = graph.n_neurons
        self.n_edges = (
            self.n_neurons * self.n_neurons
            if config.weight_mode == "dense"
            else graph.n_edges
        )

        self.register_buffer("region", topology.region, persistent=False)
        self.register_buffer("sign", topology.sign, persistent=False)
        self.register_buffer("alpha", topology.alpha, persistent=False)
        self.register_buffer("base_gain", topology.base_gain, persistent=False)

        if self.weight_mode == "sparse":
            # Graph tensors travel with the module; the CSR view is rebuilt per
            # forward (cheap: it reuses these tensors without copying them).
            self.register_buffer("src", graph.src, persistent=False)
            self.register_buffer("dst", graph.dst, persistent=False)
            self.register_buffer("weight", graph.weight, persistent=False)
            self.register_buffer("out_rowptr", graph.out_rowptr, persistent=False)
            self.register_buffer("in_rowptr", graph.in_rowptr, persistent=False)
            self.register_buffer("in_col", graph.in_col, persistent=False)
            self.register_buffer(
                "out_weight_index", graph.out_weight_index, persistent=False
            )
            self.register_buffer("in_weight", graph.in_weight, persistent=False)
        else:
            self.recurrent = nn.Parameter(
                torch.zeros(self.n_neurons, self.n_neurons)
            )
            if init_weight is not None:
                if tuple(init_weight.shape) != (
                    self.n_neurons,
                    self.n_neurons,
                ):
                    raise ValueError(
                        "init_weight must have shape "
                        f"{(self.n_neurons, self.n_neurons)}, "
                        f"got {tuple(init_weight.shape)}"
                    )
                with torch.no_grad():
                    self.recurrent.copy_(init_weight.float())
            else:
                generator = torch.Generator(device="cpu").manual_seed(
                    config.seed + 101
                )
                random_weight = torch.randn(
                    self.n_neurons,
                    self.n_neurons,
                    generator=generator,
                )
                from .transplant import spectral_radius

                radius = spectral_radius(random_weight)
                random_weight = random_weight * (
                    config.init_radius / max(radius, 1e-12)
                )
                with torch.no_grad():
                    self.recurrent.copy_(random_weight)

        self.spatial = bool(config.spatial)
        if self.spatial:
            self.geometry = PlateGeometry(
                n_neurons=self.n_neurons,
                mean_neuron_distance=1.0,
                plate_distance=50.0,
            )
            positions = self.geometry.sample_positions(seed=config.seed)
            self.register_buffer("positions", positions, persistent=False)
            mask = sparsify_weights(
                self.recurrent.detach(),
                positions,
                edges_per_neuron=config.spatial_edges_per_neuron,
                distance_decay=5.0,
            )
            self.register_buffer("edge_mask", mask, persistent=False)
            self.register_buffer(
                "last_activity", torch.zeros(self.n_neurons), persistent=False
            )
            with torch.no_grad():
                self.recurrent.mul_(mask)

        self.biological_memory = (
            self.weight_mode == "dense"
            and (
                config.eligibility_decay > 0
                or config.consolidation_rate > 0
                or config.short_term_tau > 0
            )
        )
        if self.biological_memory:
            self.register_buffer(
                "eligibility",
                torch.zeros(self.n_neurons, self.n_neurons),
                persistent=False,
            )
            if config.short_term_tau > 0:
                self.register_buffer(
                    "synaptic_resource",
                    torch.ones(self.n_neurons),
                    persistent=False,
                )
            else:
                self.register_buffer(
                    "synaptic_resource", None, persistent=False
                )
            with torch.no_grad():
                row_norm = self.recurrent.detach().norm(dim=1)
            self.register_buffer(
                "row_target", row_norm, persistent=False
            )
            self.register_buffer(
                "dopamine_baseline",
                torch.zeros((), dtype=torch.float32),
                persistent=False,
            )

        self.gain_delta: nn.Parameter | None
        if config.learnable_gain:
            self.gain_delta = nn.Parameter(torch.zeros(self.n_neurons))
        else:
            self.register_buffer("gain_delta", None, persistent=False)

        self.region_log_gain: nn.Parameter | None
        if config.learnable_region_gain:
            self.region_log_gain = nn.Parameter(
                torch.zeros(config.regions)
            )
        else:
            self.register_buffer("region_log_gain", None, persistent=False)

        self.init_state = nn.Parameter(torch.zeros(self.n_neurons))

        self.drive_norm = nn.LayerNorm(d_model)
        self.drive_proj = nn.Linear(d_model, self.n_neurons, bias=False)
        self.drive_bias = nn.Parameter(torch.zeros(self.n_neurons))
        self.drive_scale = nn.Parameter(torch.tensor(float(config.drive_scale)))

        self.readout_norm = nn.LayerNorm(self.n_neurons)
        self.readout_proj = nn.Linear(self.n_neurons, d_model)

        for module in (self.drive_proj, self.readout_proj):
            nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(self.readout_proj.bias)
        if (
            self.weight_mode == "dense"
            and self.n_neurons == d_model
        ):
            # One neuron per residual channel: start with an identity interface
            # so the transplanted recurrent matrix sees the residual stream.
            with torch.no_grad():
                self.drive_proj.weight.copy_(torch.eye(d_model))
                self.readout_proj.weight.copy_(torch.eye(d_model))

    def refresh_spatial_mask(self) -> None:
        """Recompute the sparse mask after loading a dense checkpoint."""

        if not self.spatial:
            return
        mask = sparsify_weights(
            self.recurrent.detach(),
            self.positions,
            edges_per_neuron=self.config.spatial_edges_per_neuron,
            distance_decay=5.0,
        )
        self.edge_mask.copy_(mask)
        with torch.no_grad():
            self.recurrent.mul_(mask)

    def spatial_adapt(self, step: int) -> dict[str, float]:
        """Apply one slow migration/rewiring epoch when due."""

        if not self.spatial:
            return {}
        stats: dict[str, float] = {}
        if (
            self.config.migration_interval
            and step % self.config.migration_interval == 0
        ):
            moved, displacement = migrate(
                self.positions,
                self.last_activity,
                self.geometry,
                maximum_speed=self.config.migration_speed,
            )
            self.positions.copy_(moved)
            stats["migration_mean"] = float(displacement.mean())
            stats["migration_fraction"] = float(
                (displacement > 0.05).float().mean()
            )
        if (
            self.config.rewire_interval
            and step % self.config.rewire_interval == 0
        ):
            gradient = self.recurrent.grad
            if (
                self.config.rewire_loss_driven
                and gradient is not None
            ):
                weights, mask, rewire_stats = loss_driven_rewire(
                    self.recurrent.detach(),
                    gradient.detach(),
                    self.edge_mask,
                    self.positions,
                    fraction=self.config.rewire_fraction,
                )
            else:
                weights, mask, rewire_stats = rewire(
                    self.recurrent.detach(),
                    self.edge_mask,
                    self.positions,
                    self.last_activity,
                    fraction=self.config.rewire_fraction,
                )
            self.recurrent.data.copy_(weights)
            self.edge_mask.copy_(mask)
            stats.update(rewire_stats)
        return stats

    @torch.no_grad()
    def consolidate(self, dopamine: float) -> dict[str, float]:
        """Apply eligibility traces to weights, gated by dopamine."""

        if (
            not self.biological_memory
            or self.config.consolidation_rate <= 0
        ):
            return {}
        update = (
            self.config.consolidation_rate
            * float(dopamine)
            * self.eligibility
        )
        if self.spatial:
            update = update * self.edge_mask
        self.recurrent.data.add_(update)
        if self.config.homeostatic_strength > 0:
            row_norm = self.recurrent.detach().norm(dim=1)
            scale = self.row_target / row_norm.clamp_min(1e-8)
            blend = (
                1.0
                - self.config.homeostatic_strength
                + self.config.homeostatic_strength * scale
            )
            self.recurrent.data.mul_(blend.unsqueeze(1))
        self.eligibility.mul_(0.5)
        return {
            "consolidated": float(update.abs().mean()),
            "dopamine": float(dopamine),
            "eligibility_norm": float(self.eligibility.norm()),
        }

    def _graph(self) -> SparseTopology:
        if self.weight_mode != "sparse":
            raise RuntimeError("_graph() is only valid in sparse mode")
        return SparseTopology(
            n_neurons=self.n_neurons,
            src=self.src,
            dst=self.dst,
            weight=self.weight,
            out_rowptr=self.out_rowptr,
            in_rowptr=self.in_rowptr,
            in_col=self.in_col,
            out_weight_index=self.out_weight_index,
            in_weight=self.in_weight,
        )

    def effective_gain(self) -> torch.Tensor:
        gain = self.base_gain
        if self.gain_delta is not None:
            gain = gain * torch.exp(self.gain_delta.clamp(-4.0, 4.0))
        if self.region_log_gain is not None:
            region_gain = torch.exp(self.region_log_gain.clamp(-2.0, 2.0))
            gain = gain * region_gain[self.region]
        return gain

    def initial_state(self, batch: int) -> torch.Tensor:
        return self.init_state.unsqueeze(0).expand(batch, -1).clone()

    def _project_drive(self, drive: torch.Tensor) -> torch.Tensor:
        if drive.shape[-1] != self.d_model:
            raise ValueError(
                f"drive last dimension is {drive.shape[-1]}, expected {self.d_model}"
            )
        drive = drive.float()
        return (
            self.drive_proj(self.drive_norm(drive)) * self.drive_scale
            + self.drive_bias
        )

    def forward(
        self,
        drive: torch.Tensor,
        state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        detach_interval: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Consume ``[batch, tokens, d_model]`` and return token readouts.

        Returns ``(readout, state, stats)``. ``state`` can be fed back into the
        next call; it is only meant to be reused inside one document or one
        generation run.
        """

        if drive.ndim != 3:
            raise ValueError("drive must have shape [batch, tokens, d_model]")
        batch, tokens, _ = drive.shape
        if tokens == 0:
            raise ValueError("drive must contain at least one token")
        if state is None:
            state = self.initial_state(batch)
        if state.shape != (batch, self.n_neurons):
            raise ValueError(
                f"state must have shape {(batch, self.n_neurons)}, "
                f"got {tuple(state.shape)}"
            )
        if attention_mask is not None and attention_mask.shape != (batch, tokens):
            raise ValueError("attention_mask must match [batch, tokens]")

        interval = (
            self.config.detach_interval
            if detach_interval is None
            else int(detach_interval)
        )
        projected = self._project_drive(drive)
        dynamics = Dynamics(alpha=self.alpha, gain=self.effective_gain())
        runtime: _CachedRuntime | None = None
        if self.weight_mode == "sparse":
            graph = self._graph()
            runtime = _CachedRuntime(
                graph, dynamics, state, graph.sparse_matrix()
            )

        readouts: list[torch.Tensor] = []
        activity = torch.zeros((), device=drive.device)
        saturation = torch.zeros((), device=drive.device)
        memory = torch.zeros_like(state)
        resource = (
            torch.ones_like(state)
            if self.config.short_term_tau > 0
            else None
        )
        eligibility_flat = None
        eligibility_src = None
        eligibility_dst = None
        if self.spatial and self.biological_memory:
            edge_index = self.edge_mask.nonzero(as_tuple=False)
            eligibility_src = edge_index[:, 0]
            eligibility_dst = edge_index[:, 1]
            eligibility_flat = (
                eligibility_src * self.n_neurons + eligibility_dst
            )
        pre_trace_sum = torch.zeros(
            self.n_neurons, device=state.device, dtype=state.dtype
        )
        post_trace_sum = torch.zeros_like(pre_trace_sum)
        for token in range(tokens):
            previous = state
            drive_token = projected[:, token]
            valid = None
            if attention_mask is not None:
                valid = attention_mask[:, token].to(torch.bool)
                drive_token = drive_token * valid.unsqueeze(-1)

            pooled = torch.zeros_like(state)
            for _ in range(self.config.ticks_per_token):
                step_drive = drive_token
                if (
                    self.training
                    and self.config.recurrent_noise > 0
                ):
                    step_drive = step_drive + (
                        self.config.recurrent_noise
                        * torch.randn_like(step_drive)
                    )
                if runtime is not None:
                    runtime.state = runtime.step(step_drive)
                    state = runtime.state
                elif resource is not None:
                    resource = resource + (1.0 - resource) * (
                        1.0 - math.exp(-1.0 / self.config.short_term_tau)
                    )
                    current = F.linear(
                        state * resource, self.recurrent
                    )
                    state = dynamics.advance(state, current, step_drive)
                    resource = torch.clamp(
                        resource
                        - self.config.short_term_utilization * state,
                        min=0.1,
                        max=1.0,
                    )
                else:
                    current = F.linear(state, self.recurrent)
                    state = dynamics.advance(state, current, step_drive)
                if self.trace is not None:
                    self.trace.append(state.mean(dim=0).detach())
                pooled = pooled + state
            if self.config.readout == "mean":
                pooled = pooled / self.config.ticks_per_token
            else:
                pooled = state

            if valid is not None:
                state = torch.where(valid.unsqueeze(-1), state, previous)
                pooled = pooled * valid.unsqueeze(-1)
                if runtime is not None:
                    runtime.state = state

            if self.config.memory_decay > 0:
                memory = (
                    self.config.memory_decay * memory
                    + (1.0 - self.config.memory_decay) * pooled
                )
                readout_input = pooled + memory
            else:
                readout_input = pooled
            readout = self.readout_proj(self.readout_norm(readout_input))
            if (
                self.training
                and self.config.eligibility_decay > 0
                and self.biological_memory
            ):
                pre_trace_sum = pre_trace_sum + previous.mean(dim=0)
                post_trace_sum = post_trace_sum + state.mean(dim=0)
            if valid is not None:
                readout = readout * valid.unsqueeze(-1).to(readout.dtype)
            readouts.append(readout)

            if interval and (token + 1) % interval == 0:
                state = state.detach()
                if runtime is not None:
                    runtime.state = state

            activity = activity + state.mean().detach()
            saturation = saturation + (
                (state > 0.99).float().mean().detach()
            )

        output = torch.stack(readouts, dim=1)
        if (
            self.training
            and self.config.eligibility_decay > 0
            and self.biological_memory
            and tokens > 0
        ):
            with torch.no_grad():
                pre = pre_trace_sum / tokens
                post = post_trace_sum / tokens
                if eligibility_flat is not None:
                    flat = self.eligibility.view(-1)
                    flat[eligibility_flat] = (
                        self.config.eligibility_decay
                        * flat[eligibility_flat]
                        + post[eligibility_src]
                        * pre[eligibility_dst]
                    )
                else:
                    self.eligibility.mul_(
                        self.config.eligibility_decay
                    ).add_(post.unsqueeze(1) * pre.unsqueeze(0))
        stats = {
            "activity": (activity / max(tokens, 1)).detach(),
            "saturation": (saturation / max(tokens, 1)).detach(),
            "drive_rms": projected.detach().pow(2).mean().sqrt(),
        }
        if self.spatial:
            activity_vector = state.mean(dim=0).detach().float()
            normalized = (
                activity_vector / activity_vector.max().clamp_min(1e-6)
            ).clamp(0.0, 1.0)
            with torch.no_grad():
                self.last_activity.copy_(normalized)
        return output, state, stats

    @torch.no_grad()
    def population_summary(self, batch: int = 1) -> dict[str, float]:
        """Cheap diagnostics for the dry-run report (no recurrence)."""

        state = self.initial_state(batch)
        dynamics = Dynamics(alpha=self.alpha, gain=self.effective_gain())
        if self.weight_mode == "sparse":
            graph = self._graph()
            runtime = _CachedRuntime(
                graph, dynamics, state, graph.sparse_matrix()
            )
            runtime.step(0.0)
            state = runtime.state
        else:
            state = dynamics.advance(
                state, F.linear(state, self.recurrent), 0.0
            )
        return {
            "mean_state": float(state.mean()),
            "saturated_fraction": float((state > 0.99).float().mean()),
        }
