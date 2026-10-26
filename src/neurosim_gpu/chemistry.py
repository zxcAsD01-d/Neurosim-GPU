from __future__ import annotations

import math

import torch


class RegionalDopamineField:
    """Batched regional dopamine kinetics on a small sparse region graph.

    ``release_projection[source, region]`` maps dopaminergic source populations
    to their release fields. Positive reward-prediction errors (RPEs) add a
    phasic burst. Negative RPEs suppress the non-negative tonic release rate,
    producing a dip without treating dopamine concentration as a signed value.

    Diffusion uses an undirected edge list. Contributions are reduced by a
    sorted segmented prefix sum rather than atomics, so the implementation is
    deterministic on both CPU and CUDA. One advance costs
    ``O(B * (R + E_R + S * R))``; the final term is the normally small dense
    source-to-region projection.
    """

    def __init__(
        self,
        tonic_baseline: torch.Tensor,
        uptake_rate: torch.Tensor,
        decay_rate: torch.Tensor,
        release_projection: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        diffusion_rate: torch.Tensor,
        *,
        dt: float,
        burst_gain: torch.Tensor | float = 1.0,
        dip_gain: torch.Tensor | float = 1.0,
        region_volume: torch.Tensor | None = None,
        autoreceptor_gain: torch.Tensor | float = 0.0,
        autoreceptor_half_saturation: torch.Tensor | float = 0.25,
    ) -> None:
        self._validate_region_tensor("tonic_baseline", tonic_baseline)
        self._validate_region_tensor(
            "uptake_rate", uptake_rate, like=tonic_baseline
        )
        self._validate_region_tensor("decay_rate", decay_rate, like=tonic_baseline)
        if tonic_baseline.numel() == 0:
            raise ValueError("at least one region is required")
        if torch.any(tonic_baseline < 0):
            raise ValueError("tonic_baseline must be non-negative")
        if torch.any(uptake_rate < 0) or torch.any(decay_rate < 0):
            raise ValueError("uptake_rate and decay_rate must be non-negative")

        if release_projection.ndim != 2:
            raise ValueError("release_projection must have shape [sources, regions]")
        if release_projection.shape[1] != tonic_baseline.numel():
            raise ValueError(
                "release_projection's region dimension must match tonic_baseline"
            )
        self._validate_floating_tensor(
            "release_projection", release_projection, like=tonic_baseline
        )
        if torch.any(release_projection < 0):
            raise ValueError("release_projection must be non-negative")

        if edge_src.ndim != 1 or edge_dst.ndim != 1 or diffusion_rate.ndim != 1:
            raise ValueError("edge_src, edge_dst and diffusion_rate must be rank-1")
        if not (edge_src.numel() == edge_dst.numel() == diffusion_rate.numel()):
            raise ValueError(
                "edge_src, edge_dst and diffusion_rate must have equal length"
            )
        if edge_src.dtype not in (torch.int32, torch.int64) or edge_dst.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("region edge indices must use torch.int32 or torch.int64")
        if edge_src.device != tonic_baseline.device or edge_dst.device != tonic_baseline.device:
            raise ValueError("all tensors must be on the same device")
        self._validate_floating_tensor(
            "diffusion_rate", diffusion_rate, like=tonic_baseline
        )
        if torch.any(diffusion_rate < 0):
            raise ValueError("diffusion_rate must be non-negative")

        n_regions = tonic_baseline.numel()
        src = edge_src.to(torch.int64)
        dst = edge_dst.to(torch.int64)
        if src.numel():
            if (
                int(src.min()) < 0
                or int(dst.min()) < 0
                or int(src.max()) >= n_regions
                or int(dst.max()) >= n_regions
            ):
                raise ValueError("region edge index out of range")
            if torch.any(src == dst):
                raise ValueError("diffusion self-edges are not allowed")
            lower = torch.minimum(src, dst)
            upper = torch.maximum(src, dst)
            undirected_key = lower * n_regions + upper
            if torch.unique(undirected_key).numel() != src.numel():
                raise ValueError("duplicate undirected diffusion edge")

        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        burst = self._region_parameter("burst_gain", burst_gain, tonic_baseline)
        dip = self._region_parameter("dip_gain", dip_gain, tonic_baseline)
        if region_volume is None:
            volume = torch.ones_like(tonic_baseline)
        else:
            self._validate_region_tensor(
                "region_volume", region_volume, like=tonic_baseline
            )
            if torch.any(region_volume <= 0):
                raise ValueError("region_volume must be positive")
            volume = region_volume

        self.tonic_baseline = tonic_baseline
        self.uptake_rate = uptake_rate
        self.decay_rate = decay_rate
        self.release_projection = release_projection
        self.edge_src = src
        self.edge_dst = dst
        self.diffusion_rate = diffusion_rate
        self.dt = float(dt)
        self.burst_gain = burst
        self.dip_gain = dip
        self.region_volume = volume
        source_reference = torch.ones(
            self.n_sources,
            dtype=tonic_baseline.dtype,
            device=tonic_baseline.device,
        )
        self.autoreceptor_gain = self._region_parameter(
            "autoreceptor_gain", autoreceptor_gain, source_reference
        )
        self.autoreceptor_half_saturation = self._region_parameter(
            "autoreceptor_half_saturation",
            autoreceptor_half_saturation,
            source_reference,
        )
        if torch.any(self.autoreceptor_half_saturation <= 0):
            raise ValueError("autoreceptor_half_saturation must be positive")
        self._has_autoreceptor = bool(torch.any(self.autoreceptor_gain > 0))
        projection_mass = release_projection.sum(dim=1).clamp_min(
            torch.finfo(tonic_baseline.dtype).tiny
        )
        self._source_region_weight = release_projection / projection_mass[:, None]
        tonic_source_concentration = (
            tonic_baseline @ self._source_region_weight.T
        )
        self._tonic_autoreceptor_occupancy = tonic_source_concentration / (
            self.autoreceptor_half_saturation + tonic_source_concentration
        )

        # Every physical edge produces one contribution at each endpoint.
        targets = torch.cat((src, dst))
        self._target_order = torch.argsort(targets, stable=True)
        sorted_targets = targets[self._target_order]
        self._segment_targets, counts = torch.unique_consecutive(
            sorted_targets, return_counts=True
        )
        self._segment_ends = torch.cumsum(counts, dim=0)
        self._segment_starts = self._segment_ends - counts

        self.validate_timestep()
        tonic_release = self._raw_tonic_release_rate().detach()
        tolerance = 64 * torch.finfo(tonic_baseline.dtype).eps * max(
            1.0,
            float((self.clearance_rate * tonic_baseline).detach().abs().max()),
        )
        if torch.any(tonic_release < -tolerance):
            raise ValueError(
                "tonic_baseline requires a negative tonic source under the "
                "configured diffusion; choose a feasible baseline or rates"
            )
        self._tonic_release_rate = torch.clamp(tonic_release, min=0.0)
        regional_projection = release_projection.sum(dim=0)
        projected = regional_projection > 0
        source_fraction = torch.where(
            projected.unsqueeze(0),
            release_projection
            / regional_projection.clamp_min(
                torch.finfo(tonic_baseline.dtype).tiny
            ).unsqueeze(0),
            torch.zeros_like(release_projection),
        )
        self._tonic_release_by_source = (
            source_fraction * self._tonic_release_rate.unsqueeze(0)
        )
        self._background_tonic_release = torch.where(
            projected,
            torch.zeros_like(self._tonic_release_rate),
            self._tonic_release_rate,
        )

    @staticmethod
    def _validate_floating_tensor(
        name: str,
        value: torch.Tensor,
        *,
        like: torch.Tensor | None = None,
    ) -> None:
        if not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor")
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        if like is not None and (value.dtype != like.dtype or value.device != like.device):
            raise ValueError(f"{name} must match tonic_baseline's dtype and device")

    @classmethod
    def _validate_region_tensor(
        cls,
        name: str,
        value: torch.Tensor,
        *,
        like: torch.Tensor | None = None,
    ) -> None:
        if value.ndim != 1:
            raise ValueError(f"{name} must be a rank-1 tensor")
        cls._validate_floating_tensor(name, value, like=like)
        if like is not None and value.shape != like.shape:
            raise ValueError(f"{name} must have the same shape as tonic_baseline")

    @classmethod
    def _region_parameter(
        cls,
        name: str,
        value: torch.Tensor | float,
        like: torch.Tensor,
    ) -> torch.Tensor:
        parameter = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        if parameter.ndim == 0:
            parameter = parameter.expand_as(like)
        cls._validate_region_tensor(name, parameter, like=like)
        if torch.any(parameter < 0):
            raise ValueError(f"{name} must be non-negative")
        return parameter

    @property
    def n_regions(self) -> int:
        return self.tonic_baseline.numel()

    @property
    def n_sources(self) -> int:
        return self.release_projection.shape[0]

    @property
    def clearance_rate(self) -> torch.Tensor:
        return self.uptake_rate + self.decay_rate

    def _sum_by_target(self, directed_values: torch.Tensor) -> torch.Tensor:
        """Deterministically sum directed edge values into their target regions."""

        batch = directed_values.shape[0]
        result = directed_values.new_zeros((batch, self.n_regions))
        if self._target_order.numel() == 0:
            return result
        ordered = directed_values.index_select(1, self._target_order)
        prefix = torch.cat(
            (ordered.new_zeros((batch, 1)), torch.cumsum(ordered, dim=1)),
            dim=1,
        )
        segment_sum = (
            prefix.index_select(1, self._segment_ends)
            - prefix.index_select(1, self._segment_starts)
        )
        result[:, self._segment_targets] = segment_sum
        return result

    @property
    def diffusion_degree(self) -> torch.Tensor:
        if self.diffusion_rate.numel() == 0:
            return self.diffusion_rate.new_zeros(self.n_regions)
        directed_rate = torch.cat((self.diffusion_rate, self.diffusion_rate))
        return self._sum_by_target(directed_rate.unsqueeze(0))[0] / self.region_volume

    def diffusion_flux(self, concentration: torch.Tensor) -> torch.Tensor:
        """Return the conservative regional diffusion derivative."""

        self._validate_concentration(concentration)
        return self._diffusion_flux_impl(concentration)

    def _diffusion_flux_impl(self, concentration: torch.Tensor) -> torch.Tensor:
        if self.diffusion_rate.numel() == 0:
            return torch.zeros_like(concentration)
        edge_flux = self.diffusion_rate * (
            concentration[:, self.edge_dst] - concentration[:, self.edge_src]
        )
        net_flux = self._sum_by_target(
            torch.cat((edge_flux, -edge_flux), dim=1)
        )
        return net_flux / self.region_volume

    @property
    def tonic_release_rate(self) -> torch.Tensor:
        """Non-negative source rate that holds ``tonic_baseline`` stationary."""

        return self._tonic_release_rate

    def _raw_tonic_release_rate(self) -> torch.Tensor:
        baseline = self.tonic_baseline.unsqueeze(0)
        diffusion = self.diffusion_flux(baseline)[0]
        return self.clearance_rate * self.tonic_baseline - diffusion

    @property
    def maximum_stable_timestep(self) -> float:
        """Largest Euler step that guarantees non-negative passive dynamics."""

        maximum_rate = float(
            (self.clearance_rate + self.diffusion_degree).detach().max()
        )
        return math.inf if maximum_rate == 0.0 else 1.0 / maximum_rate

    def validate_timestep(self, dt: float | None = None) -> None:
        """Validate the explicit Euler positivity/stability condition.

        For each region, ``dt * (clearance + incident diffusion) <= 1`` makes
        the passive update a non-negative weighted sum plus non-negative
        release. This is stricter than mere linear stability and therefore also
        protects the concentration invariant.
        """

        candidate = self.dt if dt is None else float(dt)
        if not math.isfinite(candidate) or candidate <= 0:
            raise ValueError("dt must be finite and positive")
        limit = self.maximum_stable_timestep
        tolerance = 16 * torch.finfo(self.tonic_baseline.dtype).eps
        if candidate > limit * (1.0 + tolerance):
            raise ValueError(
                f"dt={candidate:g} violates the non-negative Euler stability "
                f"limit dt<={limit:g}"
            )

    def _validate_concentration(self, concentration: torch.Tensor) -> None:
        if concentration.ndim != 2 or concentration.shape[1] != self.n_regions:
            raise ValueError("concentration must have shape [batch, regions]")
        self._validate_floating_tensor(
            "concentration", concentration, like=self.tonic_baseline
        )
        if torch.any(concentration < 0):
            raise ValueError("concentration must be non-negative")

    def _validate_source_signal(
        self, name: str, signal: torch.Tensor, batch: int
    ) -> torch.Tensor:
        if signal.ndim == 1:
            if self.n_sources == 1 and signal.shape[0] == batch:
                signal = signal.unsqueeze(1)
            elif batch == 1 and signal.shape[0] == self.n_sources:
                signal = signal.unsqueeze(0)
        if signal.ndim != 2 or signal.shape != (batch, self.n_sources):
            raise ValueError(f"{name} must have shape [batch, sources]")
        self._validate_floating_tensor(name, signal, like=self.tonic_baseline)
        return signal

    def advance(
        self,
        concentration: torch.Tensor,
        rpe: torch.Tensor,
        source_release: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance one chemistry tick and return ``[batch, regions]``.

        ``source_release`` is an optional non-negative source-population release
        rate that uses the same anatomical projection as the RPE response.
        """

        self._validate_concentration(concentration)
        rpe = self._validate_source_signal("rpe", rpe, concentration.shape[0])
        if source_release is not None:
            source_release = self._validate_source_signal(
                "source_release", source_release, concentration.shape[0]
            )
            if torch.any(source_release < 0):
                raise ValueError("source_release must be non-negative")
        return self._advance_impl(concentration, rpe, source_release)

    def advance_unchecked(
        self,
        concentration: torch.Tensor,
        rpe: torch.Tensor,
        source_release: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance tensors already validated by an owning runtime."""

        return self._advance_impl(concentration, rpe, source_release)

    def _advance_impl(
        self,
        concentration: torch.Tensor,
        rpe: torch.Tensor,
        source_release: torch.Tensor | None,
    ) -> torch.Tensor:
        positive_source = torch.clamp(rpe, min=0.0)
        if self._has_autoreceptor:
            source_concentration = concentration @ self._source_region_weight.T
            occupancy = source_concentration / (
                self.autoreceptor_half_saturation + source_concentration
            )
            excess = torch.clamp(
                occupancy - self._tonic_autoreceptor_occupancy,
                min=0.0,
            )
            positive_source = positive_source / (
                1.0 + self.autoreceptor_gain * excess
            )
        positive_rpe = positive_source @ self.release_projection
        negative_source = torch.clamp(-rpe, min=0.0)
        dip_fraction = torch.clamp(
            negative_source.unsqueeze(2)
            * self.release_projection.unsqueeze(0)
            * self.dip_gain.reshape(1, 1, -1),
            min=0.0,
            max=1.0,
        )
        release = self._background_tonic_release + (
            self._tonic_release_by_source.unsqueeze(0)
            * (1.0 - dip_fraction)
        ).sum(dim=1)
        release = release + self.burst_gain * positive_rpe
        if source_release is not None:
            release = release + source_release @ self.release_projection

        derivative = (
            self._diffusion_flux_impl(concentration)
            + release
            - self.clearance_rate * concentration
        )
        return torch.clamp(concentration + self.dt * derivative, min=0.0)
