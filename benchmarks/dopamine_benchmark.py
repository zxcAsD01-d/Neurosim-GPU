"""Fair component ablation for the dopamine runtime.

Each timed sample starts from an independent topology, neural state, chemistry
field, accumulators, traces, and RPE sequence.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from typing import Callable

import torch

from neurosim_gpu import (
    DopamineMatrixRuntime,
    DopaminePlasticityConfig,
    DopamineReceptors,
    Dynamics,
    MatrixRuntime,
    NativeDopamineEventRuntime,
    NativeDopamineMatrixRuntime,
    RegionalDopamineField,
    SparseTopology,
    random_topology,
)
from neurosim_gpu.native import NativeCanonicalMatrixRuntime, load_native_extension
from neurosim_gpu.runtime import MatrixRuntime as BaseMatrixRuntime
from neurosim_gpu.topology_io import topology_from_dict, topology_to_dict


class RawSparseLinear(BaseMatrixRuntime):
    def step(self, drive: float = 0.0) -> torch.Tensor:
        del drive
        return self._propagate(self.state)


def make_field(regions: int, sources: int, device: torch.device) -> RegionalDopamineField:
    dtype = torch.float32
    region = torch.arange(regions, device=device)
    projection = torch.zeros(sources, regions, device=device, dtype=dtype)
    projection[region.remainder(sources), region] = 1.0
    return RegionalDopamineField(
        tonic_baseline=torch.full((regions,), 0.5, device=device, dtype=dtype),
        uptake_rate=torch.full((regions,), 0.10, device=device, dtype=dtype),
        decay_rate=torch.full((regions,), 0.02, device=device, dtype=dtype),
        release_projection=projection,
        edge_src=region.to(torch.int64),
        edge_dst=torch.roll(region, shifts=-1).to(torch.int64),
        diffusion_rate=torch.full((regions,), 0.04, device=device, dtype=dtype),
        dt=0.1,
        burst_gain=0.5,
        dip_gain=0.5,
    )


def measure(
    factory: Callable[[], object],
    rpe_values: tuple[float, ...],
    *,
    warmup: int,
    steps: int,
    samples: int,
    device: torch.device,
) -> float:
    measurements: list[float] = []
    for _ in range(samples):
        runtime = factory()
        rpe = None
        if hasattr(runtime, "chemistry"):
            rpe = torch.zeros(
                runtime.state.shape[0], runtime.chemistry.n_sources, device=device
            )
        with torch.inference_mode():
            for tick in range(warmup):
                if rpe is None:
                    runtime.step(0.15)
                else:
                    rpe.fill_(rpe_values[tick % len(rpe_values)])
                    runtime.step(0.15, rpe=rpe)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for tick in range(steps):
                if rpe is None:
                    runtime.step(0.15)
                else:
                    rpe.fill_(rpe_values[(tick + warmup) % len(rpe_values)])
                    runtime.step(0.15, rpe=rpe)
            torch.cuda.synchronize(device)
        measurements.append((time.perf_counter() - start) * 1000.0 / steps)
        del runtime, rpe
        gc.collect()
        torch.cuda.empty_cache()
    return statistics.median(measurements)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default=None)
    parser.add_argument("--neurons", type=int, default=50_000)
    parser.add_argument("--edges", type=int, default=1_000_000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--regions", type=int, default=64)
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--theta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    if args.topology:
        topology = load_topology(args.topology, device=device)
        if not isinstance(topology, SparseTopology):
            raise SystemExit("mutable SparseTopology is required")
        n_neurons = topology.n_neurons
    else:
        topology = random_topology(
            args.neurons, args.edges, device=device, seed=args.seed
        )
        n_neurons = args.neurons
    topology_payload = topology_to_dict(topology)
    dynamics = Dynamics(
        alpha=torch.full((n_neurons,), 0.95, device=device),
        gain=torch.full((n_neurons,), 0.8, device=device),
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    initial = torch.rand(
        args.batch, n_neurons, device=device, generator=generator
    )
    neuron_region = torch.arange(
        n_neurons, device=device, dtype=torch.int64
    ).remainder(args.regions)
    d1 = (torch.arange(n_neurons, device=device) % 2 == 0).float()
    receptors = DopamineReceptors(
        d1_density=d1,
        d2_density=1.0 - d1,
        tonic_concentration=0.5,
        d1_half_saturation=0.5,
        d2_half_saturation=0.25,
    )
    rpe_values = (0.25, 0.0, -0.20, 0.0)

    def topo() -> SparseTopology:
        loaded = topology_from_dict(topology_payload, device=device)
        assert isinstance(loaded, SparseTopology)
        return loaded

    def raw_runtime() -> RawSparseLinear:
        return RawSparseLinear(topo(), dynamics, initial.clone())

    def base_runtime() -> MatrixRuntime:
        return MatrixRuntime(topo(), dynamics, initial.clone())

    def dopamine_runtime(track: bool) -> DopamineMatrixRuntime:
        return DopamineMatrixRuntime(
            topo(),
            dynamics,
            initial.clone(),
            chemistry=make_field(args.regions, args.sources, device),
            receptors=receptors,
            neuron_region=neuron_region,
            config=DopaminePlasticityConfig(
                track_eligibility=track,
                chemistry_interval=10,
                validate_inputs=False,
            ),
        )

    factories: dict[str, Callable[[], object]] = {
        "raw-spmm": raw_runtime,
        "matrix": base_runtime,
        "dopamine-chemistry-k10": lambda: dopamine_runtime(False),
        "dopamine-eligibility-k10": lambda: dopamine_runtime(True),
    }
    if args.native:
        extension = load_native_extension(verbose=False)
        factories["native-matrix-baseline"] = lambda: NativeCanonicalMatrixRuntime(
            topo(), dynamics, initial.clone(), extension=extension
        )
        factories["native-dopamine-k10"] = lambda: NativeDopamineMatrixRuntime(
            topo(),
            dynamics,
            initial.clone(),
            chemistry=make_field(args.regions, args.sources, device),
            receptors=receptors,
            neuron_region=neuron_region,
            config=DopaminePlasticityConfig(
                track_eligibility=True,
                chemistry_interval=10,
                validate_inputs=False,
            ),
            extension=extension,
        )
        factories["native-dopamine-event-k10"] = lambda: NativeDopamineEventRuntime(
            topo(),
            dynamics,
            initial.clone(),
            chemistry=make_field(args.regions, args.sources, device),
            receptors=receptors,
            neuron_region=neuron_region,
            theta=args.theta,
            config=DopaminePlasticityConfig(
                track_eligibility=False,
                chemistry_interval=10,
                validate_inputs=False,
            ),
            extension=extension,
        )

    torch.cuda.synchronize(device)
    print(
        f"device={torch.cuda.get_device_name(device)} N={n_neurons:,} "
        f"M={topology.n_edges:,} B={args.batch} R={args.regions} "
        f"samples={args.samples} independent=true"
    )
    print("path                         median_ms   ratio_to_raw")
    timings: dict[str, float] = {}
    for name, factory in factories.items():
        timings[name] = measure(
            factory,
            rpe_values,
            warmup=args.warmup,
            steps=args.steps,
            samples=args.samples,
            device=device,
        )
        print(
            f"{name:28} {timings[name]:10.4f} "
            f"{timings[name] / timings['raw-spmm']:13.4f}"
        )


if __name__ == "__main__":
    main()
