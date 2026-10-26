from __future__ import annotations

import argparse
import gc
import statistics
import time

import torch

from neurosim_gpu import Dynamics, HybridPELSRuntime, MatrixRuntime, random_topology


class RawSparseLinear(MatrixRuntime):
    """Same-M cuSPARSE linear layer used as the parity denominator."""

    def step(self, drive: float = 0.0) -> torch.Tensor:
        del drive
        self.output = self._propagate(self.state)
        return self.output


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure(
    factory,
    steps: int,
    warmup: int,
    drive: float,
    samples: int,
) -> tuple[float, float]:
    measurements = []
    for _ in range(samples):
        runtime = factory()
        for _ in range(warmup):
            runtime.step(drive)
        synchronize(runtime.state.device)
        start = time.perf_counter()
        for _ in range(steps):
            runtime.step(drive)
        synchronize(runtime.state.device)
        measurements.append((time.perf_counter() - start) * 1000.0 / steps)
    return statistics.median(measurements), min(measurements)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, default=100_000)
    parser.add_argument("--edges", type=int, default=5_000_000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--theta", type=float, default=0.01)
    parser.add_argument("--drive", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--scatter-blocks", type=int, default=128)
    parser.add_argument("--canonical-blocks", type=int, default=4096)
    parser.add_argument("--suite", choices=("full", "parity"), default="full")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this benchmark")
    device = torch.device("cuda")
    topology = random_topology(
        args.neurons,
        args.edges,
        device=device,
        seed=args.seed,
    )
    dynamics = Dynamics(
        alpha=torch.full((args.neurons,), 0.95, device=device),
        gain=torch.full((args.neurons,), 0.8, device=device),
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    initial = torch.rand(
        args.batch,
        args.neurons,
        device=device,
        generator=generator,
    )
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()

    factories = {
        "raw-spmm": lambda: RawSparseLinear(
            topology, dynamics, initial.clone()
        ),
    }
    if args.suite == "full":
        factories.update(
            {
                "matrix": lambda: MatrixRuntime(
                    topology, dynamics, initial.clone()
                ),
                "pels-matrix": lambda: HybridPELSRuntime(
                    topology,
                    dynamics,
                    initial.clone(),
                    theta=args.theta,
                    propagation="matrix",
                ),
                "pels-auto": lambda: HybridPELSRuntime(
                    topology,
                    dynamics,
                    initial.clone(),
                    theta=args.theta,
                    propagation="auto",
                ),
            }
        )
    if args.native:
        from neurosim_gpu.native import (
            NativeEventPELSRuntime,
            NativeCanonicalLinear,
            NativeCanonicalMatrixRuntime,
            NativeMatrixRuntime,
            load_native_extension,
        )

        extension = load_native_extension(verbose=False)
        factories["native-matrix"] = lambda: NativeMatrixRuntime(
            topology,
            dynamics,
            initial.clone(),
            extension=extension,
        )
        factories["canonical-spmm"] = lambda: NativeCanonicalLinear(
            topology,
            initial.clone(),
            blocks=args.canonical_blocks,
            extension=extension,
        )
        factories["canonical-fused"] = lambda: NativeCanonicalMatrixRuntime(
            topology,
            dynamics,
            initial.clone(),
            blocks=args.canonical_blocks,
            extension=extension,
        )
        if args.suite == "full":
            factories["native-event"] = lambda: NativeEventPELSRuntime(
                topology,
                dynamics,
                initial.clone(),
                theta=args.theta,
                scatter_blocks=args.scatter_blocks,
                extension=extension,
            )

    actual_edges = topology.n_edges
    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"N={args.neurons:,} M={actual_edges:,} B={args.batch} "
        f"allocated_topology={topology.storage_bytes() / 2**20:.1f} MiB "
        f"torch_hybrid={topology.torch_hybrid_storage_bytes() / 2**20:.1f} MiB "
        f"native_topology={topology.native_storage_bytes() / 2**20:.1f} MiB"
    )
    print(
        "cuda_memory   "
        f"allocated={torch.cuda.memory_allocated(device) / 2**20:.1f} MiB "
        f"reserved={torch.cuda.memory_reserved(device) / 2**20:.1f} MiB"
    )
    print("runtime       median_ms   best_ms   full_graph_equiv_Medges/s")
    timings: dict[str, float] = {}
    for name, factory in factories.items():
        median_ms, best_ms = measure(
            factory,
            args.steps,
            args.warmup,
            args.drive,
            args.samples,
        )
        timings[name] = median_ms
        edge_rate = actual_edges * args.batch / (median_ms * 1e3)
        print(f"{name:12} {median_ms:10.4f} {best_ms:9.4f} {edge_rate:11.2f}")
    if args.native:
        print(
            "parity        "
            f"native/cusparse={timings['native-matrix'] / timings['raw-spmm']:.4f} "
            f"fused/canonical={timings['canonical-fused'] / timings['canonical-spmm']:.4f} "
            f"fused/cusparse={timings['canonical-fused'] / timings['raw-spmm']:.4f} "
            f"native_memory/torch={topology.native_storage_bytes() / topology.torch_hybrid_storage_bytes():.4f}"
        )

    if args.suite != "full":
        return

    probe = HybridPELSRuntime(
        topology,
        dynamics,
        initial.clone(),
        theta=args.theta,
        propagation="auto",
        policy_interval=1,
    )
    path_counts: dict[str, int] = {}
    active_neurons = []
    active_edges = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            probe.step(args.drive)
        for _ in range(args.steps):
            _, stats = probe.step(args.drive, collect_stats=True)
            path_counts[stats.path] = path_counts.get(stats.path, 0) + 1
            active_neurons.append(stats.active_neurons)
            active_edges.append(stats.active_edges)
    print(
        "activity      "
        f"paths={path_counts} "
        f"mean_active_neurons={statistics.mean(active_neurons):.1f} "
        f"mean_active_edges={statistics.mean(active_edges):.1f}"
    )


if __name__ == "__main__":
    main()
