"""Measure real-time per-tick latency and deadline misses.

Unlike throughput benchmarks, this synchronizes around every tick. It measures
the latency an embodied control loop observes, including Python dispatch and
CUDA completion. Topology construction and JIT compilation are outside samples.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import torch

from neurosim_gpu import (
    Dynamics,
    load_topology,
    random_topology,
)
from neurosim_gpu.native import (
    NativeCanonicalMatrixRuntime,
    NativeEventPELSRuntime,
    load_native_extension,
)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default=None)
    parser.add_argument("--neurons", type=int, default=50_000)
    parser.add_argument("--edges", type=int, default=1_000_000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--theta", type=float, default=0.05)
    parser.add_argument("--mode", choices=("matrix", "event"), default="matrix")
    parser.add_argument("--deadline-ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=131)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for real-time latency measurement")
    device = torch.device("cuda")
    extension = load_native_extension(verbose=False)
    if args.topology:
        topology = load_topology(args.topology, device=device)
        n_neurons = topology.n_neurons
    else:
        topology = random_topology(
            args.neurons, args.edges, device=device, seed=args.seed
        )
        n_neurons = args.neurons
    dynamics = Dynamics(
        alpha=torch.full((n_neurons,), 0.95, device=device),
        gain=torch.full((n_neurons,), 0.8, device=device),
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    initial = torch.rand(
        args.batch, n_neurons, device=device, generator=generator
    )

    def factory():
        if args.mode == "matrix":
            return NativeCanonicalMatrixRuntime(
                topology, dynamics, initial.clone(), extension=extension
            )
        return NativeEventPELSRuntime(
            topology,
            dynamics,
            initial.clone(),
            theta=args.theta,
            extension=extension,
        )

    all_samples: list[list[float]] = []
    for _ in range(args.samples):
        runtime = factory()
        with torch.inference_mode():
            for _ in range(args.warmup):
                runtime.step(0.15)
            sync(device)
            latencies: list[float] = []
            for _ in range(args.ticks):
                start = time.perf_counter()
                runtime.step(0.15)
                sync(device)
                latencies.append((time.perf_counter() - start) * 1000.0)
            all_samples.append(latencies)
        del runtime
        gc.collect()
        torch.cuda.empty_cache()

    values = [value for sample in all_samples for value in sample]
    deadline_misses = (
        sum(value > args.deadline_ms for value in values)
        if args.deadline_ms > 0
        else 0
    )
    print(
        f"device={torch.cuda.get_device_name(device)} mode={args.mode} "
        f"N={n_neurons:,} M={topology.n_edges:,} B={args.batch} "
        f"samples={args.samples} ticks={args.ticks}"
    )
    print(
        "latency_ms "
        f"min={min(values):.4f} "
        f"p50={percentile(values, 0.50):.4f} "
        f"p95={percentile(values, 0.95):.4f} "
        f"p99={percentile(values, 0.99):.4f} "
        f"max={max(values):.4f} "
        f"mean={statistics.mean(values):.4f}"
    )
    if args.deadline_ms > 0:
        print(
            f"deadline_ms={args.deadline_ms:.4f} misses={deadline_misses} "
            f"miss_rate={deadline_misses / len(values):.6f}"
        )


if __name__ == "__main__":
    main()
