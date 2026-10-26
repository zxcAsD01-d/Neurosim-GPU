#!/usr/bin/env python3
"""Repeatable performance grid for the sparse neural core.

Answers three questions on whatever device you give it:

1. **Cost of the tick.** How much does the neural update cost on top of one raw
   sparse matrix multiplication over the same ``M``?
2. **Cost of event sparsity.** What does the persistent-current path cost when
   only emitted deltas are propagated, and how does that track the fraction of
   active edges as the emission threshold rises?
3. **Cost per synapse and per neuron.** Throughput (M synapses/s) and memory
   (bytes per synapse) as ``N``, ``M`` and the batch size grow.

Runs on CPU (no CUDA needed) and on CUDA. Results are emitted as JSON so the
documentation can quote machine-readable numbers:

    python benchmarks/performance_suite.py --device cpu
    python benchmarks/performance_suite.py --device cuda --native

The script never mutates the topology between runtimes, so every number in a
row uses identical weights, activity and batch.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from neurosim_gpu import Dynamics, HybridPELSRuntime, MatrixRuntime, random_topology


class RawSparseLinear(MatrixRuntime):
    """One sparse multiplication, no neural update: the parity denominator."""

    def step(self, drive: torch.Tensor | float = 0.0) -> torch.Tensor:
        del drive
        return self._propagate(self.state)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))]


def measure(
    factory: Callable[[], Any],
    *,
    steps: int,
    warmup: int,
    repeats: int,
    drive: float,
    device: torch.device,
) -> dict[str, float]:
    """Median/p95 per-step latency in milliseconds."""

    samples: list[float] = []
    best = float("inf")
    for _ in range(repeats):
        runtime = factory()
        with torch.inference_mode():
            for _ in range(warmup):
                runtime.step(drive)
        sync(device)
        repeat_samples: list[float] = []
        for _ in range(steps):
            start = time.perf_counter()
            with torch.inference_mode():
                runtime.step(drive)
            sync(device)
            repeat_samples.append((time.perf_counter() - start) * 1000.0)
        # The first step of a fresh runtime pays lazy allocation; drop it.
        kept = repeat_samples[1:] if len(repeat_samples) > 1 else repeat_samples
        samples.extend(kept)
        best = min(best, min(kept))
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "best_ms": best,
    }


def activity_profile(
    topology,
    dynamics,
    initial: torch.Tensor,
    *,
    theta: float,
    steps: int,
    warmup: int,
    drive: float,
) -> dict[str, float]:
    runtime = HybridPELSRuntime(
        topology,
        dynamics,
        initial.clone(),
        theta=theta,
        propagation="event",
    )
    with torch.inference_mode():
        for _ in range(warmup):
            runtime.step(drive)
        active_edges = 0
        active_neurons = 0
        for _ in range(steps):
            _, stats = runtime.step(drive, collect_stats=True)
            active_edges += stats.active_edges
            active_neurons += stats.active_neurons
    total_edges = max(steps * topology.n_edges * initial.shape[0], 1)
    total_neurons = max(steps * topology.n_neurons * initial.shape[0], 1)
    return {
        "active_edge_fraction": active_edges / total_edges,
        "active_neuron_fraction": active_neurons / total_neurons,
    }


def run_config(
    *,
    device: torch.device,
    neurons: int,
    degree: int,
    batch: int,
    steps: int,
    warmup: int,
    repeats: int,
    drive: float,
    thetas: list[float],
    seed: int,
    native: bool,
) -> list[dict[str, Any]]:
    build_start = time.perf_counter()
    topology = random_topology(neurons, neurons * degree, device=device, seed=seed)
    dynamics = Dynamics(
        alpha=torch.full((neurons,), 0.95, device=device),
        gain=torch.full((neurons,), 0.8, device=device),
    )
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    initial = torch.rand(batch, neurons, device=device, generator=generator)
    sync(device)
    build_seconds = time.perf_counter() - build_start

    edges = topology.n_edges
    common = {
        "device": str(device),
        "neurons": neurons,
        "degree": degree,
        "synapses_requested": neurons * degree,
        "synapses": edges,
        "batch": batch,
        "build_seconds": round(build_seconds, 4),
        "storage_mib": round(topology.storage_bytes() / 2**20, 3),
        "torch_hybrid_mib": round(topology.torch_hybrid_storage_bytes() / 2**20, 3),
        "native_mib": round(topology.native_storage_bytes() / 2**20, 3),
        "bytes_per_synapse": round(topology.storage_bytes() / max(edges, 1), 2),
    }

    factories: dict[str, Callable[[], Any]] = {
        "raw-spmm": lambda: RawSparseLinear(topology, dynamics, initial.clone()),
        "matrix": lambda: MatrixRuntime(topology, dynamics, initial.clone()),
        "pels-matrix": lambda: HybridPELSRuntime(
            topology,
            dynamics,
            initial.clone(),
            theta=0.05,
            propagation="matrix",
        ),
    }
    if native and device.type == "cuda":
        from neurosim_gpu.native import (
            NativeCanonicalMatrixRuntime,
            NativeEventPELSRuntime,
            load_native_extension,
        )

        extension = load_native_extension()
        factories["native-matrix"] = lambda: NativeCanonicalMatrixRuntime(
            topology, dynamics, initial.clone(), extension=extension
        )
        factories["native-event"] = lambda: NativeEventPELSRuntime(
            topology,
            dynamics,
            initial.clone(),
            theta=0.05,
            extension=extension,
        )

    rows: list[dict[str, Any]] = []
    raw_median: float | None = None
    for name, factory in factories.items():
        timing = measure(
            factory, steps=steps, warmup=warmup, repeats=repeats, drive=drive, device=device
        )
        if name == "raw-spmm":
            raw_median = timing["median_ms"]
        rows.append(
            {
                **common,
                "runtime": name,
                "theta": None,
                "steps": steps,
                "repeats": repeats,
                **timing,
                "synapses_per_second": edges
                * batch
                / max(timing["median_ms"] / 1e3, 1e-9),
                "ratio_to_raw": (
                    timing["median_ms"] / raw_median if raw_median else None
                ),
            }
        )

    for theta in thetas:
        factory = lambda theta=theta: HybridPELSRuntime(
            topology,
            dynamics,
            initial.clone(),
            theta=theta,
            propagation="event",
        )
        timing = measure(
            factory, steps=steps, warmup=warmup, repeats=repeats, drive=drive, device=device
        )
        profile = activity_profile(
            topology,
            dynamics,
            initial,
            theta=theta,
            steps=steps,
            warmup=warmup,
            drive=drive,
        )
        rows.append(
            {
                **common,
                "runtime": "event",
                "theta": theta,
                "steps": steps,
                "repeats": repeats,
                **timing,
                **profile,
                "synapses_per_second": edges
                * batch
                * profile["active_edge_fraction"]
                / max(timing["median_ms"] / 1e3, 1e-9),
                "ratio_to_raw": timing["median_ms"] / raw_median if raw_median else None,
            }
        )
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| runtime | theta | median ms | p95 ms | ratio to raw | active edges | M synapses/s |"
    )
    lines = [header, "|---|---|---|---|---|---|---|"]
    for row in rows:
        theta = "-" if row["theta"] is None else f"{row['theta']:g}"
        active = (
            "-"
            if row.get("active_edge_fraction") is None
            else f"{row['active_edge_fraction'] * 100:.1f}%"
        )
        ratio = "-" if row["ratio_to_raw"] is None else f"{row['ratio_to_raw']:.3f}x"
        lines.append(
            f"| {row['runtime']} | {theta} | {row['median_ms']:.3f} | "
            f"{row['p95_ms']:.3f} | {ratio} | {active} | "
            f"{row['synapses_per_second'] / 1e6:.2f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--neurons", default="2000,8000,32000")
    parser.add_argument("--degrees", default="8,32")
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--thetas", default="0.02,0.05,0.1")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--drive", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--native", action="store_true", help="include native CUDA runtimes")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available", file=sys.stderr)
        return 2
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    neurons = [int(v) for v in args.neurons.split(",")]
    degrees = [int(v) for v in args.degrees.split(",")]
    batches = [int(v) for v in args.batches.split(",")]
    thetas = [float(v) for v in args.thetas.split(",")]

    metadata = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cpu_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": {
            "steps": args.steps,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "drive": args.drive,
            "seed": args.seed,
        },
    }
    print(
        f"device={metadata['device']} ({metadata['device_name']}) "
        f"torch={metadata['torch']} threads={metadata['cpu_threads']}"
    )

    rows: list[dict[str, Any]] = []
    markdown: list[str] = []
    for batch in batches:
        for neurons_count in neurons:
            for degree in degrees:
                print(
                    f"\n### N={neurons_count:,} degree={degree} batch={batch}",
                    flush=True,
                )
                config_rows = run_config(
                    device=device,
                    neurons=neurons_count,
                    degree=degree,
                    batch=batch,
                    steps=args.steps,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    drive=args.drive,
                    thetas=thetas,
                    seed=args.seed,
                    native=args.native,
                )
                rows.extend(config_rows)
                table = format_table(config_rows)
                print(table, flush=True)
                first = config_rows[0]
                markdown.append(
                    f"**N={neurons_count:,}, M={first['synapses']:,}, "
                    f"batch={batch}** (topology {first['storage_mib']} MiB, "
                    f"{first['bytes_per_synapse']} B/synapse)\n\n{table}\n"
                )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"metadata": metadata, "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            f"# Performance run\n\n```text\n{json.dumps(metadata, indent=2)}\n```\n\n"
            + "\n".join(markdown),
            encoding="utf-8",
        )
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
