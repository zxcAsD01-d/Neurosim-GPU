"""Benchmark fused CUDA Matrix inference with reduced sparse weight storage."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
from neurosim_gpu import random_topology
from neurosim_gpu.native import load_native_extension
from neurosim_gpu.native_compartmental import NativeDendriticMatrixRuntime
from precision_compartmental import (
    PRECISIONS, STATE_FIELDS, make_model, sample_order, step, timed, timing_row,
)

OPTIONS = {"fp32": {}, "fp16": {"weight_dtype": torch.float16},
           "fp8": {"weight_format": "fp8"}, "nvfp4": {"weight_format": "nvfp4"}}


@torch.no_grad()
def benchmark(args, extension, neurons, edges, batch):
    graph = random_topology(neurons, edges, device="cuda", seed=23, weight_scale=0.15)
    initial = torch.full((batch, neurons), 0.25, device="cuda")
    model = make_model(neurons, initial.device)
    orders = list(sample_order(args.repeats))

    def runtime(mode):
        return NativeDendriticMatrixRuntime(
            graph, model, initial.clone(), extension=extension, **OPTIONS[mode],
        )

    samples = {mode: [] for mode in PRECISIONS}
    resident = {mode: [] for mode in PRECISIONS}
    metadata = {}
    for order in orders:
        for mode in order:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            runner = runtime(mode)
            step(runner, 0)
            torch.cuda.synchronize()
            resident[mode].append(torch.cuda.memory_allocated() - before)
            for tick in range(1, args.warmup):
                step(runner, tick)
            samples[mode].append(timed(
                lambda tick: step(runner, args.warmup + tick), args.iterations,
            ))
            backend = "native" if mode == "fp32" else f"native-{mode}-weight"
            if runner.last_backend != backend:
                raise RuntimeError(f"Expected {backend}, got {runner.last_backend}")
            metadata[mode] = {"execution": backend, "threads_per_row": runner.threads_per_row,
                              "weight_storage_bytes": runner.weight_storage_bytes}
            del runner

    rows = {mode: timing_row(samples[mode], "tick") for mode in PRECISIONS}
    baseline = rows["fp32"]["median_wall_ms_per_tick"]
    probe_ticks = args.warmup + args.iterations
    for mode, row in rows.items():
        reference, candidate = runtime("fp32"), runtime(mode)
        trajectory = {field: [] for field in STATE_FIELDS}
        for tick in range(probe_ticks):
            step(reference, tick)
            step(candidate, tick)
            for field in STATE_FIELDS:
                error = (getattr(candidate, field) - getattr(reference, field)).abs()
                trajectory[field].append(torch.stack((error.max(), error.mean())))
        summary = {}
        for field, ticks in trajectory.items():
            values = torch.stack(ticks)
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError(f"Non-finite {field} trajectory for native {mode}")
            summary[field] = {"max_abs_error": float(values[:, 0].max()),
                              "mean_abs_error": float(values[:, 1].mean()),
                              "final_max_abs_error": float(values[-1, 0])}
        row.update(metadata[mode])
        row.update({
            "trajectory_error_vs_fp32": summary,
            "trajectory_probe_ticks": probe_ticks,
            "resident_extra_allocated_bytes_after_first_tick_samples": resident[mode],
            "speedup_wall_vs_fp32": baseline / row["median_wall_ms_per_tick"],
        })
    return {"neurons": neurons, "candidate_edges": edges, "edges": graph.n_edges,
            "batch": batch, "sample_order": orders, "precisions": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/precision_native.json"))
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=12)
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.warmup) < 1:
        parser.error("iterations, repeats and warmup must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for fused inference")

    torch.manual_seed(23)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    tracked = (
        "csrc/bindings.cpp", "csrc/runtime.cu", "csrc/compartmental.cu",
        "src/neurosim_gpu/compartmental.py", "src/neurosim_gpu/native.py",
        "src/neurosim_gpu/topology.py", "src/neurosim_gpu/precision.py",
        "src/neurosim_gpu/native_compartmental.py", "benchmarks/precision_compartmental.py",
        "benchmarks/precision_native.py",
    )
    output = {
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "protocol": {
            "seed": 23, "warmup_ticks": args.warmup,
            "iterations_per_repeat": args.iterations, "repeats": args.repeats,
            "timer": "CUDA events and synchronized wall clock",
            "timed": "Fused steady-state ticks after extension loading and weight cache preparation",
            "arithmetic": "Sparse propagation decodes weights to FP32 and updates states in FP32",
            "resident_memory": "CUDA allocator delta after construction and the first tick; baseline includes the graph, model and input",
            "weight_storage_bytes": "Logical inference weight payload; all modes retain the original FP32 topology",
        },
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in tracked
        },
        "workloads": {},
    }
    extension = load_native_extension()
    for name, neurons, edges, batch in (("small", 1024, 16384, 8), ("large", 50000, 1000000, 8)):
        row = benchmark(args, extension, neurons, edges, batch)
        output["workloads"][name] = row
        print(json.dumps({name: {mode: v["median_wall_ms_per_tick"]
                                 for mode, v in row["precisions"].items()}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
