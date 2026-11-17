"""Same-model CUDA compartment benchmark. Build/JIT excluded; raw samples retained.

Run with scripts/native_env.bat python benchmarks/compartmental.py --output ...
No speedup is inferred from another model, dtype, or emission threshold.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

import torch

from neurosim_gpu import (
    Dynamics, DendriticDynamics, DendriticMatrixRuntime, DendriticPELSRuntime,
    NativeDendriticMatrixRuntime, NativeDendriticPELSRuntime, random_topology,
)
from neurosim_gpu.native import load_native_extension


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/compartmental_performance.json"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--large", action="store_true", help="Include 10M and 25M candidate edges")
    parser.add_argument("--profile", action="store_true", help="Count GPU kernels for the first case")
    args = parser.parse_args()
    if args.iterations <= 0 or args.repeats < 3:
        parser.error("iterations must be positive and repeats >= 3")
    torch.set_num_threads(1)
    extension = load_native_extension()
    cases = [(4096, 65536, 1), (50000, 1000000, 1), (50000, 1000000, 8)]
    if args.large:
        cases += [(166000, 10000000, 1), (166000, 25000000, 1)]
    output = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
               "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
               "platform": platform.platform()},
              "protocol": {"warmup_ticks": 30, "iterations": args.iterations,
               "repeats": args.repeats, "timing": "CUDA events and synchronized wall time per block",
               "order": "interleaved, shuffled per repeat (seed 19)",
               "dtype": "float32", "seed": 23, "theta": 0.02,
               "drive": "0.2 + 0.1*sin(tick*0.11)", "dendritic_drive": 0.1,
               "parameters": {"alpha": 0.72, "gain": 0.8, "tau_dendrite": 4,
                              "tau_adaptation": 25, "dt": 1, "coupling": 0.5,
                              "adaptation_strength": 0.2}}, "cases": []}
    root = Path(__file__).resolve().parents[1]
    output["source_sha256"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ("csrc/compartmental.cu", "src/neurosim_gpu/compartmental.py",
                     "src/neurosim_gpu/native_compartmental.py", "benchmarks/compartmental.py")}
    order_generator = random.Random(19)
    with torch.inference_mode():
        for n, edges, batch in cases:
            graph = random_topology(n, edges, device="cuda", seed=23)
            model = DendriticDynamics.from_time_constants(
                Dynamics(torch.full((n,), .72, device="cuda"),
                         torch.full((n,), .8, device="cuda")),
                dt=1, tau_dendrite=4, tau_adaptation=25)
            initial = torch.full((batch,n), .2, device="cuda")
            runtimes = {
                "reference_matrix": DendriticMatrixRuntime(graph, model, initial.clone()),
                "native_matrix": NativeDendriticMatrixRuntime(graph, model, initial.clone(), extension=extension),
                "reference_pels_matrix": DendriticPELSRuntime(graph, model, initial.clone(), theta=.02, propagation="matrix"),
                "native_pels": NativeDendriticPELSRuntime(graph, model, initial.clone(), theta=.02, extension=extension),
            }
            row = {"n": n, "candidate_edges": edges, "edges": graph.n_edges,
                   "batch": batch, "threads_per_row": runtimes["native_matrix"].threads_per_row,
                   "runtimes": {name: {"gpu_ms": [], "wall_ms": []} for name in runtimes}}
            for runtime in runtimes.values():
                for t in range(30):
                    runtime.step(.2+.1*math.sin(t*.11), dendritic_drive=.1)
            torch.cuda.synchronize()
            for repeat in range(args.repeats):
                names = list(runtimes)
                order_generator.shuffle(names)
                for name in names:
                    runtime = runtimes[name]
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    wall = time.perf_counter()
                    start.record()
                    for t in range(args.iterations):
                        tick = 30 + repeat * args.iterations + t
                        runtime.step(.2+.1*math.sin(tick*.11), dendritic_drive=.1)
                    end.record()
                    end.synchronize()
                    row["runtimes"][name]["wall_ms"].append((time.perf_counter()-wall)*1000/args.iterations)
                    row["runtimes"][name]["gpu_ms"].append(start.elapsed_time(end)/args.iterations)
            for name, runtime in runtimes.items():
                data = row["runtimes"][name]
                data["median_wall_ms"] = statistics.median(data["wall_ms"])
                data["median_gpu_ms"] = statistics.median(data["gpu_ms"])
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                runtime.step(.2, dendritic_drive=.1)
                torch.cuda.synchronize()
                data["tick_peak_extra_bytes"] = torch.cuda.max_memory_allocated() - before
            # States have identical input history, including the memory-probe tick.
            for name in ("native_matrix", "native_pels"):
                native = runtimes[name]
                reference = runtimes["reference_pels_matrix" if "pels" in name else "reference_matrix"]
                row["runtimes"][name]["state_max_abs_error"] = float((native.state-reference.state).abs().max())
                row["runtimes"][name]["state_max_abs_error_vs_exact_matrix"] = float((native.state-runtimes["reference_matrix"].state).abs().max())
                for field in ("state", "dendritic_state", "adaptation_state"):
                    # Nonzero-threshold FP32 rounding can change emission timing.
                    # This is an empirical absolute acceptance limit, not a bound.
                    torch.testing.assert_close(getattr(native, field), getattr(reference, field),
                        rtol=0 if "pels" in name else 2e-4,
                        atol=2e-3 if "pels" in name else 2e-5)
            row["pels_empirical_parity_atol"] = 2e-3
            # Outside timing: bind event speed to measured source/edge activity.
            active_sources, active_edges = [], []
            event_runtime = runtimes["native_pels"]
            degree = graph.out_degree
            for t in range(16):
                before_emitted = event_runtime.emitted_state
                event_runtime.step(.2 + .1 * math.sin((30 + args.repeats * args.iterations + 1 + t) * .11),
                                   dendritic_drive=.1)
                active = event_runtime.emitted_state != before_emitted
                active_sources.append(float(active.float().mean()))
                active_edges.append(float((active * degree).sum()) / max(graph.n_edges * batch, 1))
            row["native_pels_activity_probe"] = {
                "ticks": 16, "mean_changed_source_fraction": statistics.mean(active_sources),
                "mean_changed_edge_fraction": statistics.mean(active_edges),
                "note": "post-timing continuation; zero delta at theta=0 is not counted",
            }
            row["matrix_speedup"] = row["runtimes"]["reference_matrix"]["median_wall_ms"] / row["runtimes"]["native_matrix"]["median_wall_ms"]
            row["pels_speedup_vs_reference_matrix_propagation"] = row["runtimes"]["reference_pels_matrix"]["median_wall_ms"] / row["runtimes"]["native_pels"]["median_wall_ms"]
            if args.profile and not output["cases"]:
                from torch.profiler import profile, ProfilerActivity
                row["profile"] = {}
                for name, runtime in runtimes.items():
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        runtime.step(.2, dendritic_drive=.1)
                        torch.cuda.synchronize()
                    trace_path = args.output.parent / ("compartmental_trace_"+name+".json")
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    prof.export_chrome_trace(str(trace_path))
                    trace = json.loads(trace_path.read_text())
                    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
                    row["profile"][name] = {"kernel_count": len(kernels),
                                             "kernel_names": [e["name"] for e in kernels]}
            output["cases"].append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ("runtimes", "profile")}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
            del runtimes, runtime, reference, native, graph, model, initial
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
