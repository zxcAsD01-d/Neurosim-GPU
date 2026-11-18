"""Benchmark inference and rollout-backward precision modes for the three-state core.

FP8 and NVFP4 are Q/DQ simulations over FP32 CSR SpMM, not Tensor Core runs.
Use --output results/precision_compartmental.json to retain raw samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from neurosim_gpu import (
    DendriticDynamics, DendriticMatrixRuntime, DendriticPELSRuntime, Dynamics,
    random_topology,
)

PRECISIONS = ("fp32", "fp16", "fp8", "nvfp4")
ARCHITECTURES = ("matrix", "pels")
STATE_FIELDS = ("state", "dendritic_state", "adaptation_state")
ORDER_SEED = 19
LEARNING_RATE = 0.01


def make_model(n: int, device: str, trainable: bool = False) -> DendriticDynamics:
    def parameter(value):
        return torch.full((n,), value, device=device, requires_grad=trainable)

    return DendriticDynamics(
        Dynamics(parameter(0.72), parameter(0.8)),
        parameter(math.exp(-1.0 / 4.0)), parameter(math.exp(-1.0 / 20.0)),
        parameter(0.45), parameter(0.15),
    )


def make_runtime(architecture, graph, model, initial, precision):
    if architecture == "matrix":
        return DendriticMatrixRuntime(graph, model, initial.clone(), precision=precision)
    return DendriticPELSRuntime(
        graph, model, initial.clone(), theta=0.0, propagation="matrix",
        precision=precision,
    )


def step(runtime, tick: int) -> None:
    runtime.step(0.16 + 0.04 * ((tick % 7) / 6.0), dendritic_drive=0.025)


def sample_order(repeats):
    generator = random.Random(ORDER_SEED)
    for _ in range(repeats):
        order = list(PRECISIONS)
        generator.shuffle(order)
        yield order


def timing_row(samples, unit):
    gpu = [sample[0] for sample in samples]
    wall = [sample[1] for sample in samples]
    peak = [sample[2] for sample in samples]
    return {
        f"gpu_ms_per_{unit}_samples": gpu,
        f"wall_ms_per_{unit}_samples": wall,
        f"median_gpu_ms_per_{unit}": statistics.median(gpu),
        f"median_wall_ms_per_{unit}": statistics.median(wall),
        "peak_extra_allocated_bytes_samples": peak,
        "median_peak_extra_allocated_bytes": statistics.median(peak),
    }


def timed(operation, iterations):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    start_event.record()
    for index in range(iterations):
        operation(index)
    end_event.record()
    end_event.synchronize()
    wall = (time.perf_counter() - wall_start) * 1000.0 / iterations
    peak = max(0, torch.cuda.max_memory_allocated() - before)
    return start_event.elapsed_time(end_event) / iterations, wall, peak


@torch.inference_mode()
def inference_benchmark(args, graph, initial, architecture):
    samples = {precision: [] for precision in PRECISIONS}
    orders = list(sample_order(args.repeats))
    execution = {}
    model = make_model(args.neurons, initial.device)
    for order in orders:
        for precision in order:
            runtime = make_runtime(architecture, graph, model, initial, precision)
            for tick in range(args.warmup):
                step(runtime, tick)
            samples[precision].append(timed(
                lambda tick: step(runtime, args.warmup + tick), args.iterations,
            ))
            execution[precision] = runtime.precision_execution
            del runtime

    rows = {precision: timing_row(samples[precision], "tick") for precision in PRECISIONS}
    # Keep the two runtimes in lockstep outside timing. Retain device-side
    # summaries per tick rather than storing the full [T,B,N] trajectories.
    trajectory_ticks = args.warmup + args.iterations
    baseline_wall = rows["fp32"]["median_wall_ms_per_tick"]
    for precision, row in rows.items():
        reference = make_runtime(architecture, graph, model, initial, "fp32")
        candidate = make_runtime(architecture, graph, model, initial, precision)
        trajectory = {field: [] for field in STATE_FIELDS}
        for tick in range(trajectory_ticks):
            step(reference, tick)
            step(candidate, tick)
            for field in STATE_FIELDS:
                actual = getattr(candidate, field)
                expected = getattr(reference, field)
                error = (actual - expected).abs()
                trajectory[field].append(torch.stack((error.max(), error.mean())))
        summaries = {}
        for field, ticks in trajectory.items():
            values = torch.stack(ticks)
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError(f"Non-finite {field} trajectory for {architecture}/{precision}")
            summaries[field] = {
                "max_abs_error": float(values[:, 0].max()),
                "mean_abs_error": float(values[:, 1].mean()),
                "final_max_abs_error": float(values[-1, 0]),
            }
        row.update({
            "execution": execution[precision],
            "trajectory_error_vs_fp32": summaries,
            "trajectory_probe_ticks": trajectory_ticks,
            "final_state_max_abs_error_vs_fp32": summaries["state"]["final_max_abs_error"],
            "speedup_wall_vs_fp32": baseline_wall / row["median_wall_ms_per_tick"],
            "ticks_per_second_wall": 1000.0 / row["median_wall_ms_per_tick"],
        })
    return rows, orders


def prepare_training(args, graph, device):
    model = make_model(args.neurons, device, trainable=True)
    parameters = {
        "sparse_weight": graph.weight.detach().clone().requires_grad_(True),
        "alpha": model.soma.alpha, "gain": model.soma.gain,
        "dendrite_decay": model.dendrite_decay,
        "adaptation_decay": model.adaptation_decay,
        "coupling": model.coupling, "adaptation_strength": model.adaptation_strength,
    }
    optimizer = torch.optim.SGD(parameters.values(), lr=LEARNING_RATE)
    return model, parameters, optimizer


def _train_step(args, architecture, graph, in_order, model, parameters,
                initial, precision, optimizer, tick):
    optimizer.zero_grad(set_to_none=True)
    weights = parameters["sparse_weight"]
    # Reuse fixed graph indices. A fresh differentiable gather per optimizer
    # step links the CSR weights to the FP32 leaf the optimizer updates.
    train_graph = replace(graph, weight=weights, in_weight=weights.index_select(0, in_order))
    runtime = make_runtime(architecture, train_graph, model, initial, precision)
    for index in range(args.train_rollout):
        step(runtime, tick + index)
    # Average across samples so each neuron's update stays representable as N
    # grows while keeping the objective invariant to batch replication.
    loss = runtime.state.square().mean(dim=0).sum()
    loss.backward()
    optimizer.step()
    return loss.detach(), runtime.precision_execution


def training_benchmark(args, graph, initial, architecture):
    samples = {precision: [] for precision in PRECISIONS}
    orders = list(sample_order(args.repeats))
    in_order = graph.out_weight_index.argsort()
    for order in orders:
        for precision in order:
            model, parameters, optimizer = prepare_training(args, graph, initial.device)

            def optimize(iteration):
                return _train_step(args, architecture, graph, in_order, model, parameters,
                                   initial, precision, optimizer, iteration * args.train_rollout)

            for warm in range(args.train_warmup):
                optimize(warm)
            samples[precision].append(timed(
                lambda index: optimize(args.train_warmup + index), args.train_steps,
            ))
            del model, parameters, optimizer

    unit = "rollout_backward_step"
    rows = {precision: timing_row(samples[precision], unit) for precision in PRECISIONS}
    probe_gradients = {}
    baseline_wall = rows["fp32"][f"median_wall_ms_per_{unit}"]
    for precision, row in rows.items():
        model, parameters, optimizer = prepare_training(args, graph, initial.device)
        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        loss, execution = _train_step(
            args, architecture, graph, in_order, model, parameters,
            initial, precision, optimizer, tick=0,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss for {architecture}/{precision}")
        gradients = {}
        probe = {}
        for name, parameter in parameters.items():
            grad = parameter.grad
            if grad is None or not bool(torch.isfinite(grad).all()) or not bool(grad.any()):
                raise RuntimeError(f"Missing, zero or non-finite {name} gradient for {architecture}/{precision}")
            update = (parameter.detach() - before[name]).abs().max()
            if not bool(torch.isfinite(update) & (update > 0)):
                raise RuntimeError(f"Zero or non-finite {name} update for {architecture}/{precision}")
            gradients[name] = grad.detach()
            probe[name] = {
                "nonzero_count": int(torch.count_nonzero(grad)),
                "gradient_max_abs": float(grad.abs().max()),
                "parameter_update_max_abs": float(update),
            }
        probe_gradients[precision] = gradients
        for name, grad in gradients.items():
            error = (grad - probe_gradients["fp32"][name]).abs()
            probe[name]["gradient_max_abs_error_vs_fp32"] = float(error.max())
            probe[name]["gradient_mean_abs_error_vs_fp32"] = float(error.mean())
        row.update({
            "execution": execution,
            "probe_loss": float(loss),
            "probe_parameters": probe,
            "probe_gradient_max_abs_error_vs_fp32": max(
                stats["gradient_max_abs_error_vs_fp32"] for stats in probe.values()
            ),
            "speedup_wall_vs_fp32": baseline_wall / row[f"median_wall_ms_per_{unit}"],
            "rollout_backward_steps_per_second_wall": 1000.0 / row[f"median_wall_ms_per_{unit}"],
        })
    return rows, orders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/precision_compartmental.json"))
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--edges", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--train-warmup", type=int, default=1)
    parser.add_argument("--train-rollout", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this precision benchmark")
    if min(args.neurons, args.batch, args.iterations, args.repeats, args.train_steps, args.train_rollout) <= 0:
        parser.error("sizes and iteration counts must be positive")
    if not 0 < args.edges < 2**31:
        parser.error("edges must be in (0, 2^31) for the trainable sparse graph")
    if args.train_rollout < 2:
        parser.error("training requires at least two ticks for gradients through adaptation")
    if args.warmup < 0 or args.train_warmup < 0:
        parser.error("warmup counts must be nonnegative")

    torch.manual_seed(23)
    torch.set_num_threads(1)
    graph = random_topology(args.neurons, args.edges, device="cuda", seed=23, weight_scale=0.15)
    initial = torch.full((args.batch, args.neurons), 0.25, device="cuda")
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    root = Path(__file__).resolve().parents[1]
    tracked = (
        "src/neurosim_gpu/precision.py", "src/neurosim_gpu/compartmental.py",
        "src/neurosim_gpu/runtime.py", "src/neurosim_gpu/topology.py",
        "benchmarks/precision_compartmental.py",
    )
    output = {
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "gpu": prop.name,
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "protocol": {
            "seed": 23, "neurons": args.neurons, "candidate_edges": args.edges,
            "edges": graph.n_edges, "initial_graph_weight_scale": 0.15,
            "batch": args.batch, "warmup_ticks": args.warmup,
            "inference_iterations_per_repeat": args.iterations,
            "training_warmup_steps": args.train_warmup,
            "training_steps_per_repeat": args.train_steps,
            "training_rollout_ticks": args.train_rollout, "repeats": args.repeats,
            "timer": "CUDA events plus synchronized wall clock",
            "sample_order": f"Seed {ORDER_SEED} shuffles all precision modes within each repeat",
            "inference_timed": "Steady-state runtime tick after warmup using a fixed graph",
            "training_timed": "Gradient clearing, differentiable CSR value reorder, runtime construction, rollout, loss backward, and SGD",
            "preprocess_untimed": "Graph generation, edge sorting, coalescing, and CSR index mapping",
            "training_optimizer": {"type": "SGD", "lr": LEARNING_RATE, "master_dtype": "float32"},
            "training_objective": "Sum of squared final somatic states, averaged over the batch",
            "inference_error_probe": "Fresh same-input rollout sampled at every tick, outside timing",
            "gradient_probe": "Fresh optimizer step from identical parameters, outside timing",
            "pels_theta": 0.0,
            "quantization": {
                "fp16": "FP16 CSR values and source; FP32 three-state updates",
                "fp8": "E4M3 per-tensor scaled Q/DQ; FP32 CSR SpMM and state updates; STE gradients",
                "nvfp4": "E2M1 values in 16-element blocks, E4M3 block scales, FP32 global scale; FP32 CSR SpMM/state; STE gradients",
            },
        },
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in tracked
        },
        "architectures": {},
    }
    for architecture in ARCHITECTURES:
        torch.cuda.synchronize()
        inference, inference_order = inference_benchmark(args, graph, initial, architecture)
        training, training_order = training_benchmark(args, graph, initial, architecture)
        output["architectures"][architecture] = {
            "inference": inference, "training": training,
            "inference_sample_order": inference_order,
            "training_sample_order": training_order,
        }
        summary = {
            name: {
                "infer_wall_ms": round(inference[name]["median_wall_ms_per_tick"], 6),
                "infer_speedup": round(inference[name]["speedup_wall_vs_fp32"], 4),
                "train_wall_ms": round(training[name]["median_wall_ms_per_rollout_backward_step"], 6),
                "train_speedup": round(training[name]["speedup_wall_vs_fp32"], 4),
                "state_error": inference[name]["final_state_max_abs_error_vs_fp32"],
                "gradient_error": training[name]["probe_gradient_max_abs_error_vs_fp32"],
            }
            for name in PRECISIONS
        }
        print(json.dumps({architecture: summary}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
