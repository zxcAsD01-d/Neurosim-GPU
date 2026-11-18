"""Mainline soma/dendrite/adaptation simulation: python -m neurosim_gpu.simulate.

Backend and propagation are explicit: this runner does not guess a fastest path,
silently change models, or hide native build failures. It is not a benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .compartmental import DendriticDynamics, DendriticMatrixRuntime, DendriticPELSRuntime
from .dynamics import Dynamics
from .native_compartmental import NativeDendriticMatrixRuntime, NativeDendriticPELSRuntime
from .checkpoint import load_runtime_checkpoint, save_runtime_checkpoint
from .precision import PRECISIONS
from .topology import random_topology


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--backend", choices=("reference", "native"), default="reference")
    result.add_argument("--precision", choices=PRECISIONS, default="fp32",
                        help="reference propagation mode, or native matrix weight storage")
    result.add_argument("--propagation", choices=("matrix", "pels"), default="matrix")
    result.add_argument("--neurons", type=int, default=1000)
    result.add_argument("--edges", type=int, default=20_000, help="candidate edges before coalescing")
    result.add_argument("--batch", type=int, default=1)
    result.add_argument("--steps", type=int, default=100, help="ticks to execute in this invocation")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--theta", type=float, default=0.0, help="PELS emission threshold; >0 is approximate")
    result.add_argument("--drive", type=float, default=0.2)
    result.add_argument("--dendritic-drive", type=float, default=0.1)
    result.add_argument("--dt", type=float, default=1.0)
    result.add_argument("--tau-dendrite", type=float, default=4.0)
    result.add_argument("--tau-adaptation", type=float, default=25.0)
    result.add_argument("--coupling", type=float, default=0.5)
    result.add_argument("--adaptation-strength", type=float, default=0.2)
    result.add_argument("--resume", type=Path, help="restore all runtime states; same graph/model/path required")
    result.add_argument("--checkpoint", type=Path, help="save all runtime states after this run")
    result.add_argument("--output", type=Path, help="write the JSON summary")
    return result


def build_runtime(args: argparse.Namespace):
    if args.neurons <= 0 or args.batch <= 0 or args.steps <= 0 or args.edges < 0:
        raise ValueError("neurons, batch and steps must be positive; edges must be non-negative")
    if not all(math.isfinite(value) for value in (args.theta, args.drive, args.dendritic_drive)):
        raise ValueError("theta and external drives must be finite")
    if args.theta < 0:
        raise ValueError("theta must be non-negative")
    if args.propagation == "matrix" and args.theta != 0:
        raise ValueError("theta applies only to --propagation pels")
    if args.backend == "native" and args.device != "cuda":
        raise ValueError("--backend native requires --device cuda and a CUDA build toolchain")
    if args.backend == "native" and args.propagation == "pels" and args.precision != "fp32":
        raise ValueError("native PELS supports FP32 only; use native matrix or reference PELS")
    if args.precision == "fp16" and args.device != "cuda":
        raise ValueError("--precision fp16 requires --device cuda")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu --backend reference")
    graph = random_topology(args.neurons, args.edges, device=args.device, seed=args.seed)
    soma = Dynamics(torch.full((args.neurons,), 0.72, device=args.device),
                    torch.full((args.neurons,), 0.8, device=args.device))
    model = DendriticDynamics.from_time_constants(
        soma, dt=args.dt, tau_dendrite=args.tau_dendrite,
        tau_adaptation=args.tau_adaptation, coupling=args.coupling,
        adaptation_strength=args.adaptation_strength,
    )
    initial = torch.zeros(args.batch, args.neurons, device=args.device)
    if args.propagation == "matrix":
        if args.backend == "native":
            return NativeDendriticMatrixRuntime(
                graph, model, initial,
                weight_dtype=torch.float16 if args.precision == "fp16" else torch.float32,
                weight_format=args.precision if args.precision in ("fp8", "nvfp4") else None,
            )
        return DendriticMatrixRuntime(graph, model, initial, precision=args.precision)
    if args.backend == "native":
        return NativeDendriticPELSRuntime(graph, model, initial, theta=args.theta)
    # Reference PELS uses SpMM (FP32 deltas or low-precision emitted snapshots).
    return DendriticPELSRuntime(graph, model, initial, theta=args.theta,
                               propagation="matrix", precision=args.precision)


def run(args: argparse.Namespace) -> dict:
    if args.output is not None:
        output = args.output.resolve()
        for option in ("checkpoint", "resume"):
            path = getattr(args, option)
            if path is None:
                continue
            same_file = output == path.resolve()
            if not same_file and output.exists() and path.exists():
                same_file = output.samefile(path)
            if same_file:
                raise ValueError(f"--output must use a different file from --{option}")
    runtime = build_runtime(args)
    if args.resume is not None:
        load_runtime_checkpoint(args.resume, runtime)
    with torch.inference_mode():
        for _ in range(args.steps):
            runtime.step(args.drive, dendritic_drive=args.dendritic_drive)
    # Diagnostics and host synchronization occur after the rollout, not per tick.
    states = {}
    for name in ("state", "dendritic_state", "adaptation_state"):
        value = getattr(runtime, name)
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"non-finite final {name}")
        states[name] = {"shape": list(value.shape), "mean": float(value.mean()),
                        "min": float(value.min()), "max": float(value.max())}
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_runtime_checkpoint(runtime, args.checkpoint)
    result = {
        "model": "soma-dendrite-adaptation", "runtime": type(runtime).__name__,
        "backend": getattr(runtime, "last_backend", "reference"),
        "device": str(runtime.state.device), "dtype": str(runtime.state.dtype),
        "precision": args.precision, "precision_execution": runtime.precision_execution,
        "torch": torch.__version__, "propagation": args.propagation,
        "neurons": runtime.topology.n_neurons, "actual_edges": runtime.topology.n_edges,
        "steps_this_run": args.steps, "theta": args.theta,
        "approximate": args.precision != "fp32" or (args.propagation == "pels" and args.theta > 0),
        "resumed": args.resume is not None,
        "config": {name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(args).items()},
        "states": states,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    try:
        result = run(args)
    except ValueError as error:
        cli.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
