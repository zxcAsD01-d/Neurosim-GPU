#!/usr/bin/env python3
"""Where does emitted-event propagation overtake the matrix path?

The reported 3-4.5x event speedup is one point on a curve. This harness sweeps
the emission threshold, measures the resulting active-edge fraction, times both
paths on the same topology, and reports the crossing point -- the activity below
which skipping edges is cheaper than traversing all of them.

    python benchmarks/activity_sweep.py --device cpu
    scripts/native_env.bat python benchmarks/activity_sweep.py --device cuda --native

Activity is measured through the Python reference runtime (which exposes
``collect_stats``); on CUDA the timing uses the native event kernel, because that
is the deployed path. The two share weights, state and threshold, so the
activity fraction transfers.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from neurosim_gpu import Dynamics, HybridPELSRuntime, MatrixRuntime, random_topology


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_step(runtime: Any, drive: float, steps: int, device: torch.device) -> float:
    sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            runtime.step(drive)
    sync(device)
    return (time.perf_counter() - start) * 1000.0 / steps


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
        topology, dynamics, initial.clone(), theta=theta, propagation="event"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--neurons", type=int, default=20_000)
    parser.add_argument("--edges", type=int, default=1_000_000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--thetas", default="0.005,0.01,0.02,0.05,0.1,0.2,0.5")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--drive", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--native", action="store_true", help="time the native CUDA event kernel")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available", file=sys.stderr)
        return 2
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if args.native and device.type != "cuda":
        print("--native requires --device cuda", file=sys.stderr)
        return 2

    topology = random_topology(args.neurons, args.edges, device=device, seed=args.seed)
    dynamics = Dynamics(
        alpha=torch.full((args.neurons,), 0.95, device=device),
        gain=torch.full((args.neurons,), 0.8, device=device),
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    initial = torch.rand(args.batch, args.neurons, device=device, generator=generator)

    matrix = MatrixRuntime(topology, dynamics, initial.clone())
    with torch.inference_mode():
        for _ in range(args.warmup):
            matrix.step(args.drive)
    matrix_ms = timed_step(matrix, args.drive, args.steps, device)

    native_event = None
    extension = None
    if args.native:
        from neurosim_gpu.native import NativeEventPELSRuntime, load_native_extension

        extension = load_native_extension()

    def build_event(theta: float):
        if args.native:
            return NativeEventPELSRuntime(
                topology,
                dynamics,
                initial.clone(),
                theta=theta,
                extension=extension,
            )
        return HybridPELSRuntime(
            topology, dynamics, initial.clone(), theta=theta, propagation="event"
        )

    thetas = [float(t) for t in args.thetas.split(",")]
    rows: list[dict[str, Any]] = []
    print(
        f"device={device} N={args.neurons:,} M={topology.n_edges:,} B={args.batch} "
        f"matrix={matrix_ms:.3f} ms/path={'native' if args.native else 'reference'}"
    )
    print("theta   active-edges   event ms   speedup (matrix/event)")

    crossing: float | None = None
    for theta in thetas:
        profile = activity_profile(
            topology,
            dynamics,
            initial,
            theta=theta,
            steps=args.steps,
            warmup=args.warmup,
            drive=args.drive,
        )
        runtime = build_event(theta)
        with torch.inference_mode():
            for _ in range(args.warmup):
                runtime.step(args.drive)
        event_ms = timed_step(runtime, args.drive, args.steps, device)
        speedup = matrix_ms / event_ms
        if speedup >= 1.0 and crossing is None:
            crossing = theta
        rows.append(
            {
                "theta": theta,
                "active_edge_fraction": profile["active_edge_fraction"],
                "active_neuron_fraction": profile["active_neuron_fraction"],
                "event_ms": event_ms,
                "matrix_ms": matrix_ms,
                "speedup": speedup,
            }
        )
        print(
            f"{theta:5.3f}   {profile['active_edge_fraction'] * 100:9.1f}%   "
            f"{event_ms:9.3f}   {speedup:6.2f}x"
        )

    crossing_fraction = None
    if crossing is not None:
        for row in rows:
            if row["theta"] == crossing:
                crossing_fraction = row["active_edge_fraction"]
                break
        print(
            f"\ncrossing: theta={crossing:g} "
            f"(about {crossing_fraction * 100:.1f}% active edges)"
        )
    else:
        print(
            "\ncrossing: none in this sweep -- the event path never beat the matrix "
            "path on this device/implementation"
        )
        if device.type == "cpu" and not args.native:
            print(
                "note: the reference event path performs a Python-level scatter per "
                "tick; this is expected on CPU and is why the native kernel exists."
            )

    if args.json:
        payload = {
            "metadata": {
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else platform.processor()
                ),
                "torch": torch.__version__,
                "neurons": args.neurons,
                "synapses": topology.n_edges,
                "batch": args.batch,
                "steps": args.steps,
                "warmup": args.warmup,
                "seed": args.seed,
                "event_impl": "native" if args.native else "reference",
                "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "matrix_ms": matrix_ms,
            "rows": rows,
            "crossing_theta": crossing,
            "crossing_active_edge_fraction": crossing_fraction,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
