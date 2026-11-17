"""Reproducible response/ablation probe; no task-accuracy or throughput claim."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform

import torch

from neurosim_gpu import (
    DendriticDynamics, DendriticMatrixRuntime, Dynamics, MatrixRuntime,
    SparseTopology,
)


@torch.no_grad()
def run(device: str) -> dict:
    n, batch, ticks = 32, 2, 400
    dtype = torch.float64
    # Isolated cells distinguish intrinsic adaptation from recurrent inhibition.
    empty = torch.empty(0, device=device, dtype=torch.int64)
    graph = SparseTopology.from_edges(n, empty, empty,
                                     torch.empty(0, device=device, dtype=dtype))
    initial = torch.zeros(batch, n, device=device, dtype=dtype)
    soma = Dynamics(torch.full((n,), 0.7, device=device, dtype=dtype),
                    torch.full((n,), 0.8, device=device, dtype=dtype))
    model = DendriticDynamics.from_time_constants(
        soma, dt=1, tau_dendrite=torch.linspace(2, 8, n),
        tau_adaptation=torch.linspace(15, 30, n), coupling=0.7,
        adaptation_strength=1.0,
    )
    zero = torch.zeros(n, device=device, dtype=dtype)
    variants = {
        "point": MatrixRuntime(graph, soma, initial.clone()),
        "disabled": DendriticMatrixRuntime(
            graph, replace(model, coupling=zero, adaptation_strength=zero), initial.clone()),
        "dendrite_only": DendriticMatrixRuntime(
            graph, replace(model, adaptation_strength=zero), initial.clone()),
        "adaptation_only": DendriticMatrixRuntime(
            graph, replace(model, coupling=zero), initial.clone()),
        "combined": DendriticMatrixRuntime(graph, model, initial.clone()),
    }
    records = {name: [] for name in variants}
    max_disabled_error = 0.0
    for t in range(ticks):
        drive = 0.6 if 20 <= t < 200 else 0.0
        dendritic_drive = 1.0 if 280 <= t < 290 else 0.0
        for name, runtime in variants.items():
            if isinstance(runtime, DendriticMatrixRuntime):
                state = runtime.step(drive, dendritic_drive=dendritic_drive)
                d, a = runtime.dendritic_state.mean(), runtime.adaptation_state.mean()
            else:
                state = runtime.step(drive)
                d, a = 0.0, 0.0
            records[name].append({"tick": t, "drive": drive,
                                  "dendritic_drive": dendritic_drive,
                                  "soma": float(state.mean()),
                                  "dendrite": float(d), "adaptation": float(a)})
        max_disabled_error = max(max_disabled_error, float(
            (variants["point"].state - variants["disabled"].state).abs().max()))
    trace = records["combined"]
    peak = max(row["soma"] for row in trace[20:200])
    steady = trace[199]["soma"]
    return {
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": device, "hardware": torch.cuda.get_device_name()
                        if device == "cuda" else platform.processor(),
                        "cuda": torch.version.cuda, "dtype": str(dtype)},
        "config": {"n_neurons": n, "n_edges": 0, "batch": batch, "ticks": ticks,
                   "dt": 1, "time_unit": "arbitrary tick units", "alpha": 0.7,
                   "gain": 0.8, "tau_dendrite_range": [2, 8],
                   "tau_adaptation_range": [15, 30], "coupling": 0.7,
                   "adaptation_strength": 1.0},
        "metrics": {"disabled_max_abs_error": max_disabled_error,
                    "combined_step_peak": peak, "combined_step_end": steady,
                    "adaptation_peak_to_end_drop": peak - steady,
                    "post_pulse_soma_tick_295": trace[295]["soma"]},
        "limitations": "Synthetic intrinsic response only; no biological fit, task accuracy or timing claim.",
        "traces": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path,
                        default=Path("results/compartmental_probe_cpu.json"))
    parser.add_argument("--plot", type=Path, help="Optional PNG; requires matplotlib")
    args = parser.parse_args()
    result = run(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        for name in ("point", "dendrite_only", "adaptation_only", "combined"):
            axes[0].plot([r["soma"] for r in result["traces"][name]], label=name)
        axes[0].set(ylabel="Somatic activity [0, 1]",
                    title="Intrinsic response ablations (synthetic, uncalibrated)")
        axes[0].legend(ncol=2)
        for name in ("dendrite", "adaptation"):
            axes[1].plot([r[name] for r in result["traces"]["combined"]], label=name)
        axes[1].set(xlabel="Tick (arbitrary time unit)", ylabel="Compartment state")
        axes[1].legend()
        for ax in axes:
            ax.axvspan(20, 199, color="gray", alpha=0.1)
            ax.axvspan(280, 289, color="orange", alpha=0.15)
            ax.grid(alpha=0.2)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
