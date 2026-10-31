#!/usr/bin/env python3
"""Mechanism probe for the autonomous (no-input) cortical sheet.

Why does the model stop producing activity when there is no next-token demand?
This script measures:

* the linear stability of the zero/working point (Jacobian spectrum);
* the zero-drive trajectory and its fixed point;
* what injected noise does to dimensionality and spectrum.

Human EEG comparison is qualitative; no human recording is used and the tick
time is arbitrary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neurosim_gpu.cortex.brain import CorticalCore  # noqa: E402
from neurosim_gpu.cortex.config import ExperimentConfig  # noqa: E402

BACKGROUND = "#0b0f14"
TEXT = "#f1f3f5"
MUTED = "#adb5bd"
BANDS = [
    ("delta", 0.5, 4.0, "#4c6ef5"),
    ("theta", 4.0, 8.0, "#22b8cf"),
    ("alpha", 8.0, 13.0, "#51cf66"),
    ("beta", 13.0, 30.0, "#fcc419"),
    ("gamma", 30.0, 45.0, "#ff6b6b"),
]


def run(
    core: CorticalCore,
    *,
    noise: float,
    ticks: int,
    initial: torch.Tensor | None = None,
) -> tuple[np.ndarray, torch.Tensor]:
    core.trace = []
    state = (
        torch.zeros(1, core.n_neurons)
        if initial is None
        else initial.clone()
    )
    for _ in range(ticks):
        drive = (
            torch.randn(1, 1, core.d_model) * noise
            if noise > 0
            else torch.zeros(1, 1, core.d_model)
        )
        _, state, _ = core(drive, state=state)
    trace = torch.stack(core.trace).float().numpy()
    core.trace = None
    return trace, state.detach()


def noise_analysis(
    trace: np.ndarray, fs: float = 100.0
) -> dict[str, float]:
    mean = trace.mean(axis=1)
    mean = mean - mean.mean()
    std = trace.std(axis=1)
    freq, power = welch(
        mean,
        fs=fs,
        nperseg=min(256, len(mean)),
        noverlap=min(128, len(mean) // 2),
        detrend="constant",
    )
    valid = freq > 0
    slope = float(
        np.polyfit(np.log(freq[valid]), np.log(power[valid]), 1)[0]
    )
    total = np.trapezoid(power, freq)
    bands = {}
    for name, low, high, _ in BANDS:
        mask = (freq >= low) & (freq < high)
        bands[name] = float(
            np.trapezoid(power[mask], freq[mask]) / total
        )
    centered = trace - trace.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered[::2], compute_uv=False)
    energy = singular**2
    probability = energy / energy.sum()
    rank = float(
        np.exp(-(probability * np.log(probability + 1e-12)).sum())
    )
    return {
        "mean": float(trace.mean()),
        "std": float(trace.std()),
        "time_std": float(mean.std()),
        "rank": rank,
        "psd_slope": slope,
        "bands": bands,
        "frequency": freq,
        "power": power,
        "trace_std": std,
    }


def style(axis) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.tick_params(colors=MUTED, labelsize=7)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/cortex_stage3/final.pt")
    parser.add_argument(
        "--output", default="llm/results/cortex/autonomous_mechanism.png"
    )
    args = parser.parse_args()

    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    config = ExperimentConfig.from_dict(payload["config"])
    core = CorticalCore(config.brain, d_model=1024)
    state = {
        key.removeprefix("brain."): value
        for key, value in payload["model"].items()
        if key.startswith("brain.")
    }
    core.load_state_dict(state, strict=False)
    core.eval()
    core.refresh_spatial_mask()

    weights = (core.recurrent.detach() * core.edge_mask).numpy()
    alpha = core.alpha.numpy()
    gain = core.effective_gain().detach().numpy()
    jacobian = np.diag(alpha) + ((1.0 - alpha) * gain)[:, None] * weights
    eigenvalues = np.linalg.eigvals(jacobian)
    spectral_radius = float(np.max(np.abs(eigenvalues)))

    zero_trace, fixed = run(
        core, noise=0.0, ticks=2000, initial=torch.rand(1, core.n_neurons)
    )
    zero_norm = np.linalg.norm(zero_trace, axis=1)
    zero_stats = noise_analysis(zero_trace)
    levels = [0.0, 0.001, 0.005, 0.01]
    results = {}
    traces = {}
    for noise in levels:
        trace, _ = run(core, noise=noise, ticks=512)
        stats = noise_analysis(trace)
        results[str(noise)] = {
            key: value
            for key, value in stats.items()
            if key not in {"frequency", "power", "trace_std"}
        }
        traces[str(noise)] = stats
        print(
            f"noise={noise}: mean={stats['mean']:.4f} "
            f"rank={stats['rank']:.1f} slope={stats['psd_slope']:.2f}"
        )

    figure = plt.figure(figsize=(18, 10), dpi=170)
    figure.patch.set_facecolor(BACKGROUND)
    grid = figure.add_gridspec(
        2, 3, left=0.04, right=0.97, top=0.88, bottom=0.10,
        wspace=0.22, hspace=0.30,
    )

    axis = figure.add_subplot(grid[0, 0])
    style(axis)
    axis.semilogy(np.arange(len(zero_norm)), zero_norm, color="#4dd0e1", lw=1.2)
    axis.axhline(
        float(zero_norm[-1]), color="#ffd43b", lw=0.8, ls="--",
        label=f"fixed point norm {zero_norm[-1]:.2f}",
    )
    axis.set_xlabel("tick", color=MUTED, fontsize=8)
    axis.set_ylabel("||state||", color=MUTED, fontsize=8)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=7)
    axis.set_title(
        "zero drive: random state → fixed point", color=TEXT, fontsize=11, pad=8
    )

    axis = figure.add_subplot(grid[0, 1])
    style(axis)
    for name, low, high, color in BANDS:
        axis.axvspan(low, high, color=color, alpha=0.07)
    axis.axvline(10.0, color="#51cf66", lw=0.7, ls="--", alpha=0.5)
    for noise in levels:
        stats = traces[str(noise)]
        axis.loglog(
            stats["frequency"][1:],
            stats["power"][1:] + 1e-30,
            lw=1.1,
            label=f"injected noise {noise}",
        )
    axis.set_xlim(0.5, 50)
    axis.set_xlabel("Hz (reference 1 tick = 10 ms)", color=MUTED, fontsize=8)
    axis.set_ylabel("PSD", color=MUTED, fontsize=8)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=7)
    axis.set_title(
        "noise restores activity, not alpha", color=TEXT, fontsize=11, pad=8
    )

    axis = figure.add_subplot(grid[0, 2])
    style(axis)
    axis.scatter(
        eigenvalues.real,
        eigenvalues.imag,
        s=5,
        color="#4dd0e1",
        alpha=0.6,
        linewidths=0,
    )
    theta = np.linspace(0, 2 * math.pi, 300)
    axis.plot(np.cos(theta), np.sin(theta), color="#ffd43b", lw=1.0, ls="--")
    axis.axhline(0, color=MUTED, lw=0.4)
    axis.axvline(0, color=MUTED, lw=0.4)
    axis.set_aspect("equal")
    axis.set_xlabel("Re λ", color=MUTED, fontsize=8)
    axis.set_ylabel("Im λ", color=MUTED, fontsize=8)
    axis.set_title(
        f"Jacobian at zero: ρ = {spectral_radius:.4f}",
        color=TEXT,
        fontsize=11,
        pad=8,
    )

    axis = figure.add_subplot(grid[1, 0])
    style(axis)
    noise_values = [float(value) for value in results]
    ranks = [results[str(value)]["rank"] for value in noise_values]
    stds = [results[str(value)]["std"] for value in noise_values]
    axis.plot(noise_values, ranks, marker="o", color="#4dd0e1", label="rank")
    axis.set_xlabel("injected drive noise", color=MUTED, fontsize=8)
    axis.set_ylabel("effective rank", color=MUTED, fontsize=8)
    axis2 = axis.twinx()
    axis2.plot(noise_values, stds, marker="s", color="#ffa94d", label="std")
    axis2.set_ylabel("state std", color=MUTED, fontsize=8)
    axis2.tick_params(colors=MUTED, labelsize=7)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=7, loc="center right")
    axis.set_title(
        "noise → dimensionality (not oscillation)", color=TEXT, fontsize=11, pad=8
    )

    axis = figure.add_subplot(grid[1, 1])
    style(axis)
    noise_stats = traces["0.005"]
    positions = np.arange(len(BANDS))
    axis.bar(
        positions,
        [noise_stats["bands"][name] for name, *_ in BANDS],
        color=[band[3] for band in BANDS],
        alpha=0.85,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([band[0] for band in BANDS], color=MUTED, fontsize=8)
    axis.set_ylabel("relative power", color=MUTED, fontsize=8)
    axis.set_title(
        "noise-driven band power (0.005)", color=TEXT, fontsize=11, pad=8
    )

    axis = figure.add_subplot(grid[1, 2])
    style(axis)
    axis.axis("off")
    lines = [
        f"Jacobian ρ              {spectral_radius:.4f}",
        f"zero-drive fixed norm   {zero_norm[-1]:.3f}",
        f"zero-drive rank         {zero_stats['rank']:.2f}",
        f"noise 0.005 rank        {traces['0.005']['rank']:.1f}",
        f"noise 0.005 slope       {traces['0.005']['psd_slope']:.2f}",
        f"weights > 0             {(weights > 0).mean():.1%}",
        f"alpha mean              {alpha.mean():.3f}",
        f"gain mean               {gain.mean():.2f}",
    ]
    axis.text(
        0.0,
        1.0,
        "autonomous mechanism\n\n"
        + "\n".join(lines)
        + "\n\n"
        "Why no spontaneous EEG:\n"
        "• no noise at eval (training noise only)\n"
        "• zero drive has a stable fixed point\n"
        "• no delayed inhibition / pacemaker\n"
        "• trained only for next-token prediction\n"
        "• clamp/tanh make the map dissipative\n\n"
        "Human EEG has spontaneous activity,\n"
        "noise, delayed loops and alpha\n"
        "generators; this model has none.",
        color=TEXT,
        fontsize=8.5,
        va="top",
        family="monospace",
    )
    axis.set_title("mechanism summary", color=TEXT, fontsize=11, pad=8)

    figure.suptitle(
        "Why the cortical sheet goes quiet without input · stage3",
        color=TEXT,
        fontsize=16,
        x=0.04,
        ha="left",
        y=0.965,
    )
    figure.text(
        0.04,
        0.925,
        "Zero drive converges to a fixed point; injected noise restores "
        "high-dimensional fluctuations but not an alpha rhythm. Human "
        "comparison is qualitative; no human EEG was recorded.",
        color=MUTED,
        fontsize=9,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.12
    )
    plt.close(figure)

    report = {
        "checkpoint": args.checkpoint,
        "step": int(payload["step"]),
        "jacobian_spectral_radius": spectral_radius,
        "zero_drive": {
            "fixed_point_norm": float(zero_norm[-1]),
            "rank": zero_stats["rank"],
            "psd_slope": zero_stats["psd_slope"],
            "mean": zero_stats["mean"],
        },
        "noise_sweep": results,
        "weights_positive_fraction": float((weights > 0).mean()),
        "alpha_mean": float(alpha.mean()),
        "gain_mean": float(gain.mean()),
        "human_data": "none; qualitative comparison only",
        "tick_reference_seconds": 0.01,
        "caveat": (
            "injected drive noise is not biological noise; tick time is "
            "arbitrary; no human EEG was recorded"
        ),
    }
    json_path = output.with_suffix(".json")
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {output} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
