#!/usr/bin/env python3
"""Toy integration test for the cognitive-core prototypes.

Not a language experiment. It checks that the workspace, self-monitor,
hippocampal memory and active-inquiry gate have the intended behavior before
they are wired into Cortex-LM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neurosim_gpu.cortex.cognition import (
    GlobalWorkspace,
    HippocampalMemory,
    SelfMonitor,
    WorkspaceConfig,
    expected_calibration_error,
    should_ask,
)

BACKGROUND = "#0b0f14"
TEXT = "#f1f3f5"
MUTED = "#adb5bd"


def style(axis) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="llm/results/cortex/cognitive_core.png")
    args = parser.parse_args()
    torch.manual_seed(0)
    d_model = 64
    generator = torch.Generator().manual_seed(7)
    patterns = torch.randn(32, d_model, generator=generator)

    workspace = GlobalWorkspace(d_model, WorkspaceConfig(slots=32, top_k=8))
    for pattern in patterns:
        workspace(pattern.unsqueeze(0), write=True)

    loads, ignitions, broadcast_cosines = [], [], []
    for load in (1, 2, 4, 8, 16):
        mixture = patterns[:load].mean(0, keepdim=True)
        broadcast, stats = workspace(mixture)
        loads.append(load)
        ignitions.append(float(stats["ignition"]))
        target = torch.nn.functional.normalize(mixture, dim=-1)
        value = torch.nn.functional.normalize(broadcast, dim=-1)
        broadcast_cosines.append(float((target * value).sum()))

    memory = HippocampalMemory(d_model, slots=64, sparse_k=8)
    for pattern in patterns:
        memory.write(pattern)
    noises, recalls = [], []
    for noise in (0.0, 0.05, 0.1, 0.2, 0.35):
        noisy = patterns + noise * torch.randn_like(patterns)
        recalled = memory.read(noisy)
        cosine = torch.nn.functional.cosine_similarity(
            recalled, patterns, dim=-1
        )
        noises.append(noise)
        recalls.append(float(cosine.mean()))

    monitor = SelfMonitor(d_model)
    with torch.no_grad():
        scores = monitor(patterns)
    confidence = scores["confidence"]
    # Synthetic correctness correlated with confidence, plus miscalibration.
    correctness = (
        torch.rand_like(confidence) < (0.2 + 0.7 * confidence)
    ).float()
    ece = expected_calibration_error(confidence, correctness, bins=8)
    ask = should_ask(confidence, 0.5)
    confidence_np = confidence.numpy()
    correct_np = correctness.numpy()
    bins = np.linspace(0, 1, 9)
    bin_centers, bin_accuracy, bin_confidence = [], [], []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence_np >= low) & (confidence_np < high)
        if mask.any():
            bin_centers.append((low + high) / 2)
            bin_accuracy.append(float(correct_np[mask].mean()))
            bin_confidence.append(float(confidence_np[mask].mean()))

    figure = plt.figure(figsize=(16, 9), dpi=170)
    figure.patch.set_facecolor(BACKGROUND)
    grid = figure.add_gridspec(
        2, 3, left=0.05, right=0.97, top=0.88, bottom=0.10,
        hspace=0.35, wspace=0.25,
    )

    axis = figure.add_subplot(grid[0, 0])
    style(axis)
    axis.plot(loads, ignitions, marker="o", color="#4dd0e1")
    axis.set_xlabel("active patterns in workspace input", color=MUTED, fontsize=8)
    axis.set_ylabel("ignition", color=MUTED, fontsize=8)
    axis.set_title("workspace ignition vs load", color=TEXT, fontsize=10)

    axis = figure.add_subplot(grid[0, 1])
    style(axis)
    axis.plot(loads, broadcast_cosines, marker="o", color="#ffa94d")
    axis.set_ylim(0, 1)
    axis.set_xlabel("active patterns", color=MUTED, fontsize=8)
    axis.set_ylabel("broadcast cosine", color=MUTED, fontsize=8)
    axis.set_title("broadcast fidelity", color=TEXT, fontsize=10)

    axis = figure.add_subplot(grid[0, 2])
    style(axis)
    axis.plot(noises, recalls, marker="o", color="#51cf66")
    axis.set_ylim(0, 1)
    axis.set_xlabel("input noise", color=MUTED, fontsize=8)
    axis.set_ylabel("recall cosine", color=MUTED, fontsize=8)
    axis.set_title("hippocampal one-shot recall", color=TEXT, fontsize=10)

    axis = figure.add_subplot(grid[1, 0])
    style(axis)
    axis.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=0.8, label="perfect")
    axis.plot(bin_confidence, bin_accuracy, marker="o", color="#fcc419", label="probe")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("confidence", color=MUTED, fontsize=8)
    axis.set_ylabel("accuracy", color=MUTED, fontsize=8)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=7)
    axis.set_title(f"calibration probe (ECE {ece:.3f})", color=TEXT, fontsize=10)

    axis = figure.add_subplot(grid[1, 1])
    style(axis)
    axis.hist(confidence_np, bins=12, color="#b197fc", alpha=0.85)
    axis.axvline(0.5, color="#ffa94d", ls="--", lw=0.8)
    axis.set_xlabel("confidence", color=MUTED, fontsize=8)
    axis.set_ylabel("count", color=MUTED, fontsize=8)
    axis.set_title(
        f"ask rate below 0.5: {float(ask.float().mean()):.1%}",
        color=TEXT,
        fontsize=10,
    )

    axis = figure.add_subplot(grid[1, 2])
    style(axis)
    axis.axis("off")
    lines = [
        "cognitive-core toy test",
        "",
        f"d_model            {d_model}",
        f"workspace slots    {workspace.config.slots}",
        f"workspace top-k    {workspace.config.top_k}",
        f"memory slots       {memory.slots}",
        f"sparse k           {memory.sparse_k}",
        f"ECE                {ece:.3f}",
        f"ask rate           {float(ask.float().mean()):.1%}",
        "",
        "This is not a language model.",
        "No consciousness claim.",
        "Purpose: verify interfaces and",
        "ablation surfaces before wiring",
        "into Cortex-LM.",
    ]
    axis.text(0.0, 1.0, "\n".join(lines), color=TEXT, fontsize=8.5, va="top", family="monospace")
    axis.set_title("summary", color=TEXT, fontsize=10)

    figure.suptitle(
        "Cognitive-core prototypes: workspace, self-monitor, hippocampus, inquiry",
        color=TEXT,
        fontsize=16,
        x=0.05,
        ha="left",
        y=0.965,
    )
    figure.text(
        0.05,
        0.92,
        "toy vectors, not Cortex-LM activations; no consciousness claim",
        color=MUTED,
        fontsize=9,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    report = {
        "workspace": {
            "loads": loads,
            "ignition": ignitions,
            "broadcast_cosine": broadcast_cosines,
        },
        "memory": {"noise": noises, "recall_cosine": recalls},
        "self_monitor": {
            "ece": ece,
            "ask_rate": float(ask.float().mean()),
        },
        "caveat": "toy vectors; not a language model; no consciousness claim",
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
