#!/usr/bin/env python3
"""Perturbation probe: inject an impulse into the cortical state and measure
how the response propagates across regions and decays over time.

This is the first Phase-0/6 measurement for the consciousness route: a causal
probe, not just an observation. Rate neurons, not EEG.
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

from neurosim_gpu.cortex.checkpoint import load_checkpoint  # noqa: E402
from neurosim_gpu.cortex.config import ExperimentConfig  # noqa: E402
from neurosim_gpu.cortex.model import CortexSandwichLM  # noqa: E402
from neurosim_gpu.cortex.tokenization import build_tokenizer  # noqa: E402

BACKGROUND = "#0b0f14"
TEXT = "#f1f3f5"
MUTED = "#adb5bd"


def read_text(path: Path, limit: int = 100_000) -> str:
    data = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if text:
                data.append(text)
            if sum(len(x) for x in data) >= limit:
                break
    return "\n".join(data)


@torch.no_grad()
def prefilled_state(model, tokenizer, text: str):
    ids = torch.tensor(
        [tokenizer.encode(text)[:128]], dtype=torch.long, device="cuda"
    )
    output = model(ids)
    return output.state.detach().clone()


@torch.no_grad()
def run(model, state, ticks: int, impulse=None):
    model.brain.trace = []
    current = state.clone()
    zero = torch.zeros(1, 1, model.brain.d_model, device="cuda")
    for tick in range(ticks):
        if impulse is not None and tick == 0:
            current = current + impulse
        _, current, _ = model.brain(zero, state=current)
    trace = torch.stack(model.brain.trace).float().cpu().numpy()
    model.brain.trace = None
    return trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/cortex_stage3/final.pt")
    parser.add_argument(
        "--corpus", default="data/minimind/pretrain_hq.jsonl"
    )
    parser.add_argument("--ticks", type=int, default=240)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--output", default="llm/results/cortex/perturbation_probe.png")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ExperimentConfig.from_dict(payload["config"])
    tokenizer = build_tokenizer(config.data, config.backbone)
    model = CortexSandwichLM(config, tokenizer.vocab_size, tokenizer=tokenizer)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.cuda().eval()
    text = read_text(Path(args.corpus))
    state = prefilled_state(model, tokenizer, text)
    baseline = run(model, state, args.ticks)
    generator = torch.Generator(device="cuda").manual_seed(7)
    selected = torch.randperm(
        model.brain.n_neurons, generator=generator, device="cuda"
    )[: args.neurons]
    impulse = torch.zeros_like(state)
    impulse[0, selected] = 0.5
    perturbed = run(model, state, args.ticks, impulse=impulse)

    region = model.brain.region.detach().cpu().numpy()
    regions = 8
    base_regions = np.stack(
        [baseline[:, region == index].mean(1) for index in range(regions)]
    )
    pert_regions = np.stack(
        [perturbed[:, region == index].mean(1) for index in range(regions)]
    )
    delta = pert_regions - base_regions
    baseline_pop = baseline.mean(1)
    perturbed_pop = perturbed.mean(1)

    figure = plt.figure(figsize=(15, 8), dpi=170)
    figure.patch.set_facecolor(BACKGROUND)
    grid = figure.add_gridspec(
        2, 2, left=0.06, right=0.96, top=0.86, bottom=0.11,
        hspace=0.34, wspace=0.22,
    )
    axis = figure.add_subplot(grid[0, 0])
    axis.set_facecolor(BACKGROUND)
    axis.plot(baseline.mean(0), color="#4dd0e1", lw=0.8, label="baseline")
    axis.plot(perturbed.mean(0), color="#ffa94d", lw=0.8, label="perturbed")
    axis.set_xlabel("tick", color=MUTED, fontsize=8)
    axis.set_ylabel("mean state", color=MUTED, fontsize=8)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=7)
    axis.set_title("impulse response", color=TEXT, fontsize=10)
    axis.tick_params(colors=MUTED, labelsize=7)

    axis = figure.add_subplot(grid[0, 1])
    axis.set_facecolor(BACKGROUND)
    image = axis.imshow(delta, aspect="auto", cmap="coolwarm", interpolation="nearest")
    axis.set_xlabel("tick", color=MUTED, fontsize=8)
    axis.set_ylabel("region", color=MUTED, fontsize=8)
    axis.set_title("perturbed − baseline by region", color=TEXT, fontsize=10)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    colorbar.ax.tick_params(colors=MUTED, labelsize=7)

    axis = figure.add_subplot(grid[1, 0])
    axis.set_facecolor(BACKGROUND)
    for index in range(regions):
        axis.plot(delta[index], lw=0.8, label=f"R{index}")
    axis.set_xlabel("tick", color=MUTED, fontsize=8)
    axis.set_ylabel("Δ activity", color=MUTED, fontsize=8)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=6, ncol=2)
    axis.set_title("region response traces", color=TEXT, fontsize=10)
    axis.tick_params(colors=MUTED, labelsize=7)

    axis = figure.add_subplot(grid[1, 1])
    axis.set_facecolor(BACKGROUND)
    axis.axis("off")
    peak = int(np.argmax(np.abs(delta).mean(0)))
    summary = [
        "perturbation probe",
        "",
        f"checkpoint       {args.checkpoint}",
        f"state neurons    {model.brain.n_neurons}",
        f"perturbed        {args.neurons}",
        f"impulse          +0.5",
        f"ticks            {args.ticks}",
        f"peak |Δ| tick    {peak}",
        f"peak |Δ| mean    {float(np.abs(delta).mean(0).max()):.4f}",
        f"late |Δ| mean    {float(np.abs(delta[:, -20:]).mean()):.4f}",
        "",
        "This is a causal probe, but still",
        "rate-neuron activity, not EEG.",
        "No consciousness claim.",
    ]
    axis.text(0.0, 1.0, "\n".join(summary), color=TEXT, fontsize=8.5, va="top", family="monospace")
    axis.set_title("summary", color=TEXT, fontsize=10)

    figure.suptitle(
        "Causal perturbation of the cortical sheet · impulse propagation",
        color=TEXT,
        fontsize=16,
        x=0.05,
        ha="left",
        y=0.96,
    )
    figure.text(
        0.05,
        0.92,
        "inject +0.5 into 64 neurons after prefill; measure region response over 240 ticks",
        color=MUTED,
        fontsize=9,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    report = {
        "checkpoint": args.checkpoint,
        "neurons": int(model.brain.n_neurons),
        "perturbed": args.neurons,
        "ticks": args.ticks,
        "peak_tick": peak,
        "peak_delta": float(np.abs(delta).mean(0).max()),
        "late_delta": float(np.abs(delta[:, -20:]).mean()),
        "region_peak": [float(np.abs(delta[i]).max()) for i in range(regions)],
        "caveat": "rate-neuron activity, not EEG; causal impulse but no consciousness claim",
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
