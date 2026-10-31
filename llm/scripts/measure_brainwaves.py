#!/usr/bin/env python3
"""Measure the cortical sheet's population "brainwaves" during a long run.

There is no biological time in Cortex-LM: one tick is one rate-neuron update.
This script records one scalar per neuron per tick and computes a population
PSD under an explicit, arbitrary reference mapping (1 tick = 10 ms, 100 Hz).
The human comparison is therefore qualitative: no human EEG was recorded and no
absolute frequency claim is made.
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

from neurosim_gpu.cortex.checkpoint import load_checkpoint  # noqa: E402
from neurosim_gpu.cortex.config import ExperimentConfig  # noqa: E402
from neurosim_gpu.cortex.model import CortexSandwichLM  # noqa: E402
from neurosim_gpu.cortex.tokenization import build_tokenizer  # noqa: E402

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


def load_text(path: Path, max_tokens: int) -> str:
    chunks: list[str] = []
    total = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if not text:
                continue
            chunks.append(text)
            total += len(text)
            if total >= max_tokens * 4:
                break
    return "\n".join(chunks)


@torch.no_grad()
def teacher_trace(
    model: CortexSandwichLM,
    token_ids: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    model.brain.trace = []
    state = None
    for start in range(0, token_ids.shape[1], chunk):
        ids = token_ids[:, start : start + chunk].cuda()
        output = model(ids, state=state)
        state = output.state
    trace = torch.stack(model.brain.trace).float().cpu()
    model.brain.trace = None
    return trace


@torch.no_grad()
def autonomous_trace(
    model: CortexSandwichLM,
    token_ids: torch.Tensor,
    ticks: int,
) -> torch.Tensor:
    # Prefill the first block, then run the brain with zero drive.
    output = model(token_ids[:, :128].cuda())
    state = output.state
    model.brain.trace = []
    zero = torch.zeros(
        1, 1, model.brain.d_model, device=state.device
    )
    for _ in range(ticks // model.config.brain.ticks_per_token):
        _, state, _ = model.brain(zero, state=state)
    trace = torch.stack(model.brain.trace).float().cpu()
    model.brain.trace = None
    return trace


def band_power(freq: np.ndarray, power: np.ndarray) -> dict[str, float]:
    total = np.trapz(power, freq)
    result = {}
    for name, low, high, _ in BANDS:
        mask = (freq >= low) & (freq < high)
        result[name] = float(np.trapz(power[mask], freq[mask]) / total)
    return result


def analyze(trace: torch.Tensor, fs: float) -> dict[str, object]:
    values = trace.numpy()
    mean = values.mean(axis=1)
    mean -= mean.mean()
    std = values.std(axis=1)
    freq, power = welch(
        mean,
        fs=fs,
        nperseg=min(512, len(mean)),
        noverlap=min(256, len(mean) // 2),
        detrend="constant",
    )
    valid = freq > 0
    slope = float(
        np.polyfit(np.log(freq[valid]), np.log(power[valid]), 1)[0]
    )
    autocorrelation = np.correlate(mean, mean, mode="full")[
        len(mean) - 1 :
    ]
    autocorrelation /= autocorrelation[0]
    below = np.flatnonzero(np.abs(autocorrelation) < 1 / math.e)
    correlation_time = int(below[0]) if below.size else len(mean)
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered[::2], compute_uv=False)
    energy = singular**2
    probability = energy / energy.sum()
    effective_rank = float(
        np.exp(-(probability * np.log(probability + 1e-12)).sum())
    )
    return {
        "ticks": int(len(mean)),
        "mean_activity": float(values.mean()),
        "std_activity": float(values.std()),
        "population_mean_std": float(mean.std()),
        "region_std_mean": float(std.mean()),
        "psd_slope": slope,
        "relative_band_power": band_power(freq, power),
        "autocorrelation_time_ticks": correlation_time,
        "effective_rank": effective_rank,
        "frequency": freq,
        "power": power,
        "autocorrelation": autocorrelation[:300],
        "mean_trace": mean,
    }


def style_axis(axis) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.tick_params(colors=MUTED, labelsize=7)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/cortex_stage3/final.pt")
    parser.add_argument(
        "--corpus",
        default="data/minimind/pretrain_hq.jsonl",
        help="JSONL corpus; not included in the repository",
    )
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--autonomous-ticks", type=int, default=2048)
    parser.add_argument(
        "--output", default="llm/results/cortex/brainwaves.png"
    )
    args = parser.parse_args()

    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    config = ExperimentConfig.from_dict(payload["config"])
    tokenizer = build_tokenizer(config.data, config.backbone)
    model = CortexSandwichLM(
        config, tokenizer.vocab_size, tokenizer=tokenizer
    )
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.cuda().eval()

    corpus = Path(args.corpus)
    if not corpus.exists():
        raise SystemExit(
            f"corpus not found: {corpus}; pass --corpus with a local JSONL"
        )
    text = load_text(corpus, args.tokens)
    token_ids = torch.tensor(
        [tokenizer.encode(text)[: args.tokens]], dtype=torch.long
    )
    if token_ids.shape[1] < 256:
        raise SystemExit("corpus returned too few tokens")

    print("recording teacher-forced activity...")
    teacher = teacher_trace(model, token_ids, chunk=128)
    print("recording autonomous activity...")
    autonomous = autonomous_trace(
        model, token_ids, args.autonomous_ticks
    )
    fs = 100.0  # arbitrary reference: 1 tick = 10 ms
    teacher_stats = analyze(teacher, fs)
    autonomous_stats = analyze(autonomous, fs)

    figure = plt.figure(figsize=(18, 10), dpi=170)
    figure.patch.set_facecolor(BACKGROUND)
    grid = figure.add_gridspec(
        2, 3, left=0.04, right=0.97, top=0.88, bottom=0.10,
        wspace=0.20, hspace=0.30,
    )

    axis = figure.add_subplot(grid[0, 0])
    style_axis(axis)
    ticks = np.arange(min(1500, len(teacher_stats["mean_trace"])))
    axis.plot(
        ticks,
        teacher_stats["mean_trace"][ticks],
        color="#4dd0e1",
        lw=1.0,
        label="teacher-forced",
    )
    autonomous_ticks = np.arange(
        min(1500, len(autonomous_stats["mean_trace"]))
    )
    axis.plot(
        autonomous_ticks,
        autonomous_stats["mean_trace"][autonomous_ticks],
        color="#ffa94d",
        lw=1.0,
        alpha=0.9,
        label="autonomous",
    )
    axis.set_xlabel("tick", color=MUTED, fontsize=8)
    axis.set_ylabel("population mean", color=MUTED, fontsize=8)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=7)
    axis.set_title(
        "population activity over time", color=TEXT, fontsize=11, pad=8
    )

    axis = figure.add_subplot(grid[0, 1])
    style_axis(axis)
    for name, low, high, color in BANDS:
        axis.axvspan(low, high, color=color, alpha=0.08)
    axis.axvline(10.0, color="#51cf66", lw=0.8, ls="--", alpha=0.6)
    axis.loglog(
        teacher_stats["frequency"][1:],
        teacher_stats["power"][1:],
        color="#4dd0e1",
        lw=1.2,
        label="teacher-forced",
    )
    axis.loglog(
        autonomous_stats["frequency"][1:],
        autonomous_stats["power"][1:],
        color="#ffa94d",
        lw=1.2,
        label="autonomous",
    )
    axis.set_xlim(0.5, 50)
    axis.set_xlabel("Hz (reference: 1 tick = 10 ms)", color=MUTED, fontsize=8)
    axis.set_ylabel("PSD", color=MUTED, fontsize=8)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=7)
    axis.set_title(
        "population PSD with human EEG bands", color=TEXT, fontsize=11, pad=8
    )

    axis = figure.add_subplot(grid[0, 2])
    style_axis(axis)
    axis.plot(
        teacher_stats["autocorrelation"],
        color="#4dd0e1",
        lw=1.2,
        label="teacher-forced",
    )
    axis.plot(
        autonomous_stats["autocorrelation"],
        color="#ffa94d",
        lw=1.2,
        label="autonomous",
    )
    axis.axhline(0, color=MUTED, lw=0.5)
    axis.axhline(1 / math.e, color=MUTED, lw=0.5, ls=":")
    axis.set_xlabel("lag (ticks)", color=MUTED, fontsize=8)
    axis.set_ylabel("autocorrelation", color=MUTED, fontsize=8)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=7)
    axis.set_title(
        "autocorrelation and decay time", color=TEXT, fontsize=11, pad=8
    )

    axis = figure.add_subplot(grid[1, 0])
    style_axis(axis)
    positions = np.arange(len(BANDS))
    width = 0.36
    teacher_values = [
        teacher_stats["relative_band_power"][name] for name, *_ in BANDS
    ]
    autonomous_values = [
        autonomous_stats["relative_band_power"][name] for name, *_ in BANDS
    ]
    axis.bar(
        positions - width / 2,
        teacher_values,
        width,
        color="#4dd0e1",
        label="teacher-forced",
    )
    axis.bar(
        positions + width / 2,
        autonomous_values,
        width,
        color="#ffa94d",
        label="autonomous",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([band[0] for band in BANDS], color=MUTED, fontsize=8)
    axis.set_ylabel("relative power", color=MUTED, fontsize=8)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=7)
    axis.set_title(
        "band power under the 100 Hz reference", color=TEXT, fontsize=11, pad=8
    )

    axis = figure.add_subplot(grid[1, 1])
    style_axis(axis)
    region = np.arange(1024) * 8 // 1024
    region_trace = np.stack(
        [teacher.numpy()[:, region == index].mean(1) for index in range(8)]
    )
    region_trace = region_trace[:, :800]
    image = axis.imshow(
        region_trace,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    axis.set_xlabel("tick", color=MUTED, fontsize=8)
    axis.set_ylabel("region", color=MUTED, fontsize=8)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    colorbar.ax.tick_params(colors=MUTED, labelsize=7)
    colorbar.outline.set_edgecolor("#2b3038")
    axis.set_title(
        "8-region activity (teacher-forced)", color=TEXT, fontsize=11, pad=8
    )

    axis = figure.add_subplot(grid[1, 2])
    style_axis(axis)
    axis.axis("off")
    lines = [
        ("ticks", f"{teacher_stats['ticks']} / {autonomous_stats['ticks']}"),
        ("PSD slope", f"{teacher_stats['psd_slope']:.2f} / {autonomous_stats['psd_slope']:.2f}"),
        ("autocorr τ", f"{teacher_stats['autocorrelation_time_ticks']} / {autonomous_stats['autocorrelation_time_ticks']} ticks"),
        ("effective rank", f"{teacher_stats['effective_rank']:.1f} / {autonomous_stats['effective_rank']:.1f}"),
        ("mean activity", f"{teacher_stats['mean_activity']:.3f} / {autonomous_stats['mean_activity']:.3f}"),
        ("std activity", f"{teacher_stats['std_activity']:.3f} / {autonomous_stats['std_activity']:.3f}"),
    ]
    axis.text(
        0.0,
        1.0,
        "model measurements (teacher / autonomous)\n\n"
        + "\n".join(f"{name:16s} {value}" for name, value in lines)
        + "\n\nHuman comparison is qualitative:\n"
        "no human EEG was recorded.\n"
        "Human resting EEG has a 1/f-like PSD,\n"
        "prominent alpha (8–13 Hz), and\n"
        "state-dependent band ratios.\n"
        "The model has no biological time,\n"
        "spikes, or alpha generator; the\n"
        "100 Hz axis is an arbitrary reference.",
        color=TEXT,
        fontsize=8.5,
        va="top",
        family="monospace",
    )
    axis.set_title("measurement summary", color=TEXT, fontsize=11, pad=8)

    figure.suptitle(
        "Cortex-LM population activity · stage3 · no spikes, no biological time",
        color=TEXT,
        fontsize=16,
        x=0.04,
        ha="left",
        y=0.965,
    )
    figure.text(
        0.04,
        0.925,
        "Each tick = one rate-neuron update. 1 tick = 10 ms is an explicit "
        "reference mapping, not a measured time scale.",
        color=MUTED,
        fontsize=9,
        ha="left",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.12
    )
    plt.close(figure)

    def serialisable(stats: dict[str, object]) -> dict[str, object]:
        return {
            key: value
            for key, value in stats.items()
            if key
            not in {
                "frequency",
                "power",
                "autocorrelation",
                "mean_trace",
            }
        }

    report = {
        "checkpoint": args.checkpoint,
        "step": int(payload["step"]),
        "reference_tick_seconds": 0.01,
        "human_data": "none; qualitative comparison only",
        "teacher_forced": serialisable(teacher_stats),
        "autonomous": serialisable(autonomous_stats),
        "caveat": (
            "rate neurons with no spikes; tick time is arbitrary; no human "
            "EEG recording was used"
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
