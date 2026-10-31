#!/usr/bin/env python3
"""Comprehensive brain-activity dashboard for stage3 vs lowfreq.

Panels: raw population trace, PSD + 1/f fit, band power, autocorrelation, DFA,
phase-amplitude coupling, region coherence, region heatmap, state transitions,
effective rank/avalanche summary.

This is rate-neuron population activity, not EEG. The 100 Hz axis is the same
arbitrary 1 tick = 10 ms reference used elsewhere.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, coherence, hilbert, sosfiltfilt, welch
from sklearn.cluster import KMeans

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

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
COLORS = {
    "stage3 teacher": "#4dd0e1",
    "stage3 autonomous": "#74c0fc",
    "lowfreq teacher": "#ffa94d",
    "lowfreq autonomous": "#ff6b6b",
}


def style(axis) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.tick_params(colors=MUTED, labelsize=7)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")


def load_text(path: Path, max_tokens: int) -> str:
    chunks, total = [], 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if text:
                chunks.append(text)
                total += len(text)
            if total >= max_tokens * 4:
                break
    return "\n".join(chunks)


@torch.no_grad()
def teacher_trace(model, tokenizer, text: str, tokens: int) -> np.ndarray:
    ids = torch.tensor(
        [tokenizer.encode(text)[:tokens]], dtype=torch.long, device="cuda"
    )
    model.brain.trace = []
    state = None
    for start in range(0, ids.shape[1], 128):
        output = model(ids[:, start : start + 128], state=state)
        state = output.state
    trace = torch.stack(model.brain.trace).float().cpu().numpy()
    model.brain.trace = None
    return trace


@torch.no_grad()
def autonomous_trace(model, tokenizer, text: str, ticks: int) -> np.ndarray:
    ids = torch.tensor(
        [tokenizer.encode(text)[:128]], dtype=torch.long, device="cuda"
    )
    output = model(ids)
    state = output.state
    model.brain.trace = []
    zero = torch.zeros(1, 1, model.brain.d_model, device="cuda")
    for _ in range(ticks // model.config.brain.ticks_per_token):
        _, state, _ = model.brain(zero, state=state)
    trace = torch.stack(model.brain.trace).float().cpu().numpy()
    model.brain.trace = None
    return trace


def region_trace(trace: np.ndarray, regions: int) -> np.ndarray:
    n = trace.shape[1]
    return np.stack(
        [trace[:, np.arange(n) * regions // n == index].mean(1)
         for index in range(regions)]
    )


def spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x - x.mean()
    return welch(
        x,
        fs=100.0,
        nperseg=min(512, len(x)),
        noverlap=min(256, len(x) // 2),
        detrend="constant",
    )


def metrics(trace: np.ndarray) -> dict[str, object]:
    regions = region_trace(trace, 8)
    pop = trace.mean(1)
    freq, power = spectrum(pop)
    valid = (freq >= 0.5) & (freq <= 45)
    slope = float(np.polyfit(np.log(freq[valid]), np.log(power[valid]), 1)[0])
    total = np.trapezoid(power, freq)
    bands = {}
    for name, low, high, _ in BANDS:
        mask = (freq >= low) & (freq < high)
        bands[name] = float(np.trapezoid(power[mask], freq[mask]) / total)
    centered = pop - pop.mean()
    acf = np.correlate(centered, centered, mode="full")[len(pop) - 1 :]
    acf = acf / max(acf[0], 1e-12)
    below = np.flatnonzero(np.abs(acf) < 1 / math.e)
    decay = int(below[0]) if below.size else len(pop)
    dfa = dfa_slope(pop)
    # Phase-amplitude coupling: theta phase, gamma amplitude.
    theta = sosfiltfilt(
        butter(3, [4, 8], btype="bandpass", fs=100, output="sos"), pop
    )
    gamma = sosfiltfilt(
        butter(3, [30, 45], btype="bandpass", fs=100, output="sos"), pop
    )
    phase = np.angle(hilbert(theta))
    amplitude = np.abs(hilbert(gamma))
    pac = float(
        np.abs(np.mean(amplitude * np.exp(1j * phase)))
        / max(np.mean(amplitude), 1e-12)
    )
    # Region coherence at 1-8 Hz.
    matrix = np.eye(8)
    for i in range(8):
        for j in range(i + 1, 8):
            f, c = coherence(regions[i], regions[j], fs=100, nperseg=256)
            matrix[i, j] = matrix[j, i] = float(
                c[(f >= 1) & (f <= 8)].mean()
            )
    # State transitions from 4-means clusters of region patterns.
    features = (regions.T - regions.T.mean(0)) / (
        regions.T.std(0) + 1e-8
    )
    labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(
        features
    )
    transitions = np.zeros((4, 4))
    for a, b in zip(labels[:-1], labels[1:]):
        transitions[a, b] += 1
    transitions /= transitions.sum(1, keepdims=True).clip(min=1)
    dwell = 1.0 / (1.0 - np.diag(transitions).clip(max=0.999))
    # Effective rank over time.
    singular = np.linalg.svd(trace - trace.mean(0), compute_uv=False)
    energy = singular**2
    prob = energy / energy.sum()
    rank = float(np.exp(-(prob * np.log(prob + 1e-12)).sum()))
    # Simple avalanche statistics: contiguous above-threshold population runs.
    threshold = pop.mean() + pop.std()
    active = pop > threshold
    sizes = []
    run = 0
    for value in active:
        if value:
            run += 1
        elif run:
            sizes.append(run)
            run = 0
    if run:
        sizes.append(run)
    return {
        "freq": freq,
        "power": power,
        "slope": slope,
        "bands": bands,
        "acf": acf[:300],
        "decay": decay,
        "dfa": dfa,
        "pac": pac,
        "coherence": matrix,
        "transitions": transitions,
        "dwell": float(dwell.mean()),
        "rank": rank,
        "avalanche_count": len(sizes),
        "avalanche_max": int(max(sizes) if sizes else 0),
        "avalanche_mean": float(np.mean(sizes) if sizes else 0),
        "pop_std": float(pop.std()),
        "regions": regions,
    }


def dfa_slope(x: np.ndarray) -> float:
    y = np.cumsum(x - x.mean())
    scales = np.unique(
        np.logspace(np.log10(8), np.log10(max(16, len(x) // 4)), 12).astype(int)
    )
    fluctuation = []
    used = []
    for scale in scales:
        segments = len(y) // scale
        if segments < 2:
            continue
        rms = []
        for index in range(segments):
            segment = y[index * scale : (index + 1) * scale]
            t = np.arange(scale)
            coeff = np.polyfit(t, segment, 1)
            rms.append(np.sqrt(np.mean((segment - np.polyval(coeff, t)) ** 2)))
        fluctuation.append(np.mean(rms))
        used.append(scale)
    if len(used) < 3:
        return 0.0
    return float(np.polyfit(np.log(used), np.log(fluctuation), 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs=2, default=["runs/cortex_stage3/final.pt", "runs/cortex_lowfreq/final.pt"])
    parser.add_argument("--models", nargs=2, default=["stage3", "lowfreq"])
    parser.add_argument("--corpus", default="data/minimind/pretrain_hq.jsonl")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--ticks", type=int, default=2048)
    parser.add_argument("--output", default="llm/results/cortex/brainwave_dashboard.png")
    args = parser.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        raise SystemExit(f"corpus not found: {corpus}")
    text = load_text(corpus, args.tokens)
    traces = {}
    for name, checkpoint in zip(args.models, args.checkpoints):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = ExperimentConfig.from_dict(payload["config"])
        tokenizer = build_tokenizer(config.data, config.backbone)
        model = CortexSandwichLM(config, tokenizer.vocab_size, tokenizer=tokenizer)
        load_checkpoint(checkpoint, model, map_location="cpu")
        model.cuda().eval()
        print(f"recording {name} teacher...")
        traces[f"{name} teacher"] = teacher_trace(model, tokenizer, text, args.tokens)
        print(f"recording {name} autonomous...")
        traces[f"{name} autonomous"] = autonomous_trace(model, tokenizer, text, args.ticks)

    results = {name: metrics(trace) for name, trace in traces.items()}

    figure = plt.figure(figsize=(21, 14), dpi=160)
    figure.patch.set_facecolor(BACKGROUND)
    grid = figure.add_gridspec(
        3, 4, left=0.04, right=0.975, top=0.90, bottom=0.07,
        hspace=0.42, wspace=0.28,
    )

    # 1 raw population traces
    axis = figure.add_subplot(grid[0, 0])
    style(axis)
    for name, trace in traces.items():
        pop = trace.mean(1) - trace.mean()
        axis.plot(pop[:1400], color=COLORS[name], lw=0.8, alpha=0.9, label=name)
    axis.set_xlabel("tick", color=MUTED, fontsize=7)
    axis.set_ylabel("centered pop.", color=MUTED, fontsize=7)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=6)
    axis.set_title("1 raw population traces", color=TEXT, fontsize=10, pad=6)

    # 2 PSD
    axis = figure.add_subplot(grid[0, 1])
    style(axis)
    for name, result in results.items():
        axis.loglog(result["freq"][1:], result["power"][1:] + 1e-30, color=COLORS[name], lw=1.0, label=name)
    for _, low, high, color in BANDS:
        axis.axvspan(low, high, color=color, alpha=0.06)
    axis.set_xlim(0.5, 50)
    axis.set_xlabel("Hz (1 tick = 10 ms)", color=MUTED, fontsize=7)
    axis.set_ylabel("PSD", color=MUTED, fontsize=7)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=6)
    axis.set_title("2 PSD, 1/f slope", color=TEXT, fontsize=10, pad=6)

    # 3 band power
    axis = figure.add_subplot(grid[0, 2])
    style(axis)
    names = list(results)
    positions = np.arange(len(BANDS))
    width = 0.8 / len(names)
    for index, name in enumerate(names):
        values = [results[name]["bands"][band[0]] for band in BANDS]
        axis.bar(positions + index * width - 0.4, values, width, color=COLORS[name], label=name)
    axis.set_xticks(positions)
    axis.set_xticklabels([b[0] for b in BANDS], color=MUTED, fontsize=7)
    axis.set_ylabel("relative power", color=MUTED, fontsize=7)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=6)
    axis.set_title("3 five-band power", color=TEXT, fontsize=10, pad=6)

    # 4 autocorrelation
    axis = figure.add_subplot(grid[0, 3])
    style(axis)
    for name, result in results.items():
        axis.plot(result["acf"], color=COLORS[name], lw=1.0, label=name)
    axis.axhline(0, color=MUTED, lw=0.5)
    axis.set_xlabel("lag (ticks)", color=MUTED, fontsize=7)
    axis.set_ylabel("ACF", color=MUTED, fontsize=7)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=6)
    axis.set_title("4 autocorrelation", color=TEXT, fontsize=10, pad=6)

    # 5 DFA
    axis = figure.add_subplot(grid[1, 0])
    style(axis)
    for name, result in results.items():
        axis.bar(name, result["dfa"], color=COLORS[name])
    axis.set_ylabel("DFA α", color=MUTED, fontsize=7)
    axis.tick_params(axis="x", rotation=20, labelsize=6)
    axis.set_title("5 long-range temporal correlation", color=TEXT, fontsize=10, pad=6)

    # 6 PAC
    axis = figure.add_subplot(grid[1, 1])
    style(axis)
    for name, result in results.items():
        axis.bar(name, result["pac"], color=COLORS[name])
    axis.set_ylabel("theta–gamma MVL", color=MUTED, fontsize=7)
    axis.text(0.02, 0.95, "stage3 autonomous PAC is inflated\nby near-zero gamma amplitude", transform=axis.transAxes, color=MUTED, fontsize=6, va="top")
    axis.tick_params(axis="x", rotation=20, labelsize=6)
    axis.set_title("6 phase–amplitude coupling", color=TEXT, fontsize=10, pad=6)

    # 7 coherence stage3 teacher
    axis = figure.add_subplot(grid[1, 2])
    style(axis)
    image = axis.imshow(results["stage3 teacher"]["coherence"], cmap="viridis", vmin=0, vmax=1)
    axis.set_title("7 region coherence (stage3 teacher)", color=TEXT, fontsize=9, pad=6)
    axis.set_xlabel("region", color=MUTED, fontsize=7)
    axis.set_ylabel("region", color=MUTED, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02).ax.tick_params(colors=MUTED, labelsize=6)

    # 8 coherence lowfreq autonomous
    axis = figure.add_subplot(grid[1, 3])
    style(axis)
    image = axis.imshow(results["lowfreq autonomous"]["coherence"], cmap="viridis", vmin=0, vmax=1)
    axis.set_title("8 region coherence (lowfreq autonomous)", color=TEXT, fontsize=9, pad=6)
    axis.set_xlabel("region", color=MUTED, fontsize=7)
    axis.set_ylabel("region", color=MUTED, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02).ax.tick_params(colors=MUTED, labelsize=6)

    # 9 region heatmap lowfreq autonomous
    axis = figure.add_subplot(grid[2, 0])
    style(axis)
    image = axis.imshow(results["lowfreq autonomous"]["regions"][:, :1200], aspect="auto", cmap="magma")
    axis.set_title("9 regions over time (lowfreq autonomous)", color=TEXT, fontsize=9, pad=6)
    axis.set_xlabel("tick", color=MUTED, fontsize=7)
    axis.set_ylabel("region", color=MUTED, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02).ax.tick_params(colors=MUTED, labelsize=6)

    # 10 state transitions
    axis = figure.add_subplot(grid[2, 1])
    style(axis)
    image = axis.imshow(results["lowfreq autonomous"]["transitions"], cmap="magma", vmin=0, vmax=1)
    axis.set_title("10 state transitions (4-means)", color=TEXT, fontsize=9, pad=6)
    axis.set_xlabel("next state", color=MUTED, fontsize=7)
    axis.set_ylabel("state", color=MUTED, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02).ax.tick_params(colors=MUTED, labelsize=6)

    # 11 rank / avalanche
    axis = figure.add_subplot(grid[2, 2])
    style(axis)
    metric_names = ["rank", "avalanche_max", "avalanche_mean", "pop_std"]
    positions = np.arange(len(metric_names))
    width = 0.8 / len(results)
    for index, name in enumerate(results):
        values = [results[name][key] for key in metric_names]
        axis.bar(positions + index * width - 0.4, values, width, color=COLORS[name], label=name)
    axis.set_xticks(positions)
    axis.set_xticklabels(metric_names, rotation=15, fontsize=6)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=6)
    axis.set_yscale("log")
    axis.set_title("11 rank / avalanche (log)", color=TEXT, fontsize=10, pad=6)

    # 12 summary table
    axis = figure.add_subplot(grid[2, 3])
    style(axis)
    axis.axis("off")
    header = f"{'condition':22s} {'slope':>6s} {'DFA':>5s} {'PAC':>6s} {'rank':>6s}"
    lines = [header, "-" * 52]
    for name, result in results.items():
        lines.append(
            f"{name:22s} {result['slope']:6.2f} {result['dfa']:5.2f} "
            f"{result['pac']:6.3f} {result['rank']:6.1f}"
        )
    lines += ["", "No human EEG was recorded.", "100 Hz axis is arbitrary.", "Rate neurons, no spikes."]
    axis.text(0.0, 1.0, "\n".join(lines), color=TEXT, fontsize=7.5, va="top", family="monospace")
    axis.set_title("12 summary", color=TEXT, fontsize=10, pad=6)

    figure.suptitle(
        "Cortex-LM comprehensive brain-activity dashboard · stage3 vs lowfreq",
        color=TEXT,
        fontsize=17,
        x=0.04,
        ha="left",
        y=0.965,
    )
    figure.text(
        0.04,
        0.925,
        "rate-neuron population activity · 4 conditions · 1 tick = 10 ms reference · no human EEG",
        color=MUTED,
        fontsize=8.5,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)

    report = {
        "conditions": {
            name: {
                key: value
                for key, value in result.items()
                if key not in {"freq", "power", "acf", "coherence", "transitions", "regions"}
            }
            for name, result in results.items()
        },
        "caveat": "rate-neuron population activity, not EEG; no human data",
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["conditions"], indent=2, ensure_ascii=False)[:2000])
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
