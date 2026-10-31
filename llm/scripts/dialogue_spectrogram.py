#!/usr/bin/env python3
"""Dialogue test with an annotated brain spectrogram.

Three annotated rows per figure:
1. token timeline: every input and output token with its text and tick span;
2. spectrogram of population activity over the same runtime timeline;
3. population activity trace.

This is rate-neuron activity, not EEG. The 1 tick = 10 ms axis is a reference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import spectrogram

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

from neurosim_gpu.cortex.checkpoint import load_checkpoint  # noqa: E402
from neurosim_gpu.cortex.config import ExperimentConfig  # noqa: E402
from neurosim_gpu.cortex.model import CortexSandwichLM  # noqa: E402
from neurosim_gpu.cortex.tokenization import build_tokenizer  # noqa: E402

BACKGROUND = "#0b0f14"
TEXT = "#f1f3f5"
MUTED = "#adb5bd"
INPUT_COLOR = "#4dd0e1"
OUTPUT_COLOR = "#ffa94d"
CJK = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")


def labels_for(tokenizer, ids: list[int]) -> list[str]:
    labels = []
    previous = ""
    for index in range(len(ids)):
        current = tokenizer.decode(ids[: index + 1])
        label = current[len(previous) :] or f"#{ids[index]}"
        labels.append(label)
        previous = current
    return labels


@torch.no_grad()
def record(model, tokenizer, prompt: str, max_new_tokens: int):
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    model.brain.trace = []
    begin = time.perf_counter()
    model(prompt_tensor)
    prefill = torch.stack(model.brain.trace).float().cpu().numpy()
    model.brain.trace = None
    prefill_seconds = time.perf_counter() - begin

    generated = prompt_tensor
    output_ids: list[int] = []
    output_traces: list[np.ndarray] = []
    latencies: list[float] = []
    for _ in range(max_new_tokens):
        model.brain.trace = []
        begin = time.perf_counter()
        output = model(generated)
        token = int(output.logits[:, -1].float().argmax(dim=-1))
        latency = time.perf_counter() - begin
        trace = torch.stack(model.brain.trace).float().cpu().numpy()
        model.brain.trace = None
        output_ids.append(token)
        output_traces.append(trace[-2:])
        latencies.append(latency)
        generated = torch.cat(
            [generated, torch.tensor([[token]], device="cuda")], dim=1
        )
        if token == tokenizer.eos_token_id:
            break

    input_labels = labels_for(tokenizer, prompt_ids)
    all_labels = labels_for(tokenizer, prompt_ids + output_ids)
    output_labels = all_labels[len(prompt_ids) :]
    if not output_labels:
        output_labels = ["<EOS>"]
    prefill_kept = prefill[:-2] if prefill.shape[0] >= 4 else prefill
    signal = np.concatenate(
        [prefill_kept.mean(1)] + [part.mean(1) for part in output_traces]
    )
    input_spans = [
        (index * 2, min((index + 1) * 2, prefill_kept.shape[0]))
        for index in range(len(input_labels))
    ]
    output_start = prefill_kept.shape[0]
    output_spans = [
        (output_start + index * 2, output_start + (index + 1) * 2)
        for index in range(len(output_labels))
    ]
    return {
        "prompt": prompt,
        "output": tokenizer.decode(output_ids) or "<EOS>",
        "input_labels": input_labels,
        "output_labels": output_labels,
        "input_spans": input_spans,
        "output_spans": output_spans,
        "signal": signal,
        "prefill_seconds": prefill_seconds,
        "latencies": latencies,
    }


def draw(record: dict, path: Path) -> None:
    signal = record["signal"]
    fs = 100.0
    nperseg = min(24, max(8, len(signal) // 2))
    frequencies, times, power = spectrogram(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=max(0, nperseg - 8),
        detrend="constant",
        scaling="density",
        mode="magnitude",
    )
    db = 20 * np.log10(power + 1e-12)
    input_end = (
        record["input_spans"][-1][1] * 0.01 if record["input_spans"] else 0
    )
    output_end = (
        record["output_spans"][-1][1] * 0.01
        if record["output_spans"]
        else input_end
    )
    end = max(output_end, input_end, 0.05)

    figure = plt.figure(figsize=(17, 10), dpi=170)
    figure.patch.set_facecolor(BACKGROUND)
    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=[0.9, 3.0, 1.0],
        left=0.07,
        right=0.93,
        top=0.84,
        bottom=0.10,
        hspace=0.16,
    )

    # Token timeline.
    axis = figure.add_subplot(grid[0, 0])
    axis.set_facecolor(BACKGROUND)
    axis.set_xlim(0, end * 1.02)
    axis.set_ylim(-0.6, 1.7)
    axis.axvspan(0, input_end, color=INPUT_COLOR, alpha=0.10)
    axis.axvspan(input_end, output_end, color=OUTPUT_COLOR, alpha=0.10)
    for index, ((start, stop), label) in enumerate(
        zip(record["input_spans"], record["input_labels"])
    ):
        axis.add_patch(
            plt.Rectangle(
                (start * 0.01, 1.0),
                max((stop - start) * 0.01, 0.008),
                0.45,
                facecolor=INPUT_COLOR,
                alpha=0.35,
                edgecolor=INPUT_COLOR,
                lw=0.6,
            )
        )
        axis.text(
            (start + stop) / 2 * 0.01,
            1.22,
            label or "·",
            color=INPUT_COLOR,
            fontsize=7.5,
            ha="center",
            va="center",
            rotation=0,
            fontproperties=CJK,
        )
    output_plot_labels = [f"t{index}" for index in range(len(record["output_labels"]))]
    for index, ((start, stop), label) in enumerate(
        zip(record["output_spans"], output_plot_labels)
    ):
        axis.add_patch(
            plt.Rectangle(
                (start * 0.01, 0.05),
                max((stop - start) * 0.01, 0.008),
                0.45,
                facecolor=OUTPUT_COLOR,
                alpha=0.35,
                edgecolor=OUTPUT_COLOR,
                lw=0.6,
            )
        )
        axis.text(
            (start + stop) / 2 * 0.01,
            0.27,
            label or "·",
            color=OUTPUT_COLOR,
            fontsize=7,
            ha="center",
            va="center",
            rotation=65,
            fontproperties=CJK,
        )
    axis.text(0.005, 1.55, "INPUT tokens", color=INPUT_COLOR, fontsize=9)
    axis.text(0.005, -0.35, "OUTPUT tokens", color=OUTPUT_COLOR, fontsize=9)
    axis.set_yticks([])
    axis.set_xticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        "token timeline: input text; output indices t0.. (full transcript in title)",
        color=TEXT,
        fontsize=10,
        pad=4,
    )

    # Spectrogram.
    axis = figure.add_subplot(grid[1, 0])
    axis.set_facecolor(BACKGROUND)
    image = axis.pcolormesh(times, frequencies, db, shading="auto", cmap="magma")
    axis.set_ylim(0, 50)
    axis.set_ylabel("frequency (Hz; 1 tick = 10 ms)", color=MUTED, fontsize=9)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")
    colorbar = figure.colorbar(image, ax=axis, pad=0.015, fraction=0.035)
    colorbar.set_label("log magnitude", color=MUTED, fontsize=8)
    colorbar.ax.tick_params(colors=MUTED, labelsize=7)
    colorbar.outline.set_edgecolor("#2b3038")
    axis.axvspan(0, input_end, color=INPUT_COLOR, alpha=0.06)
    axis.axvspan(input_end, output_end, color=OUTPUT_COLOR, alpha=0.06)
    for start, _ in record["input_spans"]:
        axis.axvline(start * 0.01, color=INPUT_COLOR, lw=0.6, ls="--", alpha=0.7)
    for start, _ in record["output_spans"]:
        axis.axvline(start * 0.01, color=OUTPUT_COLOR, lw=0.6, ls="--", alpha=0.7)
    axis.set_title(
        "brain spectrogram over the same runtime timeline",
        color=TEXT,
        fontsize=11,
        pad=6,
    )

    # Population trace.
    axis = figure.add_subplot(grid[2, 0])
    axis.set_facecolor(BACKGROUND)
    time_axis = np.arange(len(signal)) * 0.01
    axis.plot(time_axis, signal, color="#4dd0e1", lw=1.0)
    axis.axvspan(0, input_end, color=INPUT_COLOR, alpha=0.08)
    axis.axvspan(input_end, output_end, color=OUTPUT_COLOR, alpha=0.08)
    axis.axvline(input_end, color=MUTED, lw=0.8, ls=":")
    axis.set_xlim(0, end * 1.02)
    axis.set_xlabel("time (s; 1 tick = 10 ms)", color=MUTED, fontsize=9)
    axis.set_ylabel("population mean", color=MUTED, fontsize=8)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")
    axis.set_title("population activity on the same timeline", color=TEXT, fontsize=9)

    figure.suptitle(
        f"Dialogue test · prompt={record['prompt']!r} · output={record['output']!r}",
        color=TEXT,
        fontsize=15,
        x=0.05,
        ha="left",
        y=0.955,
        fontproperties=CJK,
    )
    figure.text(
        0.05,
        0.912,
        f"{len(record['input_labels'])} input tokens · "
        f"{len(record['output_labels'])} output tokens · "
        f"prefill {record['prefill_seconds']:.2f} s · "
        f"output latency {np.mean(record['latencies']):.2f} s/token · "
        "rate neurons, not EEG",
        color=MUTED,
        fontsize=9,
        fontproperties=CJK,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/cortex_stage3/final.pt")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output-dir", default="llm/results/cortex")
    args = parser.parse_args()
    prompts = [
        ("hello", "你好"),
        ("capital", "中国的首都是哪里？"),
        ("math", "2+3等于几？"),
    ]
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ExperimentConfig.from_dict(payload["config"])
    tokenizer = build_tokenizer(config.data, config.backbone)
    model = CortexSandwichLM(config, tokenizer.vocab_size, tokenizer=tokenizer)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.cuda().eval()
    records = []
    for name, prompt in prompts:
        record_data = record(model, tokenizer, prompt, args.max_new_tokens)
        path = Path(args.output_dir) / f"dialogue_spectrogram_{name}.png"
        draw(record_data, path)
        signal = record_data.pop("signal")
        record_data["signal_length"] = int(len(signal))
        record_data["activity_mean"] = float(np.mean(signal))
        records.append(record_data)
        print(f"{name}: {record_data['prompt']!r} -> {record_data['output']!r}")
    (Path(args.output_dir) / "dialogue_spectrograms.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
