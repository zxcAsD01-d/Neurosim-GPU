#!/usr/bin/env python3
"""Record the cortical sheet while generating tokens and align it to them.

The current ``generate`` path recomputes the prefix on every step. For each
generated token we therefore keep only the last input token's ticks: these are
the brain updates that immediately precede the next token. This is a runtime
trace, not a continuous-state KV-cache trace.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from neurosim_gpu.cortex.checkpoint import load_checkpoint  # noqa: E402
from neurosim_gpu.cortex.config import ExperimentConfig  # noqa: E402
from neurosim_gpu.cortex.model import CortexSandwichLM  # noqa: E402
from neurosim_gpu.cortex.tokenization import build_tokenizer  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BACKGROUND = "#0b0f14"
TEXT = "#f1f3f5"
MUTED = "#adb5bd"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/cortex_stage3/final.pt")
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--output", default="llm/results/cortex/runtime_brainwave_tokens.png"
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

    prompt_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device="cuda"
    )
    generated = prompt_ids
    records = []
    region = model.brain.region.detach().cpu().numpy()
    n_regions = int(config.brain.regions)
    for step in range(args.max_new_tokens):
        model.brain.trace = []
        begin = time.perf_counter()
        with torch.no_grad():
            output = model(generated)
            logits = output.logits[:, -1].float()
            token = int(logits.argmax(dim=-1))
        latency = time.perf_counter() - begin
        trace = torch.stack(model.brain.trace).float().cpu().numpy()
        model.brain.trace = None
        ticks = max(1, min(2, trace.shape[0]))
        recent = trace[-ticks:]
        state_mean = recent.mean(axis=0)
        region_mean = [
            float(state_mean[region == index].mean())
            if np.any(region == index)
            else 0.0
            for index in range(n_regions)
        ]
        records.append(
            {
                "step": step,
                "token_id": token,
                "token": tokenizer.decode([token]),
                "activity_mean": float(state_mean.mean()),
                "activity_std": float(state_mean.std()),
                "activity_max": float(state_mean.max()),
                "region_mean": region_mean,
                "latency_seconds": latency,
                "logit_max": float(logits.max()),
            }
        )
        generated = torch.cat(
            [generated, torch.tensor([[token]], device="cuda")], dim=1
        )
        if token == tokenizer.eos_token_id:
            break

    # Recompute token strings from cumulative decoding. Byte-level tokens can
    # decode to a replacement glyph in isolation, while the cumulative prefix
    # is valid UTF-8.
    generated_ids = generated[0, prompt_ids.shape[1] :].tolist()
    previous_text = ""
    for index, token_id in enumerate(generated_ids[: len(records)]):
        current_text = tokenizer.decode(generated_ids[: index + 1])
        records[index]["token"] = current_text[len(previous_text) :]
        previous_text = current_text
    generated_text = previous_text

    plot_tokens = [f"t{index}" for index in range(len(records))]
    tokens = plot_tokens
    means = np.array([item["activity_mean"] for item in records])
    stds = np.array([item["activity_std"] for item in records])
    regions = np.array([item["region_mean"] for item in records]).T
    steps = np.arange(len(records))

    figure = plt.figure(figsize=(18, 9), dpi=170)
    figure.patch.set_facecolor(BACKGROUND)
    grid = figure.add_gridspec(
        2, 2, left=0.06, right=0.96, top=0.86, bottom=0.14,
        width_ratios=[1.0, 1.35], hspace=0.36, wspace=0.18,
    )

    # Token strip colored by activity.
    axis = figure.add_subplot(grid[0, 0])
    axis.set_facecolor(BACKGROUND)
    norm = TwoSlopeNorm(
        vmin=min(means.min(), 0.0),
        vcenter=max(means.mean(), 1e-6),
        vmax=means.max() + 1e-6,
    )
    for index, (token, value) in enumerate(zip(tokens, means)):
        axis.add_patch(
            plt.Rectangle(
                (index - 0.45, 0),
                0.9,
                1,
                facecolor=plt.cm.coolwarm(norm(value)),
                edgecolor="#2b3038",
                linewidth=0.5,
            )
        )
        axis.text(
            index,
            0.5,
            token.replace("\n", " ")[:6],
            ha="center",
            va="center",
            color="#0b0f14",
            fontsize=7,
            rotation=45,
        )
    axis.set_xlim(-0.6, len(tokens) - 0.4)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        "generated tokens (color = population activity)",
        color=TEXT,
        fontsize=11,
        pad=8,
    )

    # Region x token heatmap.
    axis = figure.add_subplot(grid[0, 1])
    axis.set_facecolor(BACKGROUND)
    image = axis.imshow(
        regions,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    axis.set_yticks(range(n_regions))
    axis.set_yticklabels([f"R{i}" for i in range(n_regions)])
    axis.set_xticks(steps)
    axis.set_xticklabels(
        [token for token in plot_tokens],
        rotation=65,
        ha="right",
        fontsize=6.5,
    )
    axis.tick_params(colors=MUTED, labelsize=7)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    colorbar.ax.tick_params(colors=MUTED, labelsize=7)
    colorbar.outline.set_edgecolor("#2b3038")
    axis.set_title(
        "brainwave → token alignment (8 regions × generated tokens)",
        color=TEXT,
        fontsize=11,
        pad=8,
    )

    # Population activity line.
    axis = figure.add_subplot(grid[1, 0])
    axis.set_facecolor(BACKGROUND)
    axis.plot(steps, means, marker="o", color="#4dd0e1", lw=1.2, label="mean")
    axis.fill_between(
        steps, means - stds, means + stds, color="#4dd0e1", alpha=0.15
    )
    axis.set_xlabel("generated token", color=MUTED, fontsize=8)
    axis.set_ylabel("activity", color=MUTED, fontsize=8)
    axis.tick_params(colors=MUTED, labelsize=7)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=7)
    axis.set_title(
        "population mean ± std at each generated token",
        color=TEXT,
        fontsize=11,
        pad=8,
    )

    # Summary.
    axis = figure.add_subplot(grid[1, 1])
    axis.set_facecolor(BACKGROUND)
    axis.axis("off")
    total_latency = sum(item["latency_seconds"] for item in records)
    lines = [
        f"checkpoint       {args.checkpoint}",
        f"prompt           {args.prompt!r}",
        f"generated tokens {len(records)}",
        f"ticks/token      {config.brain.ticks_per_token}",
        f"mean activity    {means.mean():.4f}",
        f"activity range   {means.min():.4f} – {means.max():.4f}",
        f"latency/token    {total_latency / max(len(records), 1):.3f} s",
        f"total latency    {total_latency:.2f} s",
        "",
        "output:",
        repr(generated_text),
        "",
        "caveat: this is a rate-neuron population",
        "trace, not EEG. The current generate path",
        "recomputes the prefix each step; only the",
        "last input token's ticks are shown.",
    ]
    axis.text(
        0.0,
        1.0,
        "\n".join(lines),
        color=TEXT,
        fontsize=8.5,
        va="top",
    )
    axis.set_title("runtime summary", color=TEXT, fontsize=11, pad=8)

    figure.suptitle(
        "Runtime brainwave aligned to generated tokens · no spikes, no EEG",
        color=TEXT,
        fontsize=16,
        x=0.06,
        ha="left",
        y=0.96,
    )
    figure.text(
        0.06,
        0.92,
        "population activity of the 1024-neuron sheet, captured immediately "
        "before each generated token",
        color=MUTED,
        fontsize=9,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.1
    )
    plt.close(figure)

    report = {
        "checkpoint": args.checkpoint,
        "prompt": args.prompt,
        "output": generated_text,
        "ticks_per_token": int(config.brain.ticks_per_token),
        "records": records,
        "mean_activity": float(means.mean()),
        "latency_per_token": total_latency / max(len(records), 1),
        "caveat": (
            "rate-neuron population trace; not EEG; prefix is recomputed "
            "for every generated token"
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("prompt", "output", "mean_activity", "latency_per_token")}, ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
