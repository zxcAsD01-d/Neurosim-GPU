#!/usr/bin/env python3
"""Diagnose where the Cortex-LM language pipeline loses information.

For each checkpoint it reports:
* teacher-forced ppl with the full sandwich;
* ppl with the brain removed, with only the brain output, and decoder-only;
* a sparsity sweep (top-k edges per neuron);
* effective rank of the brain readout on one batch;
* open-loop stream samples and simple collapse statistics.

Usage:
    python llm/scripts/diagnose_cortex.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from neurosim_gpu.cortex.checkpoint import load_checkpoint
from neurosim_gpu.cortex.config import ExperimentConfig
from neurosim_gpu.cortex.data import build_dataloaders
from neurosim_gpu.cortex.model import CortexSandwichLM
from neurosim_gpu.cortex.spatial import PlateGeometry, sparsify_weights
from neurosim_gpu.cortex.tokenization import build_tokenizer


def load_model(
    checkpoint: Path, device: torch.device
) -> tuple[CortexSandwichLM, object, ExperimentConfig]:
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    config = ExperimentConfig.from_dict(payload["config"])
    tokenizer = build_tokenizer(config.data, config.backbone)
    model = CortexSandwichLM(config, tokenizer.vocab_size, tokenizer=tokenizer)
    load_checkpoint(checkpoint, model, map_location="cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, config


@torch.no_grad()
def variant_loss(
    model: CortexSandwichLM,
    batches,
    mode: str,
    amp: str,
) -> float:
    total = 0.0
    tokens = 0
    dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[amp]
    enabled = dtype is not None
    device = next(model.parameters()).device
    for input_ids, labels in batches:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.autocast("cuda", dtype=dtype, enabled=enabled):
            hidden, context = model.backbone.encode(input_ids)
            brain_out = None
            if mode != "decoder_only":
                brain_out, _, _ = model.brain(hidden.float())
                brain_out = brain_out.to(hidden.dtype)
            if mode == "full":
                hidden = hidden + model.brain_gate * brain_out
            elif mode == "no_brain":
                pass
            elif mode == "brain_only":
                hidden = model.brain_gate * brain_out
            elif mode == "decoder_only":
                hidden = torch.zeros_like(hidden)
            else:
                raise ValueError(mode)
            logits = model.backbone.logits(
                model.backbone.decode(hidden, context)
            )
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )
        total += float(loss) * labels.numel()
        tokens += labels.numel()
    return total / tokens


@torch.no_grad()
def brain_rank(
    model: CortexSandwichLM,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    input_ids, _ = batch
    device = next(model.parameters()).device
    hidden, _ = model.backbone.encode(input_ids.to(device))
    brain_out, state, _ = model.brain(hidden.float())
    flat = brain_out.reshape(-1, brain_out.shape[-1]).float()
    flat = flat - flat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(flat)
    energy = singular.pow(2)
    probability = energy / energy.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    effective_rank = math.exp(float(entropy))
    participation = float(
        energy.sum().pow(2) / energy.pow(2).sum().clamp_min(1e-12)
    )
    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation,
        "singular_max": float(singular[0]),
        "state_mean": float(state.mean()),
        "state_std": float(state.std()),
        "state_saturation": float((state > 0.99).float().mean()),
    }


@torch.no_grad()
def sparsity_curve(
    model: CortexSandwichLM,
    batches,
    amp: str,
    values: list[int],
) -> dict[str, float]:
    if model.brain is None or not hasattr(model.brain, "recurrent"):
        return {}
    weights = model.brain.recurrent.detach().clone()
    positions = getattr(
        model.brain,
        "positions",
        PlateGeometry(weights.shape[0]).sample_positions(seed=7),
    ).to(weights.device)
    result: dict[str, float] = {}
    for edges in values:
        if edges >= weights.shape[1]:
            model.brain.recurrent.data.copy_(weights)
        else:
            mask = sparsify_weights(
                weights,
                positions,
                edges_per_neuron=edges,
                distance_decay=5.0,
            )
            model.brain.recurrent.data.copy_(weights * mask)
        result[str(edges)] = variant_loss(model, batches, "full", amp)
    model.brain.recurrent.data.copy_(weights)
    return result


@torch.no_grad()
def stream_stats(
    model: CortexSandwichLM,
    tokenizer,
    batches,
    device: torch.device,
) -> dict[str, object]:
    samples = []
    collapse = []
    for input_ids, _ in batches:
        for row in input_ids[:2]:
            prompt = row[:16].unsqueeze(0).to(device)
            generated = model.stream_generate(
                prompt,
                max_new_tokens=12,
                temperature=0.8,
                top_p=0.95,
            )
            new_ids = generated[0, prompt.shape[1] :].tolist()
            samples.append(tokenizer.decode(new_ids))
            if new_ids:
                counts = torch.bincount(
                    torch.tensor(new_ids), minlength=tokenizer.vocab_size
                )
                top = float(counts.max()) / len(new_ids)
                unique = len(set(new_ids)) / len(new_ids)
                collapse.append(
                    {"top_token_fraction": top, "unique_token_fraction": unique}
                )
    return {
        "samples": samples,
        "collapse": {
            "top_token_fraction_mean": sum(
                item["top_token_fraction"] for item in collapse
            )
            / max(len(collapse), 1),
            "unique_token_fraction_mean": sum(
                item["unique_token_fraction"] for item in collapse
            )
            / max(len(collapse), 1),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "runs/cortex_long_lora/final.pt",
            "runs/cortex_spatial_train/final.pt",
        ],
    )
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--sweep-batches", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="llm/results/cortex/diagnostics.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("diagnostics require CUDA for reasonable runtime")
    device = torch.device(args.device)
    report: dict[str, object] = {}

    for checkpoint_text in args.checkpoints:
        checkpoint = Path(checkpoint_text)
        print(f"\n=== {checkpoint} ===")
        model, tokenizer, config = load_model(checkpoint, device)
        train_loader, val_loader, _ = build_dataloaders(
            config.data, tokenizer, verbose=False
        )
        batches = []
        for index, batch in enumerate(val_loader):
            if index >= args.batches:
                break
            batches.append((batch["input_ids"], batch["labels"]))
        sweep_batches = batches[: args.sweep_batches]
        normal = variant_loss(model, batches, "full", config.train.amp)
        no_brain = variant_loss(model, batches, "no_brain", config.train.amp)
        brain_only = variant_loss(
            model, batches, "brain_only", config.train.amp
        )
        decoder_only = variant_loss(
            model, batches, "decoder_only", config.train.amp
        )
        rank = brain_rank(model, batches[0])
        curve = sparsity_curve(
            model,
            sweep_batches,
            config.train.amp,
            [16, 32, 64, 128, 256, 512, 1024],
        )
        streams = stream_stats(model, tokenizer, batches[:2], device)
        entry = {
            "checkpoint": str(checkpoint),
            "step": int(
                torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )["step"]
            ),
            "ppl": {
                "full": math.exp(normal),
                "no_brain": math.exp(no_brain),
                "brain_only": math.exp(brain_only),
                "decoder_only": math.exp(decoder_only),
            },
            "loss": {
                "full": normal,
                "no_brain": no_brain,
                "brain_only": brain_only,
                "decoder_only": decoder_only,
            },
            "brain_rank": rank,
            "sparsity_ppl": {
                key: math.exp(value) for key, value in curve.items()
            },
            "open_loop": streams,
        }
        report[str(checkpoint)] = entry
        print(
            f"full ppl={entry['ppl']['full']:.2f} "
            f"no-brain={entry['ppl']['no_brain']:.2f} "
            f"brain-only={entry['ppl']['brain_only']:.2f} "
            f"decoder-only={entry['ppl']['decoder_only']:.2f}"
        )
        print(f"effective rank={rank['effective_rank']:.1f}")
        print(f"sparsity ppl={entry['sparsity_ppl']}")
        print(f"stream collapse={streams['collapse']}")
        del model
        torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
