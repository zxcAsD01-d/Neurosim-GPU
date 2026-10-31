"""Perplexity and sample generation for a Cortex-LM checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import torch

from .checkpoint import checkpoint_config, load_checkpoint
from .data import build_dataloaders
from .model import CortexSandwichLM
from .tokenization import build_tokenizer
from .utils import autocast_dtype, resolve_device


@torch.no_grad()
def evaluate_loss(
    model: CortexSandwichLM,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    max_batches: int | None = None,
    amp: str = "none",
) -> dict[str, float]:
    """Average token cross entropy over ``loader``."""

    model.eval()
    dtype = autocast_dtype(amp)
    enabled = dtype is not None and device.type == "cuda"
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(
            device_type=device.type, dtype=dtype, enabled=enabled
        ):
            output = model(input_ids, labels=labels)
        assert output.loss is not None
        tokens = labels.numel()
        total_loss += float(output.loss.detach()) * tokens
        total_tokens += tokens
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if total_tokens == 0:
        raise ValueError("evaluation loader produced no tokens")
    loss = total_loss / total_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "tokens": float(total_tokens),
        "batches": float(batches),
    }


@torch.no_grad()
def generate_samples(
    model: CortexSandwichLM,
    tokenizer,
    prompts: list[str],
    *,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stream: bool = False,
) -> list[dict[str, object]]:
    """One prompt at a time; batching padded prompts is not implemented."""

    results: list[dict[str, object]] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [tokenizer.eos_token_id]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        if stream:
            generated = model.stream_generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = generated[0, input_ids.shape[1] :].tolist()
        results.append(
            {
                "prompt": prompt,
                "completion": tokenizer.decode(new_ids),
                "new_tokens": len(new_ids),
                "mode": "open-loop-stream" if stream else "teacher-forced-prefix",
            }
        )
    return results


def _load_model_for_eval(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[CortexSandwichLM, object, int]:
    payload = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    config = checkpoint_config(payload)
    tokenizer = build_tokenizer(config.data, config.backbone)
    model = CortexSandwichLM(
        config, tokenizer.vocab_size, tokenizer=tokenizer
    )
    load_checkpoint(checkpoint_path, model, map_location="cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, int(payload["step"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity and generate samples from Cortex-LM"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", action="append", default=[])
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="open-loop readout: prefill once, then let the brain drive the back plate",
    )
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, tokenizer, step = _load_model_for_eval(args.checkpoint, device)
    config = model.config

    if args.corpus:
        config = replace(
            config, data=replace(config.data, corpus=tuple(args.corpus))
        )
    _, loader, meta = build_dataloaders(
        replace(config.data, batch_size=config.data.eval_batch_size),
        tokenizer,
        verbose=True,
    )
    metrics = evaluate_loss(
        model,
        loader,
        device,
        max_batches=args.max_batches,
        amp=config.train.amp,
    )
    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "corpus": {
            "documents": meta.n_documents,
            "val_blocks": meta.n_val_blocks,
            "block_size": meta.block_size,
        },
        "metrics": metrics,
    }
    if args.prompt:
        report["samples"] = generate_samples(
            model,
            tokenizer,
            args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stream=args.stream,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
