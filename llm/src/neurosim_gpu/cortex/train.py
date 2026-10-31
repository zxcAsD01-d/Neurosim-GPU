"""Training and pipeline dry-run entry point.

``--dry-run`` builds the configured model, runs one synthetic forward/backward
and reports parameter groups, gradient norms and peak memory. It deliberately
does **not** call ``optimizer.step()``: the dry run is a wiring check, not a
training run.

A real run needs a corpus::

    python -m neurosim_gpu.cortex.train --config configs/cortex_qwen3_06b.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig, apply_overrides, load_config
from .data import build_dataloaders
from .evaluate import evaluate_loss
from .model import CortexSandwichLM
from .tokenization import build_tokenizer
from .utils import autocast_dtype, format_count, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train (or dry-run) the Cortex-LM sandwich"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--corpus", action="append", default=[])
    parser.add_argument("--resume", default="")
    return parser


def load_experiment(args: argparse.Namespace) -> ExperimentConfig:
    config = load_config(args.config)
    if args.set:
        raw = apply_overrides(config.to_dict(), args.set)
        config = ExperimentConfig.from_dict(raw)
    if args.corpus:
        config = replace(
            config, data=replace(config.data, corpus=tuple(args.corpus))
        )
    if args.output_dir:
        config = replace(
            config, train=replace(config.train, output_dir=args.output_dir)
        )
    if args.max_steps:
        config = replace(
            config, train=replace(config.train, max_steps=args.max_steps)
        )
    if args.device:
        config = replace(
            config, train=replace(config.train, device=args.device)
        )
    if args.resume:
        config = replace(
            config, train=replace(config.train, resume=args.resume)
        )
    return config


def build_optimizer(
    model: CortexSandwichLM,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    brain: list[torch.nn.Parameter] = []
    lora: list[torch.nn.Parameter] = []
    other: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("brain.") or name == "brain_gate":
            brain.append(parameter)
        elif ".lora_" in name:
            lora.append(parameter)
        else:
            other.append(parameter)

    groups = []
    if brain:
        groups.append(
            {
                "params": brain,
                "lr": config.train.brain_lr,
                "weight_decay": 0.0,
                "name": "brain",
            }
        )
    if lora:
        groups.append(
            {
                "params": lora,
                "lr": config.train.lora_lr,
                "weight_decay": config.train.weight_decay,
                "name": "lora",
            }
        )
    if other:
        groups.append(
            {
                "params": other,
                "lr": config.train.lr,
                "weight_decay": config.train.weight_decay,
                "name": "other",
            }
        )
    if not groups:
        raise RuntimeError("model has no trainable parameters")
    return torch.optim.AdamW(groups, betas=config.train.betas)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    total = max(config.train.max_steps, 1)
    warmup = min(config.train.warmup_steps, total)
    floor = config.train.min_lr_ratio

    def schedule(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _gradient_report(model: CortexSandwichLM) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("brain.") or name == "brain_gate":
            group = "brain"
        elif ".lora_" in name:
            group = "lora"
        else:
            group = "other"
        entry = report.setdefault(
            group, {"parameters": 0, "with_grad": 0, "squared_norm": 0.0}
        )
        entry["parameters"] += 1
        if parameter.grad is not None:
            entry["with_grad"] += 1
            entry["squared_norm"] += float(parameter.grad.detach().float().pow(2).sum())
    for entry in report.values():
        entry["grad_norm"] = math.sqrt(entry.pop("squared_norm"))
    return report


def dry_run(config: ExperimentConfig, args: argparse.Namespace) -> int:
    torch.manual_seed(config.train.seed)
    device = resolve_device(config.train.device)
    tokenizer = build_tokenizer(config.data, config.backbone)
    model = CortexSandwichLM(
        config, tokenizer.vocab_size, tokenizer=tokenizer
    )
    model.to(device)
    model.train()

    descriptor = model.describe()
    group_counts = descriptor["groups"]
    print(f"device: {device}")
    print(
        "parameters: "
        f"{format_count(int(descriptor['total_parameters']))} total, "
        f"{format_count(int(descriptor['trainable_parameters']))} trainable "
        f"(brain {format_count(int(group_counts['brain']))}, "
        f"lora {format_count(int(group_counts['lora']))}, "
        f"other {format_count(int(group_counts['other']))}, "
        f"frozen {format_count(int(group_counts['frozen']))})"
    )
    if "brain" in descriptor:
        brain = descriptor["brain"]
        print(
            f"brain: {brain['n_neurons']} neurons, {brain['n_edges']} edges, "
            f"{brain['ticks_per_token']} ticks/token"
        )

    batch = 2
    tokens = min(32, config.data.block_size)
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)
    input_ids = torch.randint(
        0,
        tokenizer.vocab_size,
        (batch, tokens),
        generator=generator,
    ).to(device)
    labels = input_ids.clone()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    dtype = autocast_dtype(config.train.amp)
    enabled = dtype is not None and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
        output = model(input_ids, labels=labels)
        if output.loss is None:
            raise RuntimeError("model produced no loss")
        loss = output.loss
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite dry-run loss: {loss.item()}")

    gradients = _gradient_report(model)
    for name, entry in gradients.items():
        if entry["with_grad"] == 0:
            raise RuntimeError(f"no gradients reached group {name!r}")
        if not math.isfinite(entry["grad_norm"]):
            raise RuntimeError(f"non-finite gradients in group {name!r}")

    # Build the optimiser and schedule without stepping them: this checks the
    # parameter groups and warmup/cosine wiring while staying a dry run.
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    optimizer_groups = [
        {
            "name": group.get("name", "group"),
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "parameters": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]
    scheduler_lrs = [
        float(value) for value in scheduler.get_last_lr()
    ]

    peak = None
    if device.type == "cuda":
        peak = float(torch.cuda.max_memory_allocated(device)) / 1e6
        print(f"peak allocated: {peak:.0f} MB")

    stats = {
        key: float(value.detach().float().mean())
        for key, value in output.stats.items()
    }
    report: dict[str, Any] = {
        "name": config.name,
        "device": str(device),
        "batch": batch,
        "tokens": tokens,
        "loss": float(output.loss.detach()),
        "seconds": elapsed,
        "peak_memory_mb": peak,
        "parameters": descriptor,
        "gradients": gradients,
        "optimizer": optimizer_groups,
        "initial_lrs": scheduler_lrs,
        "stats": stats,
        "note": (
            "dry run: forward/backward only, no optimizer step; wall time "
            "includes any other workloads on the GPU"
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


def _compatible(old: ExperimentConfig, new: ExperimentConfig) -> bool:
    return (
        old.name == new.name
        and old.backbone.model_id == new.backbone.model_id
        and old.backbone.kind == new.backbone.kind
        and old.backbone.encoder_layers == new.backbone.encoder_layers
        and old.backbone.decoder_layers == new.backbone.decoder_layers
        and old.brain.n_neurons == new.brain.n_neurons
        and old.brain.weight_mode == new.brain.weight_mode
        and old.brain.regions == new.brain.regions
        and old.brain.init_radius == new.brain.init_radius
        and old.model.use_brain == new.model.use_brain
        and old.model.brain_gate_init == new.model.brain_gate_init
        and old.model.dropout == new.model.dropout
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _generate_sample(
    model: CortexSandwichLM,
    tokenizer,
    config: ExperimentConfig,
    device: torch.device,
) -> str:
    prompt_ids = tokenizer.encode(config.train.sample_prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids,
        max_new_tokens=config.train.sample_tokens,
        temperature=config.train.sample_temperature,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = generated[0, input_ids.shape[1] :].tolist()
    return tokenizer.decode(new_ids)


def train(config: ExperimentConfig) -> int:
    device = resolve_device(config.train.device)
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    torch.manual_seed(config.train.seed)
    tokenizer = build_tokenizer(config.data, config.backbone)
    train_loader, val_loader, meta = build_dataloaders(
        config.data, tokenizer, verbose=True
    )
    model = CortexSandwichLM(
        config, tokenizer.vocab_size, tokenizer=tokenizer
    )
    model.to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.train.amp == "fp16" and device.type == "cuda",
    )

    step = 0
    best_val = float("inf")
    resume_path = Path(config.train.resume) if config.train.resume else None
    if resume_path is not None:
        previous_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        previous = ExperimentConfig.from_dict(previous_payload["config"])
        if not _compatible(previous, config):
            raise ValueError(
                "checkpoint config is incompatible with the current config"
            )
        payload = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            map_location=device,
            restore_rng=True,
        )
        step = int(payload["step"])
        best_val = float(payload.get("best_val") or float("inf"))
        if config.brain.spatial and model.brain is not None:
            model.brain.refresh_spatial_mask()
        print(f"resumed from {resume_path} at step {step}")

    dtype = autocast_dtype(config.train.amp)
    amp_enabled = dtype is not None and device.type == "cuda"
    if dtype is not None and not amp_enabled:
        print("note: AMP is CUDA-only; running float32 on this device")

    log_path = output_dir / "train.jsonl"
    print(
        f"training {config.name}: {meta.n_train_blocks} train blocks, "
        f"{meta.n_val_blocks} val blocks, block={meta.block_size}, "
        f"batch={config.data.batch_size}, accum={config.train.grad_accum}"
    )
    model.train()
    tokens_seen = 0
    started = time.perf_counter()
    last_log = started
    last_tokens = 0
    pending = 0

    while step < config.train.max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=amp_enabled
            ):
                output = model(input_ids, labels=labels)
                if output.loss is None:
                    raise RuntimeError("model produced no loss")
                loss_value = float(output.loss.detach())
                loss = output.loss / config.train.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            scaler.scale(loss).backward()
            pending += 1
            tokens_seen += int(input_ids.numel())

            if pending < config.train.grad_accum:
                continue

            scaler.unscale_(optimizer)
            if config.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    config.train.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pending = 0
            step += 1

            adapt_stats: dict[str, float] = {}
            if (
                config.brain.spatial
                and config.model.use_brain
                and model.brain is not None
            ):
                adapt_stats = model.brain.spatial_adapt(step)
            if (
                config.model.use_brain
                and model.brain is not None
                and config.brain.consolidation_rate > 0
                and step % config.brain.consolidation_interval == 0
            ):
                baseline = float(model.brain.dopamine_baseline)
                dopamine = math.tanh(
                    (loss_value - baseline)
                    / config.brain.dopamine_scale
                )
                model.brain.dopamine_baseline.mul_(0.95).add_(
                    0.05 * loss_value
                )
                adapt_stats.update(model.brain.consolidate(dopamine))
            optimizer.zero_grad(set_to_none=True)

            if step % config.train.log_interval == 0:
                now = time.perf_counter()
                elapsed_log = max(now - last_log, 1e-6)
                throughput = (tokens_seen - last_tokens) / elapsed_log
                last_log, last_tokens = now, tokens_seen
                record = {
                    "step": step,
                    "loss": float(output.loss.detach()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "tokens": tokens_seen,
                    "tokens_per_second": throughput,
                    "elapsed": now - started,
                    "stats": {
                        key: float(value.detach().float().mean())
                        for key, value in output.stats.items()
                    },
                    "adaptation": adapt_stats,
                }
                print(
                    f"step {step}: loss {record['loss']:.4f} "
                    f"lr {record['lr']:.2e} "
                    f"{throughput:.0f} tok/s"
                )
                _append_jsonl(log_path, record)

            if step % config.train.eval_interval == 0:
                metrics = evaluate_loss(
                    model,
                    val_loader,
                    device,
                    max_batches=config.train.eval_batches,
                    amp=config.train.amp,
                )
                model.train()
                print(
                    f"eval step {step}: loss {metrics['loss']:.4f} "
                    f"ppl {metrics['perplexity']:.2f}"
                )
                _append_jsonl(
                    log_path, {"step": step, "eval": metrics}
                )
                if metrics["loss"] < best_val:
                    best_val = metrics["loss"]
                    save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        step,
                        config,
                        metrics=metrics,
                        best_val=best_val,
                    )

            if step % config.train.save_interval == 0:
                save_checkpoint(
                    output_dir / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    step,
                    config,
                    metrics={"loss": float(output.loss.detach())},
                    best_val=best_val,
                )

            if (
                config.train.sample_interval
                and step % config.train.sample_interval == 0
            ):
                sample = _generate_sample(model, tokenizer, config, device)
                model.train()
                print(f"sample step {step}: {sample!r}")
                _append_jsonl(
                    log_path, {"step": step, "sample": sample}
                )

            if step >= config.train.max_steps:
                break

    if pending:
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            step,
            config,
            metrics={},
            best_val=best_val,
        )
    save_checkpoint(
        output_dir / "final.pt",
        model,
        optimizer,
        scheduler,
        step,
        config,
        metrics={},
        best_val=best_val,
    )
    print(f"wrote {output_dir / 'final.pt'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment(args)
    if args.dry_run:
        return dry_run(config, args)
    return train(config)


if __name__ == "__main__":
    raise SystemExit(main())
