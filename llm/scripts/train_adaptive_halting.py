#!/usr/bin/env python3
"""Adaptive-halting sequential training for Cortex-LM.

Each token gets up to a large internal budget. A halting head decides when to
stop; training uses soft halting weights and a slowly increasing ponder cost.
The time budget is reduced from 8 to 4 internal calls over the run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from neurosim_gpu.cortex.checkpoint import load_checkpoint
from neurosim_gpu.cortex.config import ExperimentConfig
from neurosim_gpu.cortex.data import build_dataloaders
from neurosim_gpu.cortex.halting import (
    AdaptiveHaltingHead,
    HaltingConfig,
    hard_ticks,
    ponder_cost,
    soft_halting,
)
from neurosim_gpu.cortex.joint import JointConfig, JointObjective
from neurosim_gpu.cortex.model import CortexSandwichLM
from neurosim_gpu.cortex.tokenization import build_tokenizer


def question_ids(tokenizer) -> tuple[int, ...]:
    ids: list[int] = []
    for text in ("？", "?"):
        for token in tokenizer.encode(text):
            if 0 <= token < tokenizer.vocab_size and token not in ids:
                ids.append(int(token))
    return tuple(ids)


@torch.no_grad()
def evaluate(
    model: CortexSandwichLM,
    joint: JointObjective,
    halt: AdaptiveHaltingHead,
    loader,
    tokens: int,
    max_ticks: int,
    batches: int,
) -> dict[str, float]:
    model.eval()
    joint.eval()
    halt.eval()
    total = 0.0
    expected_sum = 0.0
    hard_sum = 0.0
    time_mae = 0.0
    count = 0
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        ids = batch["input_ids"][:1, :tokens].cuda()
        labels = batch["labels"][:1, :tokens].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, _ = model.backbone.encode(ids)
        hidden_parts = []
        state = None
        for t in range(tokens - 1):
            drive = hidden[:, t : t + 1].float()
            readouts = []
            for _ in range(max_ticks):
                brain_out, state, _ = model.brain(drive, state=state)
                readouts.append(brain_out)
            stacked = torch.cat(readouts, dim=1)
            logits = halt.logits(stacked)
            weighted, _, expected = soft_halting(
                stacked, logits, HaltingConfig()
            )
            hidden_parts.append(
                hidden[:, t : t + 1]
                + model.brain_gate.to(hidden.dtype)
                * weighted.to(hidden.dtype)
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded = model.backbone.decode_hidden(
                    torch.cat(hidden_parts, dim=1)
                )
                logits_lm = model.backbone.logits(decoded)[:, -1].float()
            ce = F.cross_entropy(logits_lm, labels[:, t + 1])
            _, metrics = joint(
                weighted,
                ce,
                labels[:, t + 1],
                t / max(tokens - 2, 1),
                int(expected.mean().round().item()),
                max_ticks,
            )
            total += float(ce)
            expected_sum += float(expected.mean())
            hard_sum += float(hard_ticks(logits, 0.5).float().mean())
            time_mae += float(metrics["time_mae"])
            count += 1
    return {
        "lm_loss": total / max(count, 1),
        "expected_ticks": expected_sum / max(count, 1),
        "hard_ticks": hard_sum / max(count, 1),
        "time_mae": time_mae / max(count, 1),
        "tokens": count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/cortex_stage3/final.pt")
    parser.add_argument(
        "--corpus", default="data/minimind/pretrain_hq.jsonl"
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--max-ticks", type=int, default=8)
    parser.add_argument("--min-ticks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="runs/cortex_halting")
    args = parser.parse_args()

    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    config = ExperimentConfig.from_dict(payload["config"])
    config = replace(
        config, data=replace(config.data, corpus=[args.corpus])
    )
    tokenizer = build_tokenizer(config.data, config.backbone)
    model = CortexSandwichLM(
        config, tokenizer.vocab_size, tokenizer=tokenizer
    )
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.cuda()
    model.train()

    qids = question_ids(tokenizer)
    joint = JointObjective(
        1024,
        question_token_ids=qids,
        config=JointConfig(
            time_weight=0.2, question_weight=0.2, compute_weight=0.0
        ),
    ).cuda()
    halt = AdaptiveHaltingHead(1024).cuda()
    with torch.no_grad():
        halt.halt.bias.fill_(-1.5)
    optimizer = torch.optim.AdamW(
        [p for _, p in model.trainable_parameters()]
        + list(joint.parameters())
        + list(halt.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    train_loader, val_loader, _ = build_dataloaders(
        config.data, tokenizer, verbose=False
    )
    before = evaluate(
        model, joint, halt, val_loader, args.tokens, args.max_ticks, 4
    )
    model.train()
    joint.train()
    halt.train()
    iterator = iter(train_loader)
    history = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        ids = batch["input_ids"][:1, : args.tokens].cuda()
        labels = batch["labels"][:1, : args.tokens].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, _ = model.backbone.encode(ids)
        max_ticks = args.max_ticks
        target_ticks = max(
            args.min_ticks,
            round(
                args.max_ticks
                - (args.max_ticks - args.min_ticks)
                * step
                / max(args.steps, 1)
            ),
        )
        ponder_weight = 0.4 * step / max(args.steps, 1)
        budget_weight = 0.5
        hidden_parts = []
        state = None
        total_loss = 0.0
        metrics_sum = {}
        count = 0
        for t in range(args.tokens - 1):
            drive = hidden[:, t : t + 1].float()
            readouts = []
            for _ in range(max_ticks):
                brain_out, state, _ = model.brain(drive, state=state)
                readouts.append(brain_out)
            if (t + 1) % 8 == 0:
                state = state.detach()
            stacked = torch.cat(readouts, dim=1)
            logits = halt.logits(stacked)
            weighted, _, expected = soft_halting(
                stacked, logits, HaltingConfig()
            )
            hidden_parts.append(
                hidden[:, t : t + 1]
                + model.brain_gate.to(hidden.dtype)
                * weighted.to(hidden.dtype)
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded = model.backbone.decode_hidden(
                    torch.cat(hidden_parts, dim=1)
                )
                logits_lm = model.backbone.logits(decoded)[:, -1].float()
            ce = F.cross_entropy(logits_lm, labels[:, t + 1])
            joint_loss, metrics = joint(
                weighted,
                ce,
                labels[:, t + 1],
                t / max(args.tokens - 2, 1),
                int(expected.mean().round().item()),
                max_ticks,
            )
            budget_loss = (
                expected.mean() - target_ticks
            ) ** 2
            total_loss = (
                total_loss
                + joint_loss
                + ponder_weight * ponder_cost(expected, max_ticks)
                + budget_weight * budget_loss
            )
            metrics_sum["budget_loss"] = (
                metrics_sum.get("budget_loss", 0.0) + float(budget_loss)
            )
            for name, value in metrics.items():
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
            metrics_sum["expected_ticks"] = (
                metrics_sum.get("expected_ticks", 0.0)
                + float(expected.mean())
            )
            metrics_sum["hard_ticks"] = (
                metrics_sum.get("hard_ticks", 0.0)
                + float(hard_ticks(logits, 0.5).float().mean())
            )
            count += 1
        loss = total_loss / max(count, 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                p
                for p in list(model.parameters())
                + list(joint.parameters())
                + list(halt.parameters())
                if p.requires_grad
            ],
            1.0,
        )
        optimizer.step()
        record = {
            "step": step,
            "budget": max_ticks,
            "target_ticks": target_ticks,
            "ponder_weight": ponder_weight,
            "loss": float(loss.detach()),
            **{
                name: value / max(count, 1)
                for name, value in metrics_sum.items()
            },
        }
        history.append(record)
        if step % 5 == 0 or step == 1:
            print(
                f"step {step}: budget={max_ticks} loss={record['loss']:.3f} "
                f"expected={record.get('expected_ticks', 0):.2f} "
                f"hard={record.get('hard_ticks', 0):.2f} "
                f"time_mae={record.get('time_mae', 0):.3f}"
            )

    after = evaluate(
        model, joint, halt, val_loader, args.tokens, args.max_ticks, 4
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "time_head": joint.time_head.state_dict(),
            "question_head": joint.question_head.state_dict(),
            "halt_head": halt.state_dict(),
        },
        output / "heads.pt",
    )
    report = {
        "before": before,
        "after": after,
        "history": history,
        "question_token_ids": list(qids),
        "caveat": (
            "soft halting is trained with a ponder cost; question head is a "
            "tendency probe, not a dialogue policy; no human data"
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"before": before, "after": after}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
