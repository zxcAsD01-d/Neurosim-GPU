#!/usr/bin/env python3
"""Sequential, time-curriculum training for Cortex-LM.

One token is processed at a time. Each token receives a variable number of
internal brain updates, starting with a generous budget and slowly reducing it.
The joint loss combines next-token CE, a time-perception head and a
question-tendency head.
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
    heads: JointObjective,
    loader,
    tokens: int,
    ticks: int,
    thought_steps: int,
    batches: int,
) -> dict[str, float]:
    model.eval()
    heads.eval()
    total = 0.0
    time_mae = 0.0
    question_hits = 0.0
    question_true = 0
    question_pred = 0
    question_tp = 0
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
            for _ in range(ticks):
                brain_out, state, _ = model.brain(drive, state=state)
            for _ in range(thought_steps):
                brain_out, state, _ = model.brain(brain_out, state=state)
            hidden_parts.append(
                hidden[:, t : t + 1]
                + model.brain_gate.to(hidden.dtype)
                * brain_out.to(hidden.dtype)
            )
            history = torch.cat(hidden_parts, dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded = model.backbone.decode_hidden(history)
                logits = model.backbone.logits(decoded)[:, -1].float()
            ce = F.cross_entropy(logits, labels[:, t + 1])
            _, metrics = heads(
                brain_out,
                ce,
                labels[:, t + 1],
                t / max(tokens - 2, 1),
                ticks,
                ticks,
            )
            total += float(ce)
            time_mae += float(metrics["time_mae"])
            question_hits += float(metrics["question_accuracy"])
            with torch.no_grad():
                target = torch.isin(
                    labels[:, t + 1], heads.question_ids
                )
                prediction = (
                    heads.question_head(brain_out[:, -1]).squeeze(-1) > 0
                )
                question_true += int(target.sum())
                question_pred += int(prediction.sum())
                question_tp += int((target & prediction).sum())
            count += 1
    return {
        "lm_loss": total / max(count, 1),
        "time_mae": time_mae / max(count, 1),
        "question_accuracy": question_hits / max(count, 1),
        "question_rate": question_true / max(count, 1),
        "question_precision": question_tp / max(question_pred, 1),
        "question_recall": question_tp / max(question_true, 1),
        "tokens": count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/cortex_stage3/final.pt")
    parser.add_argument(
        "--corpus",
        default="data/minimind/pretrain_hq.jsonl",
        help="JSONL corpus; not shipped with the repo",
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--max-ticks", type=int, default=6)
    parser.add_argument("--min-ticks", type=int, default=2)
    parser.add_argument("--thought-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="runs/cortex_joint")
    args = parser.parse_args()

    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    config = ExperimentConfig.from_dict(payload["config"])
    config = replace(
        config,
        data=replace(config.data, corpus=[args.corpus]),
    )
    tokenizer = build_tokenizer(config.data, config.backbone)
    model = CortexSandwichLM(
        config, tokenizer.vocab_size, tokenizer=tokenizer
    )
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.cuda()
    model.train()

    qids = question_ids(tokenizer)
    heads = JointObjective(
        1024,
        question_token_ids=qids,
        config=JointConfig(
            time_weight=0.2, question_weight=0.2, compute_weight=0.0
        ),
    ).cuda()
    optimizer = torch.optim.AdamW(
        [
            parameter
            for _, parameter in model.trainable_parameters()
        ]
        + list(heads.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    train_loader, val_loader, _ = build_dataloaders(
        config.data, tokenizer, verbose=False
    )
    before_metrics = evaluate(
        model, heads, val_loader, args.tokens, args.min_ticks, args.thought_steps, 4
    )
    model.train()
    heads.train()
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
        ticks = max(
            args.min_ticks,
            round(
                args.max_ticks
                - (args.max_ticks - args.min_ticks)
                * step
                / max(args.steps, 1)
            ),
        )
        hidden_parts = []
        state = None
        total_loss = 0.0
        metrics_sum = {}
        count = 0
        for t in range(args.tokens - 1):
            drive = hidden[:, t : t + 1].float()
            for _ in range(ticks):
                brain_out, state, _ = model.brain(drive, state=state)
            for _ in range(args.thought_steps):
                brain_out, state, _ = model.brain(brain_out, state=state)
            if (t + 1) % 8 == 0:
                state = state.detach()
            hidden_parts.append(
                hidden[:, t : t + 1]
                + model.brain_gate.to(hidden.dtype)
                * brain_out.to(hidden.dtype)
            )
            history_hidden = torch.cat(hidden_parts, dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded = model.backbone.decode_hidden(history_hidden)
                logits = model.backbone.logits(decoded)[:, -1].float()
            ce = F.cross_entropy(logits, labels[:, t + 1])
            joint, metrics = heads(
                brain_out,
                ce,
                labels[:, t + 1],
                t / max(args.tokens - 2, 1),
                ticks,
                args.max_ticks,
            )
            compute_weight = 0.5 * step / max(args.steps, 1)
            total_loss = total_loss + joint + compute_weight * metrics["compute"]
            for name, value in metrics.items():
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
            count += 1
        loss = total_loss / max(count, 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                p
                for p in list(model.parameters()) + list(heads.parameters())
                if p.requires_grad
            ],
            1.0,
        )
        optimizer.step()
        record = {
            "step": step,
            "ticks": ticks,
            "loss": float(loss.detach()),
            **{
                name: value / max(count, 1)
                for name, value in metrics_sum.items()
            },
        }
        history.append(record)
        if step % 5 == 0 or step == 1:
            print(
                f"step {step}: ticks={ticks} loss={record['loss']:.4f} "
                f"lm={record.get('lm', 0):.3f} time={record.get('time', 0):.3f} "
                f"q={record.get('question', 0):.3f}"
            )

    metrics = {
        "before": before_metrics,
        "after": evaluate(
            model, heads, val_loader, args.tokens, args.min_ticks, args.thought_steps, 4
        ),
        "history": history,
        "question_token_ids": list(qids),
        "checkpoint": args.checkpoint,
        "corpus": "local JSONL (not committed)",
        "caveat": (
            "question head is a tendency probe on question-mark targets, "
            "not a dialogue policy; time head predicts sequence position"
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "time_head": heads.time_head.state_dict(),
            "question_head": heads.question_head.state_dict(),
            "config": {
                "time_weight": heads.config.time_weight,
                "question_weight": heads.config.question_weight,
            },
        },
        output / "heads.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
