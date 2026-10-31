#!/usr/bin/env python3
"""Supervised CoT fine-tuning on the local R1-distillation corpus.

Target format:
    用户：<question>
    助手：<reasoning>
    最终答案：<answer>

The prompt is masked; only reasoning + final answer are supervised. Evaluation
is greedy (output constrained) and reports format compliance, answer overlap,
arithmetic exact match and wall-clock latency.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
from pathlib import Path

import torch

from neurosim_gpu.cortex.checkpoint import load_checkpoint, save_checkpoint
from neurosim_gpu.cortex.config import ExperimentConfig
from neurosim_gpu.cortex.model import CortexSandwichLM
from neurosim_gpu.cortex.tokenization import build_tokenizer


class CoTData:
    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        self.rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def batch(self, indices: list[int], pad_id: int):
        sequences = []
        labels = []
        for index in indices:
            row = self.rows[index]
            prompt_ids = self.tokenizer.encode(row["prompt"])
            target_ids = self.tokenizer.encode(row["target"]) + [
                self.tokenizer.eos_token_id
            ]
            sequence = (prompt_ids + target_ids)[: self.max_length]
            label = [-100] * len(sequence)
            prompt_length = len(prompt_ids)
            if prompt_length > 0 and prompt_length < len(sequence):
                label[prompt_length - 1 : len(sequence) - 1] = sequence[
                    prompt_length:
                ]
            sequences.append(sequence)
            labels.append(label)
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(sequences), width), dtype=torch.long)
        label_tensor = torch.full((len(sequences), width), -100, dtype=torch.long)
        for row, (sequence, label) in enumerate(zip(sequences, labels)):
            input_ids[row, : len(sequence)] = torch.tensor(sequence)
            attention[row, : len(sequence)] = 1
            label_tensor[row, : len(label)] = torch.tensor(label)
        return input_ids, attention, label_tensor


def final_answer(text: str) -> str:
    marker = "最终答案："
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def numeric(text: str) -> str:
    numbers = re.findall(r"-?\d+", text)
    return numbers[-1] if numbers else ""


def load_arith(path: Path, limit: int):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= limit:
            break
    return rows


@torch.no_grad()
def evaluate(
    model: CortexSandwichLM,
    tokenizer,
    cot: CoTData,
    arith: list[dict[str, str]],
    *,
    cot_limit: int,
    max_new_tokens: int,
) -> dict[str, object]:
    model.eval()
    cot_rows = cot.rows[:cot_limit]
    format_hits = 0
    similarities = 0.0
    cot_latency = 0.0
    examples = []
    for row in cot_rows:
        ids = torch.tensor(
            [tokenizer.encode(row["prompt"])], dtype=torch.long, device="cuda"
        )
        begin = time.perf_counter()
        generated = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        cot_latency += time.perf_counter() - begin
        text = tokenizer.decode(generated[0, ids.shape[1] :].tolist())
        has_final = "最终答案：" in text
        format_hits += int(has_final)
        predicted = final_answer(text)
        similarities += difflib.SequenceMatcher(
            None, predicted, row["answer"]
        ).ratio()
        if len(examples) < 6:
            examples.append(
                {
                    "prompt": row["prompt"],
                    "reference": row["answer"][:160],
                    "generated": text[:240],
                    "has_final": has_final,
                }
            )
    correct = 0
    arith_latency = 0.0
    arith_examples = []
    for row in arith:
        ids = torch.tensor(
            [tokenizer.encode(row["prompt"])], dtype=torch.long, device="cuda"
        )
        begin = time.perf_counter()
        generated = model.generate(
            ids,
            max_new_tokens=24,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        arith_latency += time.perf_counter() - begin
        text = tokenizer.decode(generated[0, ids.shape[1] :].tolist())
        prediction = numeric(text)
        ok = prediction == str(row["final"])
        correct += int(ok)
        if len(arith_examples) < 6:
            arith_examples.append(
                {
                    "prompt": row["prompt"],
                    "expected": str(row["final"]),
                    "prediction": prediction,
                    "text": text[:120],
                    "correct": bool(ok),
                }
            )
    model.train()
    return {
        "cot_format_rate": format_hits / max(len(cot_rows), 1),
        "cot_similarity": similarities / max(len(cot_rows), 1),
        "cot_latency_seconds": cot_latency / max(len(cot_rows), 1),
        "arith_accuracy": correct / max(len(arith), 1),
        "arith_latency_seconds": arith_latency / max(len(arith), 1),
        "cot_examples": examples,
        "arith_examples": arith_examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/cortex_stage3/final.pt")
    parser.add_argument("--data", default="data/cot_r1")
    parser.add_argument("--arith", default="data/qa_arith/test.jsonl")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-cot", type=int, default=8)
    parser.add_argument("--eval-arith", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="runs/cortex_cot")
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
    model.cuda()
    model.train()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    cot = CoTData(Path(args.data) / "train.jsonl", tokenizer, args.max_length)
    val = CoTData(Path(args.data) / "val.jsonl", tokenizer, args.max_length)
    arith = load_arith(Path(args.arith), args.eval_arith)
    optimizer = torch.optim.AdamW(
        [p for _, p in model.trainable_parameters()],
        lr=args.lr,
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(0)
    before = evaluate(
        model,
        tokenizer,
        val,
        arith,
        cot_limit=args.eval_cot,
        max_new_tokens=args.max_new_tokens,
    )
    print(
        f"before: cot_format={before['cot_format_rate']:.2f} "
        f"sim={before['cot_similarity']:.3f} "
        f"arith={before['arith_accuracy']:.2f}"
    )
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            0, len(cot), (args.batch,), generator=generator
        ).tolist()
        input_ids, attention, labels = cot.batch(indices, pad_id)
        input_ids = input_ids.cuda()
        attention = attention.cuda()
        labels = labels.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids, attention_mask=attention, labels=labels)
            loss = output.loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        if step % 5 == 0 or step == 1:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "elapsed": time.perf_counter() - started,
            }
            history.append(record)
            print(
                f"step {step}: loss={record['loss']:.4f} "
                f"elapsed={record['elapsed']:.1f}s"
            )
        if step % 30 == 0:
            metrics = evaluate(
                model,
                tokenizer,
                val,
                arith,
                cot_limit=min(4, args.eval_cot),
                max_new_tokens=48,
            )
            print(
                f"eval {step}: format={metrics['cot_format_rate']:.2f} "
                f"sim={metrics['cot_similarity']:.3f} "
                f"arith={metrics['arith_accuracy']:.2f}"
            )
    after = evaluate(
        model,
        tokenizer,
        val,
        arith,
        cot_limit=args.eval_cot,
        max_new_tokens=args.max_new_tokens,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output / "final.pt",
        model,
        optimizer,
        None,
        args.steps,
        config,
        metrics={"cot_similarity": after["cot_similarity"]},
    )
    report = {
        "checkpoint": args.checkpoint,
        "before": before,
        "after": after,
        "history": history,
        "source": "local R1-distillation corpus (not committed)",
        "caveat": (
            "greedy output; CoT target format is synthetic; arithmetic transfer "
            "is a small synthetic test"
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "before": {
                    k: before[k]
                    for k in (
                        "cot_format_rate",
                        "cot_similarity",
                        "arith_accuracy",
                        "cot_latency_seconds",
                    )
                },
                "after": {
                    k: after[k]
                    for k in (
                        "cot_format_rate",
                        "cot_similarity",
                        "arith_accuracy",
                        "cot_latency_seconds",
                    )
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
