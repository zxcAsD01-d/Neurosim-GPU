#!/usr/bin/env python3
"""Answer-constrained arithmetic QA fine-tuning for Cortex-LM.

The model is trained only on answer tokens. Evaluation uses greedy decoding
(no sampling), exact numeric matching, and measures wall-clock latency. A
test-time tick sweep checks whether more internal compute improves accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from neurosim_gpu.cortex.checkpoint import load_checkpoint, save_checkpoint
from neurosim_gpu.cortex.config import ExperimentConfig
from neurosim_gpu.cortex.model import CortexSandwichLM
from neurosim_gpu.cortex.tokenization import build_tokenizer


class QAData:
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
        prompts = [self.rows[i]["prompt"] for i in indices]
        answers = [self.rows[i]["answer"] for i in indices]
        prompt_ids = [self.tokenizer.encode(p) for p in prompts]
        answer_ids = [
            self.tokenizer.encode(a) + [self.tokenizer.eos_token_id]
            for a in answers
        ]
        sequences = []
        labels = []
        for prompt, answer in zip(prompt_ids, answer_ids):
            sequence = (prompt + answer)[: self.max_length]
            label = [-100] * len(sequence)
            prompt_length = len(prompt)
            if prompt_length > 0 and prompt_length < len(sequence):
                # Logits at prompt_length - 1 predict the first answer token.
                label[prompt_length - 1 : len(sequence) - 1] = sequence[
                    prompt_length:
                ]
            sequences.append(sequence)
            labels.append(label)
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), width), pad_id, dtype=torch.long
        )
        attention = torch.zeros((len(sequences), width), dtype=torch.long)
        label_tensor = torch.full(
            (len(sequences), width), -100, dtype=torch.long
        )
        for row, (sequence, label) in enumerate(zip(sequences, labels)):
            input_ids[row, : len(sequence)] = torch.tensor(sequence)
            attention[row, : len(sequence)] = 1
            label_tensor[row, : len(label)] = torch.tensor(label)
        return input_ids, attention, label_tensor


def extract_final(text: str) -> str:
    match = re.search(r"答案是\s*(-?\d+)", text)
    if match:
        return match.group(1)
    numbers = re.findall(r"-?\d+", text)
    return numbers[-1] if numbers else ""


@torch.no_grad()
def evaluate(
    model: CortexSandwichLM,
    data: QAData,
    tokenizer,
    limit: int,
    ticks: int | None,
    max_new_tokens: int,
) -> dict[str, float | list[dict[str, str]]]:
    model.eval()
    original_ticks = model.brain.config.ticks_per_token
    if ticks is not None:
        model.brain.config = replace(
            model.brain.config, ticks_per_token=ticks
        )
    correct = 0
    total = 0
    elapsed = 0.0
    generated_tokens = 0
    examples = []
    started = time.perf_counter()
    for row in data.rows[:limit]:
        prompt = row["prompt"]
        ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device="cuda"
        )
        begin = time.perf_counter()
        generated = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        elapsed += time.perf_counter() - begin
        new_ids = generated[0, ids.shape[1] :].tolist()
        generated_tokens += len(new_ids)
        text = tokenizer.decode(new_ids)
        prediction = extract_final(text)
        ok = prediction == str(row["final"])
        correct += int(ok)
        total += 1
        if len(examples) < 12:
            examples.append(
                {
                    "prompt": prompt,
                    "expected": str(row["final"]),
                    "prediction": prediction,
                    "text": text[:120],
                    "correct": bool(ok),
                }
            )
    wall = time.perf_counter() - started
    model.brain.config = replace(
        model.brain.config, ticks_per_token=original_ticks
    )
    model.train()
    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "latency_seconds": elapsed / max(total, 1),
        "wall_seconds": wall,
        "tokens_per_second": generated_tokens / max(elapsed, 1e-9),
        "ticks": ticks if ticks is not None else original_ticks,
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/cortex_stage3/final.pt")
    parser.add_argument("--data", default="data/qa_arith")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", default="runs/cortex_qa")
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
    train = QAData(Path(args.data) / "train.jsonl", tokenizer, args.max_length)
    val = QAData(Path(args.data) / "val.jsonl", tokenizer, args.max_length)
    test = QAData(Path(args.data) / "test.jsonl", tokenizer, args.max_length)
    optimizer = torch.optim.AdamW(
        [p for _, p in model.trainable_parameters()],
        lr=args.lr,
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(0)
    before = evaluate(
        model, val, tokenizer, args.eval_limit, 2, args.max_new_tokens
    )
    print(
        f"before: acc={before['accuracy']:.3f} "
        f"lat={before['latency_seconds']:.3f}s"
    )
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            0, len(train), (args.batch,), generator=generator
        ).tolist()
        input_ids, attention, labels = train.batch(indices, pad_id)
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
        if step % 10 == 0 or step == 1:
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
        if step % 50 == 0:
            metrics = evaluate(
                model,
                val,
                tokenizer,
                min(50, args.eval_limit),
                2,
                args.max_new_tokens,
            )
            print(
                f"eval {step}: acc={metrics['accuracy']:.3f} "
                f"lat={metrics['latency_seconds']:.3f}s"
            )
    after = evaluate(
        model, val, tokenizer, args.eval_limit, 2, args.max_new_tokens
    )
    test_metrics = evaluate(
        model, test, tokenizer, args.eval_limit, 2, args.max_new_tokens
    )
    tick_sweep = {}
    for ticks in (1, 2, 4, 8):
        tick_sweep[str(ticks)] = evaluate(
            model,
            test,
            tokenizer,
            min(40, args.eval_limit),
            ticks,
            args.max_new_tokens,
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
        metrics={"val_accuracy": after["accuracy"]},
    )
    report = {
        "checkpoint": args.checkpoint,
        "before": before,
        "after": after,
        "test": test_metrics,
        "tick_sweep": {
            key: {
                k: v
                for k, v in value.items()
                if k != "examples"
            }
            for key, value in tick_sweep.items()
        },
        "history": history,
        "steps": args.steps,
        "data": "synthetic arithmetic QA; generated locally, not committed",
        "caveat": (
            "greedy numeric answers; test prompts are unique but the template "
            "space is synthetic and narrow"
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "before": {k: before[k] for k in ("accuracy", "latency_seconds")},
                "after": {k: after[k] for k in ("accuracy", "latency_seconds")},
                "test": {k: test_metrics[k] for k in ("accuracy", "latency_seconds")},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
