#!/usr/bin/env python3
"""Build a small CoT corpus from the local R1-distillation file.

Source (not committed):
  F:/归档/data/train_data/distill_r1_110k.jsonl
Fields: input, reasoning_content, content.

The output uses a deterministic prompt/answer format:
  用户：<input>
  助手：<reasoning_content>
  最终答案：<content>
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="F:/归档/data/train_data/distill_r1_110k.jsonl",
    )
    parser.add_argument("--output", default="data/cot_r1")
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--val", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"source not found: {source}")
    rows = []
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(item.get("input", "")).strip()
            reasoning = str(item.get("reasoning_content", "")).strip()
            answer = str(item.get("content", "")).strip()
            if not prompt or not reasoning or not answer:
                continue
            if not (20 <= len(reasoning) <= 2200):
                continue
            if not (1 <= len(answer) <= 600):
                continue
            if len(prompt) > 400:
                continue
            target = f"{reasoning}\n最终答案：{answer}"
            if len(prompt) + len(target) > 1200:
                continue
            rows.append(
                {
                    "prompt": f"用户：{prompt}\n助手：",
                    "reasoning": reasoning,
                    "answer": answer,
                    "target": target,
                }
            )
            if len(rows) >= args.count * 2:
                break
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.count]
    if len(rows) < args.val + 100:
        raise SystemExit(f"only collected {len(rows)} rows")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    val = rows[: args.val]
    train = rows[args.val :]
    for name, split in (("train", train), ("val", val)):
        path = output / f"{name}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in split)
            + "\n",
            encoding="utf-8",
        )
        print(f"{path}: {len(split)} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
