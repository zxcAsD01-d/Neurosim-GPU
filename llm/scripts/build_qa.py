#!/usr/bin/env python3
"""Build a small arithmetic/reasoning QA corpus for answer-constrained training.

The corpus is generated locally and is not committed. It has direct arithmetic
questions and two-step CoT questions; the answer target is always deterministic.
Rows are unique by prompt and split once, so test prompts do not appear in train.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def direct(rng: random.Random) -> tuple[str, str, str]:
    kind = rng.choice(["add", "sub", "mul"])
    if kind == "add":
        a, b = rng.randint(1, 40), rng.randint(1, 40)
        prompt = f"问：{a}+{b}等于几？\n答："
        answer = str(a + b)
    elif kind == "sub":
        a, b = rng.randint(1, 40), rng.randint(1, 40)
        a, b = max(a, b), min(a, b)
        prompt = f"问：{a}-{b}等于几？\n答："
        answer = str(a - b)
    else:
        a, b = rng.randint(1, 40), rng.randint(1, 40)
        prompt = f"问：{a}×{b}等于几？\n答："
        answer = str(a * b)
    return prompt, answer, answer


def cot(rng: random.Random) -> tuple[str, str, str]:
    a, b, c = (rng.randint(1, 20) for _ in range(3))
    first = a + b
    total = first + c
    prompt = f"问：{a}+{b}+{c}等于几？\n答："
    answer = f"{a}+{b}={first}，{first}+{c}={total}，所以答案是{total}。"
    return prompt, answer, str(total)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/qa_arith")
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--val", type=int, default=300)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows: dict[str, dict[str, str]] = {}
    attempts = 0
    needed = args.train + args.val + args.test
    while len(rows) < needed and attempts < needed * 20:
        attempts += 1
        prompt, answer, final = (
            cot(rng) if rng.random() < 0.35 else direct(rng)
        )
        rows.setdefault(
            prompt,
            {"prompt": prompt, "answer": answer, "final": final},
        )
    pool = list(rows.values())
    rng.shuffle(pool)
    if len(pool) < needed:
        raise SystemExit(f"only generated {len(pool)} unique rows")
    train = pool[: args.train]
    val = pool[args.train : args.train + args.val]
    test = pool[args.train + args.val : args.train + args.val + args.test]
    for name, split in (("train", train), ("val", val), ("test", test)):
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
