"""Grid driver for the task-level experiment suites.

Each experiment script is a self-contained CLI that prints one ``RESULT {json}``
line. This driver enumerates the grid, runs the scripts as subprocesses, stores
the raw payloads and writes tidy CSVs for the report.

Usage:
    python experiments/run_suite.py --list
    python experiments/run_suite.py --suite mnist_models
    python experiments/run_suite.py --suite all
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import RAW, RESULTS, ROOT, ensure_dir, percentile, slug

PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PYTHON).exists():  # pragma: no cover - non-Windows fallback
    PYTHON = sys.executable

BASE_MNIST: dict[str, Any] = {
    "epochs": 10,
    "train_size": 10000,
    "test_size": 2000,
    "batch": 64,
    "lr": 2e-3,
}

SUITES: dict[str, dict[str, Any]] = {
    "mnist_models": {
        "script": "experiments/mnist_experiments.py",
        "seeds": [0, 1, 2],
        "configs": [
            {**BASE_MNIST, "model": "logreg"},
            {**BASE_MNIST, "model": "mlp", "hidden": 6},
            {**BASE_MNIST, "model": "mlp", "hidden": 64},
            {**BASE_MNIST, "model": "sparse_brain", "neurons": 256, "degree": 8, "steps": 8},
            {**BASE_MNIST, "model": "dense_rnn", "neurons": 256, "steps": 8},
        ],
    },
    "mnist_degree": {
        "script": "experiments/mnist_experiments.py",
        "seeds": [0, 1],
        "configs": [
            {**BASE_MNIST, "model": "sparse_brain", "neurons": 256, "degree": d, "steps": 8}
            for d in (2, 4, 8, 16, 32)
        ],
    },
    "mnist_neurons": {
        "script": "experiments/mnist_experiments.py",
        "seeds": [0, 1],
        "configs": [
            {**BASE_MNIST, "model": "sparse_brain", "neurons": n, "degree": 8, "steps": 8}
            for n in (64, 128, 256, 512)
        ],
    },
    "mnist_steps": {
        "script": "experiments/mnist_experiments.py",
        "seeds": [0, 1],
        "configs": [
            {**BASE_MNIST, "model": "sparse_brain", "neurons": 256, "degree": 8, "steps": t}
            for t in (1, 2, 4, 8, 16)
        ],
    },
    "mnist_data": {
        "script": "experiments/mnist_experiments.py",
        "seeds": [0, 1],
        "configs": [
            {
                **BASE_MNIST,
                "model": "sparse_brain",
                "neurons": 256,
                "degree": 8,
                "steps": 8,
                "train_size": size,
            }
            for size in (1000, 2500, 5000, 10000, 20000)
        ],
    },
    "mnist_variants": {
        "script": "experiments/mnist_experiments.py",
        "seeds": [0, 1],
        "configs": [
            # No recurrence at all: sparse sensory projection + linear readout.
            {**BASE_MNIST, "model": "sparse_brain", "steps": 0},
            # Train the sensory fanout instead of freezing it.
            {**BASE_MNIST, "model": "sparse_brain", "learn_sensory": True},
            # Denser / stronger sensory coding.
            {**BASE_MNIST, "model": "sparse_brain", "fanout": 8},
            # Leak and recurrent gain sensitivity.
            {**BASE_MNIST, "model": "sparse_brain", "alpha": 0.5},
            {**BASE_MNIST, "model": "sparse_brain", "alpha": 0.9},
            # Initial recurrent weight scale.
            {**BASE_MNIST, "model": "sparse_brain", "weight_scale": 0.35},
        ],
    },
    "mnist_corruption": {
        "script": "experiments/mnist_experiments.py",
        "seeds": [0, 1, 2],
        "configs": [
            {**BASE_MNIST, "model": "logreg", "corruptions": True},
            {**BASE_MNIST, "model": "mlp", "hidden": 6, "corruptions": True},
            {**BASE_MNIST, "model": "mlp", "hidden": 64, "corruptions": True},
            {**BASE_MNIST, "model": "sparse_brain", "corruptions": True},
        ],
    },
    "cartpole_expert": {
        "script": "experiments/cartpole_experiments.py",
        "seeds": [0, 1, 2],
        "configs": [
            {"policy": "lqr", "eval": "closed_loop", "episodes": 100},
        ],
    },
    "cartpole_hard": {
        "script": "experiments/cartpole_experiments.py",
        "seeds": [0, 1, 2],
        "configs": [
            {"policy": "lqr", "eval": "closed_loop", "episodes": 100, "hard": True},
            {"policy": "mlp", "eval": "closed_loop", "episodes": 100, "hard": True},
            {"policy": "brain", "eval": "closed_loop", "episodes": 100, "hard": True},
        ],
    },
    "cartpole_bc": {
        "script": "experiments/cartpole_experiments.py",
        "seeds": [0, 1, 2],
        "configs": [
            {"policy": "mlp", "eval": "closed_loop", "episodes": 100},
            {"policy": "brain", "eval": "closed_loop", "episodes": 100},
        ],
    },
    "cartpole_robustness": {
        "script": "experiments/cartpole_experiments.py",
        "seeds": [0, 1],
        "configs": [
            {"policy": "brain", "eval": "robustness", "episodes": 40, "hard": True},
            {"policy": "mlp", "eval": "robustness", "episodes": 40, "hard": True},
        ],
    },
    "cartpole_online": {
        "script": "experiments/cartpole_experiments.py",
        "seeds": [0, 1, 2],
        "configs": [
            {"policy": "brain", "eval": "online", "episodes": 20, "variant": "frozen"},
            {"policy": "brain", "eval": "online", "episodes": 20, "variant": "online"},
        ],
    },
}


def run_one(script: str, config: dict[str, Any], seed: int, timeout: int = 1800) -> dict[str, Any] | None:
    command = [
        PYTHON,
        script,
        "--config-json",
        json.dumps(config),
        "--seed",
        str(seed),
        "--json",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()[-6:]
        print(f"  FAILED seed={seed} ({elapsed:.0f}s): {' | '.join(tail)}", flush=True)
        return None
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT "):
            payload = json.loads(line[len("RESULT ") :])
            payload["_wall_seconds"] = elapsed
            return payload
    print(f"  NO RESULT seed={seed}: {completed.stdout[-400:]}", flush=True)
    return None


def flatten(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": payload.get("run_id", ""),
        "experiment": payload.get("experiment", ""),
        "seed": payload.get("seed", ""),
        "wall_seconds": round(float(payload.get("_wall_seconds", 0.0)), 3),
    }
    for key, value in payload.get("config", {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[f"cfg_{key}"] = value
    for key, value in payload.get("metrics", {}).items():
        if isinstance(value, (int, float, bool)) and not isinstance(value, bool):
            row[f"m_{key}"] = round(float(value), 6)
        elif isinstance(value, bool):
            row[f"m_{key}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", default=None, help="comma separated seed override")
    args = parser.parse_args()

    if args.list:
        for name, suite in SUITES.items():
            from common import slug as _slug

            seeds = suite["seeds"]
            print(f"{name:22s} {len(suite['configs'])} configs x {len(seeds)} seeds = "
                  f"{len(suite['configs']) * len(seeds):2d} runs  [{','.join(map(str, seeds))}]")
        return

    names = list(SUITES) if args.suite == "all" else [args.suite]
    unknown = [name for name in names if name not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s): {unknown}")

    ensure_dir(RAW)
    all_rows: list[dict[str, Any]] = []
    for name in names:
        suite = SUITES[name]
        script = str(ROOT / suite["script"])
        if not Path(script).exists():
            print(f"SKIP {name}: {suite['script']} not implemented yet", flush=True)
            continue
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else suite["seeds"]
        print(f"== {name}: {len(suite['configs'])} configs x {len(seeds)} seeds ==", flush=True)
        rows: list[dict[str, Any]] = []
        for config in suite["configs"]:
            for seed in seeds:
                label = config.get("model") or config.get("policy")
                print(f"  {label} seed={seed} {json.dumps(config, sort_keys=True)}", flush=True)
                if args.dry_run:
                    continue
                payload = run_one(suite["script"], config, seed)
                if payload is None:
                    continue
                payload.setdefault("run_id", f"{name}_{slug(config)}_seed{seed}")
                payload["run_id"] = f"{name}_{payload['run_id'].split('_seed')[0]}_seed{seed}"
                row = flatten(payload)
                rows.append(row)
                all_rows.append(row)
                (RAW / f"{payload['run_id']}.json").write_text(
                    json.dumps(payload, indent=2, default=str), encoding="utf-8"
                )
        if rows:
            columns = sorted({key for row in rows for key in row})
            path = RESULTS / f"summary_{name}.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  wrote {path.name} ({len(rows)} rows)", flush=True)

    if all_rows:
        columns = sorted({key for row in all_rows for key in row})
        path = RESULTS / "summary_all.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {path} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
