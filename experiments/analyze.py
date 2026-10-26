"""Aggregate ``results/raw/*.json`` into markdown tables for the report.

    python experiments/analyze.py            # writes results/TABLES.md
    python experiments/analyze.py --stdout   # also print to the console

Runs are identified by their suite prefix (``mnist_degree_...`` etc.), so every
number in the report can be traced back to one raw payload and one suite.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from typing import Any, Iterable

from common import RAW, RESULTS, ensure_dir

SUITE_PREFIXES = (
    "mnist_models",
    "mnist_degree",
    "mnist_neurons",
    "mnist_steps",
    "mnist_data",
    "mnist_variants",
    "mnist_corruption",
    "cartpole_expert",
    "cartpole_hard",
    "cartpole_bc",
    "cartpole_robustness",
    "cartpole_online",
)


def load_runs() -> list[dict[str, Any]]:
    runs = []
    if not RAW.exists():
        return runs
    for path in sorted(RAW.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "metrics" not in payload:
            continue
        payload["_path"] = path.name
        runs.append(payload)
    return runs


def suite_of(run: dict[str, Any]) -> str:
    run_id = str(run.get("run_id", ""))
    for prefix in SUITE_PREFIXES:
        if run_id.startswith(prefix + "_"):
            return prefix
    return "other"


def stat(values: Iterable[float]) -> tuple[float, float, int]:
    collected = [float(v) for v in values]
    if not collected:
        return float("nan"), float("nan"), 0
    mean = statistics.fmean(collected)
    std = statistics.pstdev(collected) if len(collected) > 1 else 0.0
    return mean, std, len(collected)


def fmt(values: Iterable[float], digits: int = 4) -> str:
    mean, std, count = stat(values)
    if count == 0:
        return "—"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def metric_values(runs: list[dict[str, Any]], name: str) -> list[float]:
    values = []
    for run in runs:
        value = run.get("metrics", {}).get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def grouped_by(runs: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        buckets[run.get("config", {}).get(key)].append(run)
    return buckets


def mnist_sections(runs: list[dict[str, Any]]) -> list[str]:
    mnist = [
        r
        for r in runs
        if r.get("experiment") == "mnist" and suite_of(r) != "other"
    ]
    if not mnist:
        return []
    sections: list[str] = []

    def model_label(run: dict[str, Any]) -> str:
        cfg = run["config"]
        model = cfg.get("model")
        if model == "mlp":
            return f"mlp (h={cfg.get('hidden')})"
        if model == "sparse_brain":
            return f"sparse brain (N={cfg.get('neurons')}, d={cfg.get('degree')}, T={cfg.get('steps')})"
        if model == "dense_rnn":
            return f"dense RNN (N={cfg.get('neurons')}, T={cfg.get('steps')})"
        return str(model)

    models = [r for r in mnist if suite_of(r) == "mnist_models"]
    if models:
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for run in models:
            cfg = run["config"]
            buckets[
                (
                    str(cfg.get("model")),
                    cfg.get("hidden"),
                    cfg.get("neurons"),
                    cfg.get("steps"),
                )
            ].append(run)
        order = ("logreg", "mlp", "sparse_brain", "dense_rnn")
        rows = []
        for key in sorted(
            buckets,
            key=lambda item: (
                order.index(item[0]) if item[0] in order else 99,
                item[1] or 0,
            ),
        ):
            bucket = buckets[key]
            label = model_label(bucket[0])
            acc = metric_values(bucket, "test_accuracy")
            params = metric_values(bucket, "trainable_params")
            edges = metric_values(bucket, "edges")
            seconds = metric_values(bucket, "train_seconds")
            latency = metric_values(bucket, "p50_ms")
            rows.append(
                [
                    label,
                    fmt(acc),
                    str(int(stat(params)[0])) if params else "—",
                    str(int(stat(edges)[0])) if edges else "—",
                    f"{stat(seconds)[0]:.1f}" if seconds else "—",
                    f"{stat(latency)[0]*1000:.0f} µs" if latency else "—",
                ]
            )
        sections.append(
            "### MNIST: dense baselines versus sparse brain (10k train, 10 epochs, 3 seeds)\n\n"
            + markdown_table(
                ["model", "test accuracy", "trainable params", "synapses", "train s", "p50 latency"],
                rows,
            )
            + "\n"
        )

    def sweep(suite: str, key: str, title: str, extra: str = "") -> None:
        group = [r for r in mnist if suite_of(r) == suite]
        if not group:
            return
        buckets = grouped_by(group, key)
        rows = []
        for value in sorted(buckets, key=lambda v: (v is None, v)):
            bucket = buckets[value]
            acc = metric_values(bucket, "test_accuracy")
            params = metric_values(bucket, "trainable_params")
            edges = metric_values(bucket, "edges")
            seconds = metric_values(bucket, "train_seconds")
            latency = metric_values(bucket, "p50_ms")
            row = [str(value), fmt(acc)]
            if extra == "edges":
                row.append(f"{stat(edges)[0]:.0f}" if edges else "—")
                row.append(f"{stat(params)[0]:.0f}" if params else "—")
            if extra == "latency":
                row.append(f"{stat(latency)[0]*1000:.0f} µs" if latency else "—")
            if extra == "time":
                row.append(f"{stat(seconds)[0]:.1f}" if seconds else "—")
            rows.append(row)
        headers = [key] + ["test accuracy"]
        if extra == "edges":
            headers += ["synapses", "trainable params"]
        if extra == "latency":
            headers += ["p50 latency"]
        if extra == "time":
            headers += ["train s"]
        sections.append(f"### {title}\n\n" + markdown_table(headers, rows) + "\n")

    sweep("mnist_degree", "degree", "MNIST: synapses per neuron (N=256, T=8)", "edges")
    sweep("mnist_neurons", "neurons", "MNIST: neuron count (d=8, T=8)", "latency")
    sweep("mnist_steps", "steps", "MNIST: recurrent ticks (N=256, d=8)", "time")
    sweep("mnist_data", "train_size", "MNIST: training-set size (N=256, d=8, T=8)")

    variants = [r for r in mnist if suite_of(r) == "mnist_variants"]
    if variants:
        def variant_label(run: dict[str, Any]) -> str:
            cfg = run["config"]
            bits = []
            if cfg.get("steps") == 0:
                bits.append("no recurrence (T=0)")
            if cfg.get("learn_sensory"):
                bits.append("learned sensory fanout")
            if cfg.get("fanout") == 8:
                bits.append("sensory fanout 8")
            if cfg.get("alpha") == 0.5:
                bits.append("leak alpha 0.5")
            if cfg.get("alpha") == 0.9:
                bits.append("leak alpha 0.9")
            if cfg.get("weight_scale") == 0.35:
                bits.append("weight scale 0.35")
            return ", ".join(bits) or "baseline"

        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in variants:
            buckets[variant_label(run)].append(run)
        rows = []
        for label in sorted(buckets):
            bucket = buckets[label]
            rows.append(
                [
                    label,
                    fmt(metric_values(bucket, "test_accuracy")),
                    f"{stat(metric_values(bucket, 'trainable_params'))[0]:.0f}",
                    f"{stat(metric_values(bucket, 'train_seconds'))[0]:.1f}",
                ]
            )
        sections.append(
            "### MNIST: architecture variants (N=256, d=8, 10k train)\n\n"
            + markdown_table(["variant", "test accuracy", "trainable params", "train s"], rows)
            + "\n"
        )

    corrupt = [r for r in mnist if suite_of(r) == "mnist_corruption"]
    if corrupt:
        cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for run in corrupt:
            label = model_label(run)
            for row in run.get("metrics", {}).get("corruptions", []) or []:
                cells[str(row["condition"])][label].append(float(row["accuracy"]))
        labels = sorted({label for bucket in cells.values() for label in bucket})
        rows = []
        for condition in sorted(cells):
            rows.append([condition] + [fmt(cells[condition].get(label, [])) for label in labels])
        sections.append(
            "### MNIST: distribution shift (test split, 3 seeds)\n\n"
            + markdown_table(["corruption"] + labels, rows)
            + "\n"
        )

    theta_acc: dict[float, list[float]] = defaultdict(list)
    theta_delta: dict[float, list[float]] = defaultdict(list)
    theta_edges: dict[float, list[float]] = defaultdict(list)
    theta_neurons: dict[float, list[float]] = defaultdict(list)
    for run in mnist:
        for row in run.get("metrics", {}).get("activity", []) or []:
            theta = round(float(row["theta"]), 4)
            theta_acc[theta].append(float(row["accuracy"]))
            theta_delta[theta].append(float(row["accuracy_delta_vs_matrix"]))
            theta_edges[theta].append(float(row["active_edge_fraction"]))
            theta_neurons[theta].append(float(row["active_neuron_fraction"]))
    if theta_acc:
        rows = []
        for theta in sorted(theta_acc):
            rows.append(
                [
                    f"{theta:g}",
                    fmt(theta_acc[theta]),
                    f"{stat(theta_delta[theta])[0]*100:+.2f} pp",
                    f"{stat(theta_edges[theta])[0]*100:.1f}%",
                    f"{stat(theta_neurons[theta])[0]*100:.1f}%",
                    str(len(theta_acc[theta])),
                ]
            )
        sections.append(
            "### MNIST: emitted-event threshold (`HybridPELSRuntime`, every brain run)\n\n"
            "Threshold on the per-tick state delta. Accuracy is measured through the "
            "persistent-current runtime, so `theta > 0` is the bounded-error path.\n\n"
            + markdown_table(
                ["theta", "test accuracy", "Δ vs matrix", "active edges", "active neurons", "runs"],
                rows,
            )
            + "\n"
        )
    return sections


def cartpole_sections(runs: list[dict[str, Any]]) -> list[str]:
    cart = [
        r
        for r in runs
        if r.get("experiment") == "cartpole" and suite_of(r) != "other"
    ]
    if not cart:
        return []
    sections: list[str] = []

    closed = [r for r in cart if suite_of(r) in ("cartpole_expert", "cartpole_bc", "cartpole_hard")]
    if closed:
        buckets: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
        for run in closed:
            key = (str(run["config"].get("policy")), bool(run["config"].get("hard")))
            buckets[key].append(run)
        rows = []
        for key in sorted(buckets, key=lambda item: (item[0], item[1])):
            bucket = buckets[key]
            policy, hard = key
            survival = metric_values(bucket, "survival_mean")
            success = metric_values(bucket, "success_rate")
            angle = metric_values(bucket, "angle_rms")
            switch = metric_values(bucket, "switch_rate")
            bc = metric_values(bucket, "bc_accuracy")
            ablation = metric_values(bucket, "recurrent_ablation_survival")
            rows.append(
                [
                    f"{policy} ({'hard' if hard else 'nominal'})",
                    fmt(survival, digits=1),
                    fmt(success),
                    fmt(angle, digits=2),
                    fmt(switch, digits=3),
                    fmt(bc) if bc else "—",
                    fmt(ablation, digits=1) if ablation else "—",
                ]
            )
        sections.append(
            "### Cart-pole: closed-loop stability (100 episodes, 3 seeds)\n\n"
            + markdown_table(
                [
                    "policy / protocol",
                    "survival steps / 200",
                    "success rate",
                    "|angle| RMS (deg)",
                    "action switch rate",
                    "BC accuracy",
                    "recurrent ablation",
                ],
                rows,
            )
            + "\n"
        )

    robust = [r for r in cart if suite_of(r) == "cartpole_robustness"]
    if robust:
        conditions: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for run in robust:
            policy = str(run["config"].get("policy"))
            for row in run.get("metrics", {}).get("conditions", []) or []:
                label = str(row.get("condition") or row.get("name") or "nominal")
                value = row.get("survival_mean")
                if isinstance(value, (int, float)):
                    conditions[label][policy].append(float(value))
        if conditions:
            policies = sorted({p for bucket in conditions.values() for p in bucket})
            rows = []
            for label in sorted(conditions):
                row = [label]
                for policy in policies:
                    row.append(fmt(conditions[label].get(policy, []), digits=1))
                rows.append(row)
            sections.append(
                "### Cart-pole: robustness grid (survival steps)\n\n"
                + markdown_table(["condition"] + policies, rows)
                + "\n"
            )

    online = [r for r in cart if suite_of(r) == "cartpole_online"]
    if online:
        buckets = grouped_by(online, "variant")
        rows = []
        for variant in sorted(buckets, key=str):
            bucket = buckets[variant]
            rows.append(
                [
                    str(variant),
                    fmt(metric_values(bucket, "survival_pre"), digits=1),
                    fmt(metric_values(bucket, "survival_post"), digits=1),
                    fmt(metric_values(bucket, "adaptation_gain"), digits=1),
                    fmt(metric_values(bucket, "recovery_episode"), digits=1),
                    fmt(metric_values(bucket, "weight_change_l2"), digits=3),
                ]
            )
        sections.append(
            "### Cart-pole: online three-factor plasticity after an unannounced plant change\n\n"
            + markdown_table(
                [
                    "variant",
                    "survival pre",
                    "survival post",
                    "adaptation gain",
                    "recovery episode",
                    "‖ΔW‖",
                ],
                rows,
            )
            + "\n"
        )
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    runs = load_runs()
    ensure_dir(RESULTS)
    if not runs:
        raise SystemExit("no raw results found; run experiments/run_suite.py first")

    suites: dict[str, int] = defaultdict(int)
    for run in runs:
        suites[suite_of(run)] += 1

    parts = ["# Aggregated experiment tables\n"]
    parts.append(f"Raw runs: **{len(runs)}** — " + ", ".join(f"`{k}`: {v}" for k, v in sorted(suites.items())) + "\n")
    parts.extend(mnist_sections(runs))
    parts.extend(cartpole_sections(runs))
    parts.append("<details><summary>Raw run inventory</summary>\n")
    parts.append("\n".join(f"- `{run['_path']}`" for run in runs))
    parts.append("\n</details>\n")

    text = "\n".join(parts)
    output = RESULTS / "TABLES.md"
    output.write_text(text, encoding="utf-8")
    if args.stdout:
        print(text)
    print(f"wrote {output} from {len(runs)} runs")


if __name__ == "__main__":
    main()
