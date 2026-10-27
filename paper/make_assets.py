#!/usr/bin/env python3
"""Generate the LaTeX tables and pgfplots figures from the committed results.

Every number in the paper comes from either:

* ``results/raw/*.json`` (the experiment grid), or
* ``paper/data/gpu_*.csv`` (the accepted GPU numbers transcribed from
  ``docs/results.md``, with hardware recorded).

Running this script is part of the reproducibility claim: it must be possible to
regenerate the paper's assets from the repository alone.

    python paper/make_assets.py
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
DATA = ROOT / "paper" / "data"
TABLES = ROOT / "paper" / "tables"
FIGURES = ROOT / "paper" / "figures"


def load_runs(prefix: str) -> list[dict]:
    runs = []
    for path in sorted(RAW.glob(f"{prefix}*.json")):
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# MNIST model comparison
# ---------------------------------------------------------------------------


def model_table() -> str:
    runs = load_runs("mnist_models_")
    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        cfg = run["config"]
        if cfg["model"] == "mlp":
            key = f"MLP, $h={cfg['hidden']}$"
        elif cfg["model"] == "sparse_brain":
            key = "Sparse brain ($N{=}256$, $d{=}8$, $T{=}8$)"
        elif cfg["model"] == "dense_rnn":
            key = "Dense RNN ($N{=}256$, $T{=}8$)"
        else:
            key = "Logistic regression"
        groups[key].append(run)

    rows = []
    order = [
        "Logistic regression",
        "MLP, $h=6$",
        "MLP, $h=64$",
        "Sparse brain ($N{=}256$, $d{=}8$, $T{=}8$)",
        "Dense RNN ($N{=}256$, $T{=}8$)",
    ]
    for key in order:
        group = groups.get(key, [])
        if not group:
            continue
        acc = [r["metrics"]["test_accuracy"] for r in group]
        params = [r["metrics"]["trainable_params"] for r in group]
        acc_mean, acc_std = mean_std(acc)
        params_mean, _ = mean_std([float(p) for p in params])
        edges = [r["metrics"].get("edges") for r in group if r["metrics"].get("edges")]
        edge_text = f"{statistics.fmean(edges):,.0f}" if edges else "--"
        rows.append(
            f"{key} & {fmt(acc_mean * 100, 1)}\\,$\\pm$\\,{fmt(acc_std * 100, 1)} & "
            f"{params_mean:,.0f} & {edge_text} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{MNIST test accuracy (10{,}000 training samples, 10 epochs, 3 seeds; "
        "mean $\\pm$ population standard deviation). ``Params'' counts trainable "
        "scalars; ``synapses'' counts coalesced edges of the fixed topology.}\n"
        "\\label{tab:models}\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Model & Test accuracy (\\%) & Params & Synapses \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# Emitted-event threshold trade-off
# ---------------------------------------------------------------------------


def theta_table() -> str:
    runs = []
    for pattern in ("mnist_models_", "mnist_degree_", "mnist_neurons_",
                    "mnist_steps_", "mnist_data_", "mnist_variants_"):
        runs.extend(load_runs(pattern))
    deltas: dict[float, list[float]] = defaultdict(list)
    active: dict[float, list[float]] = defaultdict(list)
    accuracy: dict[float, list[float]] = defaultdict(list)
    for run in runs:
        for row in run["metrics"].get("activity", []) or []:
            theta = round(float(row["theta"]), 3)
            deltas[theta].append(float(row["accuracy_delta_vs_matrix"]))
            active[theta].append(float(row["active_edge_fraction"]))
            accuracy[theta].append(float(row["accuracy"]))
    lines = []
    for theta in sorted(deltas):
        delta_mean, delta_std = mean_std(deltas[theta])
        active_mean, _ = mean_std(active[theta])
        acc_mean, acc_std = mean_std(accuracy[theta])
        lines.append(
            f"{theta:.2f} & {fmt(acc_mean * 100, 2)}\\,$\\pm$\\,{fmt(acc_std * 100, 2)} & "
            f"{delta_mean * 100:+.2f} & {fmt(active_mean * 100, 1)} \\\\"
        )
    body = "\n".join(lines)
    return (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{Emitted-event threshold on MNIST, aggregated over "
        f"{len(accuracy[0.0])} trained sparse-brain configurations. Accuracy is measured "
        "through the persistent-current runtime, so $\\theta>0$ is the bounded-error "
        "path; $\\Delta$ is relative to the same run's matrix-path accuracy.}\n"
        "\\label{tab:theta}\n"
        "\\begin{tabular}{rrrr}\n\\toprule\n"
        "$\\theta$ & Accuracy (\\%) & $\\Delta$ (pp) & Active edges (\\%) \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# Cart-pole tables
# ---------------------------------------------------------------------------


def cartpole_table() -> str:
    rows = []
    specs = [
        ("cartpole_expert_", "lqr", "nominal", "LQR"),
        ("cartpole_bc_", "mlp", "nominal", "MLP (BC)"),
        ("cartpole_bc_", "brain", "nominal", "Sparse brain (BC)"),
        ("cartpole_hard_", "lqr", "hard", "LQR"),
        ("cartpole_hard_", "mlp", "hard", "MLP (BC)"),
        ("cartpole_hard_", "brain", "hard", "Sparse brain (BC)"),
    ]
    for prefix, policy, protocol, label in specs:
        group = [r for r in load_runs(prefix) if r["config"].get("policy") == policy]
        if not group:
            continue
        survival = [r["metrics"]["survival_mean"] for r in group]
        success = [r["metrics"]["success_rate"] for r in group]
        angle = [r["metrics"]["angle_rms"] for r in group]
        bc = [r["metrics"].get("bc_accuracy") for r in group if r["metrics"].get("bc_accuracy")]
        s_mean, _ = mean_std(survival)
        succ_mean, _ = mean_std(success)
        a_mean, _ = mean_std(angle)
        bc_text = f"{fmt(statistics.fmean(bc) * 100, 1)}" if bc else "--"
        rows.append(
            f"{label} & {protocol} & {fmt(s_mean, 1)} & {fmt(succ_mean * 100, 0)} & "
            f"{fmt(a_mean, 2)} & {bc_text} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{Cart-pole closed-loop evaluation (100 episodes, 3 seeds). "
        "Survival is out of 200 steps; success requires $\\geq 195$ steps. "
        "Angle RMS is in degrees. BC accuracy is the held-out imitation accuracy.}\n"
        "\\label{tab:cartpole}\n"
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Policy & Protocol & Survival & Success (\\%) & Angle RMS & BC (\\%) \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def robustness_table() -> str:
    runs = load_runs("cartpole_robustness_")
    cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        policy = run["config"]["policy"]
        for row in run["metrics"].get("conditions", []) or []:
            cells[row["condition"]][policy].append(float(row["survival_mean"]))
    order = [
        "nominal", "impulse", "noise=0.1", "noise=0.2",
        "delay=1", "delay=2", "delay=4", "delay=6",
        "length_scale=0.5", "length_scale=2", "length_scale=4",
        "mass_scale=0.3", "mass_scale=3",
    ]
    lines = []
    for condition in order:
        if condition not in cells:
            continue
        brain = cells[condition].get("brain", [])
        mlp = cells[condition].get("mlp", [])
        b_mean, _ = mean_std(brain)
        m_mean, _ = mean_std(mlp)
        label = condition.replace("_", "\\_")
        lines.append(f"{label} & {fmt(b_mean, 1)} & {fmt(m_mean, 1)} \\\\")
    body = "\n".join(lines)
    return (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{Robustness grid on the hard protocol (40 episodes per condition, "
        "2 seeds): mean survival steps out of 200. Both policies sit at the ceiling "
        "for most conditions; the failures are informative.}\n"
        "\\label{tab:robustness}\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Condition & Sparse brain & MLP \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def pgf_figure(name: str, caption: str, label: str, body: str) -> str:
    return (
        "\\begin{figure}[t]\n\\centering\n\\begin{tikzpicture}\n"
        f"{body}\n"
        "\\end{tikzpicture}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n"
    )


def fig_params() -> str:
    runs = load_runs("mnist_models_")
    points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for run in runs:
        cfg = run["config"]
        if cfg["model"] == "mlp":
            label = "MLP"
        elif cfg["model"] == "sparse_brain":
            label = "Sparse brain"
        elif cfg["model"] == "dense_rnn":
            label = "Dense RNN"
        else:
            label = "Logistic regression"
        points[label].append(
            (float(run["metrics"]["trainable_params"]), float(run["metrics"]["test_accuracy"]) * 100)
        )
    styles = {
        "Logistic regression": "mark=square*, only marks",
        "MLP": "mark=triangle*, only marks",
        "Sparse brain": "mark=*, only marks",
        "Dense RNN": "mark=diamond*, only marks",
    }
    plots = []
    for label in ("Logistic regression", "MLP", "Sparse brain", "Dense RNN"):
        coords = " ".join(f"({p:.0f},{a:.2f})" for p, a in sorted(points[label]))
        plots.append(f"\\addplot+[{styles[label]}] coordinates {{{coords}}};")
    body = (
        "\\begin{axis}[width=0.62\\textwidth, height=0.4\\textwidth, "
        "xlabel={Trainable scalars}, ylabel={MNIST test accuracy (\\%)}, "
        "xmode=log, grid=both, legend pos=south east, legend cell align=left, "
        "ymin=84, ymax=96]\n"
        + "\n".join(plots)
        + "\n\\legend{Logistic regression, MLP, Sparse brain, Dense RNN}\n\\end{axis}"
    )
    return pgf_figure(
        "params",
        "Parameter efficiency on MNIST (10k training digits, 3 seeds). The sparse "
        "brain sits at 4.6k trainable scalars between the parameter-matched MLP and "
        "the wide baselines; it does not beat a linear classifier.",
        "fig:params",
        body,
    )


def fig_noise() -> str:
    runs = load_runs("mnist_corruption_")
    curves: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        cfg = run["config"]
        key = "Logistic regression"
        if cfg["model"] == "mlp":
            key = f"MLP, $h={cfg['hidden']}$"
        elif cfg["model"] == "sparse_brain":
            key = "Sparse brain"
        for row in run["metrics"].get("corruptions", []) or []:
            condition = row["condition"]
            if not condition.startswith("gaussian_"):
                continue
            sigma = float(condition.split("_")[1])
            curves[key][sigma].append(float(row["accuracy"]) * 100)
    plots = []
    for key in ("Logistic regression", "MLP, $h=6$", "MLP, $h=64$", "Sparse brain"):
        coords = " ".join(
            f"({sigma:.2f},{statistics.fmean(values):.2f})"
            for sigma, values in sorted(curves[key].items())
        )
        plots.append(f"\\addplot+[mark=*] coordinates {{{coords}}};")
    body = (
        "\\begin{axis}[width=0.62\\textwidth, height=0.4\\textwidth, "
        "xlabel={Gaussian noise $\\sigma$}, ylabel={MNIST test accuracy (\\%)}, "
        "grid=both, legend pos=south west, legend cell align=left]\n"
        + "\n".join(plots)
        + "\n\\legend{LogReg, MLP $h{=}6$, MLP $h{=}64$, Sparse brain}\n\\end{axis}"
    )
    return pgf_figure(
        "noise",
        "Accuracy under additive Gaussian input noise (3 seeds). The saturating "
        "sparse core degrades more slowly than the ReLU baselines at high noise.",
        "fig:noise",
        body,
    )


def fig_theta() -> str:
    runs = []
    for pattern in ("mnist_models_", "mnist_degree_", "mnist_neurons_",
                    "mnist_steps_", "mnist_data_", "mnist_variants_"):
        runs.extend(load_runs(pattern))
    active: dict[float, list[float]] = defaultdict(list)
    delta: dict[float, list[float]] = defaultdict(list)
    for run in runs:
        for row in run["metrics"].get("activity", []) or []:
            theta = round(float(row["theta"]), 3)
            active[theta].append(float(row["active_edge_fraction"]) * 100)
            delta[theta].append(float(row["accuracy_delta_vs_matrix"]) * 100)
    coords = " ".join(
        f"({statistics.fmean(active[t]):.2f},{statistics.fmean(delta[t]):.2f})"
        for t in sorted(active)
    )
    body = (
        "\\begin{axis}[width=0.62\\textwidth, height=0.4\\textwidth, "
        "xlabel={Active edges (\\%)}, ylabel={Accuracy change (pp)}, "
        "grid=both, ymin=-1.0, ymax=0.18]\n"
        f"\\addplot+[mark=*, thick] coordinates {{{coords}}};\n"
        "\\end{axis}"
    )
    return pgf_figure(
        "theta",
        "Bounded-error event propagation: accuracy change against the fraction of "
        "edges that emit, averaged over 56 trained sparse-brain configurations. "
        "Propagating one fifth of the edges costs about a tenth of a point.",
        "fig:theta",
        body,
    )


def fig_tick() -> str:
    lines = [
        line
        for line in (DATA / "gpu_core.csv").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    rows = [r for r in csv.DictReader(lines) if int(r["batch"]) == 1]
    raw = " ".join(f"({float(r['n_synapses'])/1e6:.2f},{float(r['raw_spmm_ms']):.3f})" for r in rows)
    fused = " ".join(f"({float(r['n_synapses'])/1e6:.2f},{float(r['fused_tick_ms']):.3f})" for r in rows)
    body = (
        "\\begin{axis}[width=0.62\\textwidth, height=0.4\\textwidth, "
        "xlabel={Synapses $M$ (millions)}, ylabel={Per-tick time (ms)}, "
        "grid=both, xmode=log, legend pos=north west]\n"
        f"\\addplot+[mark=square*, thick] coordinates {{{raw}}};\n"
        f"\\addplot+[mark=*, thick] coordinates {{{fused}}};\n"
        "\\legend{cuSPARSE SpMM, fused neural tick}\n\\end{axis}"
    )
    return pgf_figure(
        "tick",
        "Fixed-topology tick cost against a same-parameter cuSPARSE multiplication "
        "(RTX 5070 Laptop, batch 1). The complete neural update is cheaper than the "
        "library denominator because it fuses the elementwise work.",
        "fig:tick",
        body,
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = {
        TABLES / "models.tex": model_table(),
        TABLES / "theta.tex": theta_table(),
        TABLES / "cartpole.tex": cartpole_table(),
        TABLES / "robustness.tex": robustness_table(),
        FIGURES / "params.tex": fig_params(),
        FIGURES / "noise.tex": fig_noise(),
        FIGURES / "theta_tradeoff.tex": fig_theta(),
        FIGURES / "tick_parity.tex": fig_tick(),
    }
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT)} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
